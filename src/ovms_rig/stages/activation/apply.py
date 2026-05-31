"""Apply stage: copy pristine graph.pbtxt to sibling + patch + register.

Steps per model entry in active profile:
1. Resolve target model directory (store/<hf_org>/<hf_repo>).
2. Read pristine graph.pbtxt (never mutated).
3. Create sibling copy graph.<model_name>.pbtxt.
4. Patch sibling copy (device, draft_device, draft_models_path).
5. Merge generation_config.json overrides (read from .orig pristine snapshot).
6. Register endpoint via direct config.json JSON write (not ovms CLI).
7. Cleanup obsolete sibling-graphs from previous activations.

Atomicity: snapshot of config.json is taken before processing begins.
On any error during model processing, apply fails immediately with rollback:
config.json is restored from snapshot, sibling graphs created in this run are deleted.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import (
    ConfigError,
    OvmsConfig,
    load_declaration,
)
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary
from ovms_rig.stages.activation.generation import merge as merge_generation
from ovms_rig.stages.activation.paths import model_dir, relative_posix
from ovms_rig.stages.activation.pbtxt import collect_pbtxt_fields, patch
from ovms_rig.stages.activation.registry import render_config
from ovms_rig.stages.activation import cleanup

logger = logging.getLogger(__name__)

_BUILD_DIR = Path("build")


def run(ctx: dict) -> int:
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")
    dry_run: bool = ctx.get("dry_run", False)
    extras: list[str] = list(ctx.get("extras") or [])

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        decl = load_declaration(config_path, local_path, cli_override=ovms_override)
        ovms = decl.ovms
        local = decl.local
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)

    binary, _src = ovms_binary.resolve(ovms_override, local)
    if binary is None or not binary.is_file():
        logger.error("ovms binary not resolved; run `ovms-rig status` for details")
        return 1

    store = local.models.repository_path

    # Determine active profile and get active models.
    active_profile_name = None
    active_models: set[str] = set()
    for profile_name, profile in ovms.profiles.items():
        if profile.active:
            active_profile_name = profile_name
            active_models = set(profile.models)
            break

    if active_profile_name:
        logger.info("[apply] using profile '%s' with %d model(s)", active_profile_name, len(active_models))
    else:
        logger.info("[apply] no active profile; will write empty mediapipe_config_list")

    # Snapshot state before processing (for rollback on failure).
    config_json_path = store / "config.json" if not dry_run else _BUILD_DIR / "config.json"
    config_json_snapshot: str | None = None
    existing_graphs: set[Path] = set()

    if not dry_run:
        # Snapshot config.json for potential rollback.
        if config_json_path.exists():
            try:
                config_json_snapshot = config_json_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.error("[apply] failed to snapshot config.json: %s", e)
                return 1

        # Record existing sibling-graphs across all repository dirs so
        # rollback preserves graphs from previous activations.
        for identity in ovms.repository.values():
            repo_dir = model_dir(store, identity.hf)
            if repo_dir.is_dir():
                existing_graphs.update(
                    g.resolve() for g in repo_dir.glob("graph.*.pbtxt")
                    if g.name != "graph.pbtxt"
                )

    env = build_env(binary.parent)
    processed_models: list[str] = []

    # Process only models in active profile.
    # Fail-fast: any error triggers immediate rollback and exit.
    for model_name in active_models:
        entry = ovms.models[model_name]
        model_identity = ovms.repository[entry.source]
        target_dir = model_dir(store, model_identity.hf)

        if not target_dir.is_dir():
            logger.error(
                "[apply] %s: model directory not found at %s; run fetch first",
                model_name, target_dir,
            )
            # Fail-fast: rollback and exit.
            if not dry_run:
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
            return 1

        # Plain (non-task) models register via model_config_list: no graph.pbtxt,
        # no sibling graph, no generation_config. Presence of the directory above
        # is the only per-model artifact; rendering happens below.
        if model_identity.task is None:
            logger.info("[apply] %s: plain model (model_config_list), device=%s",
                        model_name, entry.device)
            processed_models.append(model_name)
            continue

        pristine_pbtxt = target_dir / "graph.pbtxt"
        if not pristine_pbtxt.exists():
            logger.error(
                "[apply] %s: graph.pbtxt not found at %s",
                model_name, pristine_pbtxt,
            )
            # Fail-fast: rollback and exit.
            if not dry_run:
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
            return 1

        # Resolve draft path if declared.
        draft_rel: str | None = None
        if entry.draft_model is not None:
            draft_identity = ovms.repository[entry.draft_model]
            draft_dir = model_dir(store, draft_identity.hf)
            draft_rel = relative_posix(target_dir, draft_dir)

        fields = collect_pbtxt_fields(
            entry, draft_rel, cache_dir=local.runtime.cache_dir,
        )

        # Compute destination path for sibling-copy (live or build/).
        # Sibling naming: graph.<model_name>.pbtxt in the same directory as pristine.
        if dry_run:
            sibling_pbtxt = _BUILD_DIR / model_identity.hf / f"graph.{model_name}.pbtxt"
        else:
            sibling_pbtxt = target_dir / f"graph.{model_name}.pbtxt"

        # Read pristine (never mutate).
        try:
            pristine_content = pristine_pbtxt.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("[apply] %s: failed to read pristine graph.pbtxt: %s", model_name, exc)
            # Fail-fast: rollback and exit.
            if not dry_run:
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
            return 1

        # Patch the pristine content (goes to sibling, not to pristine itself).
        try:
            patched_content = patch(pristine_content, fields)
        except (ValueError, OSError) as exc:
            logger.error("[apply] %s: pbtxt patch failed: %s", model_name, exc)
            # Fail-fast: rollback and exit.
            if not dry_run:
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
            return 1

        sibling_pbtxt.parent.mkdir(parents=True, exist_ok=True)
        sibling_pbtxt.write_text(patched_content, encoding="utf-8")
        logger.info("[apply] %s: graph.%s.pbtxt written to %s", model_name, model_name, sibling_pbtxt)

        # Handle generation_config.json if overrides are declared on the entry.
        overrides = entry.generation
        if overrides:
            genconfig_orig = target_dir / "generation_config.json.orig"
            genconfig_live = target_dir / "generation_config.json"

            # For dry-run, log intention without requiring file to exist.
            if dry_run:
                dest_genconfig = _BUILD_DIR / model_identity.hf / "generation_config.json"
                logger.info(
                    "[apply] %s: would read generation overrides from %s",
                    model_name, genconfig_orig,
                )
                dest_genconfig.parent.mkdir(parents=True, exist_ok=True)
                # Write proposed result to build/ (assumes pristine .orig exists in dry-run context).
                if genconfig_orig.exists():
                    try:
                        pristine_text = genconfig_orig.read_text(encoding="utf-8")
                        new_genconfig = merge_generation(pristine_text, overrides)
                        dest_genconfig.write_text(new_genconfig, encoding="utf-8")
                    except (ValueError, OSError) as exc:
                        logger.warning(
                            "[apply] %s: could not preview generation_config merge: %s",
                            model_name, exc,
                        )
                processed_models.append(model_name)
                continue

            # Live run: pristine .orig must exist.
            if not genconfig_orig.exists():
                logger.error(
                    "[apply] %s: generation_config.json.orig not found at %s; "
                    "run fetch or re-fetch the model first",
                    model_name, genconfig_orig,
                )
                # Fail-fast: rollback and exit.
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
                return 1

            # Merge from pristine and write to live.
            try:
                pristine_text = genconfig_orig.read_text(encoding="utf-8")
                new_genconfig = merge_generation(pristine_text, overrides)
            except (ValueError, OSError) as exc:
                logger.error(
                    "[apply] %s: generation_config merge failed: %s",
                    model_name, exc,
                )
                # Fail-fast: rollback and exit.
                _rollback(config_json_path, config_json_snapshot, existing_graphs)
                return 1

            genconfig_live.write_text(new_genconfig, encoding="utf-8")
            logger.info(
                "[apply] %s: generation_config.json written to %s",
                model_name, genconfig_live,
            )

        processed_models.append(model_name)

    # Render config.json from active_models (not processed_models).
    # This guarantees config.json is always complete projection of active profile.
    # Failure recovery is handled by in-memory snapshot and rollback; no disk backup needed.
    if not dry_run:
        config_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Build desired entries from active_models (not processed_models).
    # If a model failed, it won't reach here due to fail-fast above.
    # Mediapipe (LLM) vs plain (model_config_list) is decided by the source task.
    mediapipe_entries: dict[str, tuple[Path, str]] = {}
    model_entries: dict[str, tuple[Path, str, dict[str, str] | None, dict[str, object] | None]] = {}
    for model_name in active_models:
        entry = ovms.models[model_name]
        model_identity = ovms.repository[entry.source]
        target_dir = model_dir(store, model_identity.hf).resolve()
        if model_identity.task is None:
            model_entries[model_name] = (target_dir, entry.device, entry.plugin_config, entry.plain)
        else:
            mediapipe_entries[model_name] = (target_dir, f"graph.{model_name}.pbtxt")

    try:
        # dry_run renders to build/, live renders to store/.
        if dry_run:
            config_json_path = _BUILD_DIR / "config.json"
        else:
            config_json_path = store / "config.json"
        config_json_path.parent.mkdir(parents=True, exist_ok=True)
        render_config(config_json_path, mediapipe_entries, model_entries)
        logger.info("[apply] config.json rendered with %d model(s)",
                    len(mediapipe_entries) + len(model_entries))
    except (OSError, ValueError) as exc:
        logger.error("[apply] failed to render config.json: %s", exc)
        # Fail-fast: rollback and exit.
        if not dry_run:
            _rollback(config_json_path, config_json_snapshot, existing_graphs)
        return 1

    # Cleanup obsolete sibling-graphs from previous activations (live run only).
    if not dry_run:
        cleaned_up = cleanup.cleanup_obsolete_sibling_graphs(store, active_models, ovms)
        if cleaned_up:
            logger.info("[apply] cleaned up %d obsolete sibling-graph(s)", len(cleaned_up))

    mode = "dry-run -> build/" if dry_run else "live"
    logger.info("[apply] done (%s) with %d model(s)", mode, len(processed_models))
    return 0


def _rollback(config_json_path: Path, config_json_snapshot: str | None, existing_graphs: set[Path]) -> None:
    """Restore config.json from snapshot and delete graph files created in this run.

    Args:
        config_json_path: path to config.json.
        config_json_snapshot: original content of config.json (None if didn't exist).
        existing_graphs: set of graph.<model_name>.pbtxt files that existed before run.
    """
    # Restore config.json.
    if config_json_snapshot is not None:
        try:
            config_json_path.write_text(config_json_snapshot, encoding="utf-8")
            logger.info("[rollback] restored config.json from snapshot")
        except OSError as e:
            logger.error("[rollback] failed to restore config.json: %s", e)
    else:
        # config.json didn't exist before; delete it if it exists now.
        try:
            config_json_path.unlink(missing_ok=True)
            logger.info("[rollback] deleted config.json (didn't exist before run)")
        except OSError as e:
            logger.error("[rollback] failed to delete config.json: %s", e)

    # Delete graph files created in this run (those not in existing_graphs).
    # Walk the store looking for graph.<name>.pbtxt files and remove those
    # that are not in existing_graphs. Paths are resolved for consistent comparison.
    store = config_json_path.parent
    try:
        for graph_file in store.rglob("graph.*.pbtxt"):
            if graph_file.resolve() not in existing_graphs:
                try:
                    graph_file.unlink()
                    logger.debug("[rollback] deleted graph file: %s", graph_file)
                except OSError as e:
                    logger.error("[rollback] failed to delete graph file %s: %s", graph_file, e)
    except OSError as e:
        logger.error("[rollback] error walking store directory: %s", e)
