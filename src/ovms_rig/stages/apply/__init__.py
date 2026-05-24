"""Apply stage: copy pristine graph.pbtxt to sibling + patch + register.

Steps per model entry in active profile:
1. Resolve target model directory (store/<hf_org>/<hf_repo>).
2. Read pristine graph.pbtxt (never mutated).
3. Create sibling copy graph.<model_name>.pbtxt.
4. Patch sibling copy (device, draft_device, draft_models_path).
5. Merge generation_config.json overrides if declared on model.
6. Register endpoint via direct config.json JSON write (not ovms CLI).
7. Backup config.json after registration (live run only).
8. Cleanup obsolete sibling-graphs from previous activations.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import (
    ConfigError,
    OvmsConfig,
    load_local,
    load_ovms,
)
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary
from ovms_rig.stages.apply.generation import merge as merge_generation
from ovms_rig.stages.apply.paths import model_dir, relative_posix
from ovms_rig.stages.apply.pbtxt import collect_pbtxt_fields, patch
from ovms_rig.stages.apply.registry import reconcile_mediapipe_entries

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
        ovms = load_ovms(config_path)
        local = load_local(local_path)
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
        logger.info("[apply] no active profile; will register empty config")

    env = build_env(binary.parent)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures: list[str] = []
    processed_models: list[str] = []

    # Process only models in active profile.
    for model_name in active_models:
        entry = ovms.models[model_name]
        model_identity = ovms.repository[entry.source]
        target_dir = model_dir(store, model_identity.hf)

        if not target_dir.is_dir():
            logger.error(
                "[apply] %s: model directory not found at %s; run fetch first",
                model_name, target_dir,
            )
            failures.append(model_name)
            continue

        pristine_pbtxt = target_dir / "graph.pbtxt"
        if not pristine_pbtxt.exists():
            logger.error(
                "[apply] %s: graph.pbtxt not found at %s",
                model_name, pristine_pbtxt,
            )
            failures.append(model_name)
            continue

        # Resolve draft path if declared.
        draft_rel: str | None = None
        if entry.graph.draft_model is not None:
            draft_identity = ovms.repository[entry.graph.draft_model]
            draft_dir = model_dir(store, draft_identity.hf)
            draft_rel = relative_posix(target_dir, draft_dir)

        fields = collect_pbtxt_fields(
            entry.graph, draft_rel, cache_dir=local.runtime.cache_dir,
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
            failures.append(model_name)
            continue

        # Patch the pristine content (goes to sibling, not to pristine itself).
        try:
            patched_content = patch(pristine_content, fields)
        except (ValueError, OSError) as exc:
            logger.error("[apply] %s: pbtxt patch failed: %s", model_name, exc)
            failures.append(model_name)
            continue

        sibling_pbtxt.parent.mkdir(parents=True, exist_ok=True)
        sibling_pbtxt.write_text(patched_content, encoding="utf-8")
        logger.info("[apply] %s: graph.%s.pbtxt written to %s", model_name, model_name, sibling_pbtxt)

        # Handle generation_config.json if overrides are declared on the entry.
        overrides = entry.generation
        if overrides:
            genconfig_path = target_dir / "generation_config.json"

            # For dry-run, log intention without requiring file to exist.
            if dry_run:
                dest_genconfig = _BUILD_DIR / model_identity.hf / "generation_config.json"
                logger.info(
                    "[apply] %s: would read generation overrides from %s",
                    model_name, genconfig_path,
                )
                dest_genconfig.parent.mkdir(parents=True, exist_ok=True)
                # Write proposed result to build/ (assumes pristine exists in dry-run context).
                if genconfig_path.exists():
                    try:
                        existing_text = genconfig_path.read_text(encoding="utf-8")
                        new_genconfig = merge_generation(existing_text, overrides)
                        dest_genconfig.write_text(new_genconfig, encoding="utf-8")
                    except (ValueError, OSError) as exc:
                        logger.warning(
                            "[apply] %s: could not preview generation_config merge: %s",
                            model_name, exc,
                        )
                continue

            # Live run: file must exist.
            if not genconfig_path.exists():
                logger.error(
                    "[apply] %s: generation_config.json not found at %s",
                    model_name, genconfig_path,
                )
                failures.append(model_name)
                continue

            dest_genconfig = genconfig_path

            # Backup before write (live run only).
            _backup_file(genconfig_path, timestamp)

            # Merge and write.
            try:
                existing_text = genconfig_path.read_text(encoding="utf-8")
                new_genconfig = merge_generation(existing_text, overrides)
            except (ValueError, OSError) as exc:
                logger.error(
                    "[apply] %s: generation_config merge failed: %s",
                    model_name, exc,
                )
                failures.append(model_name)
                continue

            dest_genconfig.write_text(new_genconfig, encoding="utf-8")
            logger.info(
                "[apply] %s: generation_config.json written to %s",
                model_name, dest_genconfig,
            )

        processed_models.append(model_name)

    # Reconcile config.json with only active models.
    if dry_run:
        config_json_path = _BUILD_DIR / "config.json"
    else:
        config_json_path = store / "config.json"

    config_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Backup config.json before mutation (live run only).
    if not dry_run and config_json_path.exists():
        _backup_file(config_json_path, timestamp)

    # Build desired entries dict from processed models.
    desired_entries: dict[str, tuple[Path, str]] = {}
    for model_name in processed_models:
        entry = ovms.models[model_name]
        model_identity = ovms.repository[entry.source]
        target_dir = model_dir(store, model_identity.hf)
        desired_entries[model_name] = (target_dir.resolve(), f"graph.{model_name}.pbtxt")

    try:
        reconcile_mediapipe_entries(config_json_path, desired_entries)
        logger.info("[apply] config.json reconciled with %d model(s)", len(desired_entries))
    except (OSError, ValueError) as exc:
        logger.error("[apply] failed to reconcile config.json: %s", exc)
        failures.append("config.json")

    # Cleanup obsolete sibling-graphs from previous activations (live run only).
    if not dry_run:
        cleaned_up = _cleanup_obsolete_sibling_graphs(store, active_models, ovms)
        if cleaned_up:
            logger.info("[apply] cleaned up %d obsolete sibling-graph(s)", len(cleaned_up))

    if failures:
        logger.error("apply failed for: %s", ", ".join(failures))
        return 1

    mode = "dry-run -> build/" if dry_run else "live"
    logger.info("[apply] done (%s) with %d model(s)", mode, len(processed_models))
    return 0


def _cleanup_obsolete_sibling_graphs(store: Path, active_models: set[str], ovms: OvmsConfig) -> list[str]:
    """Remove sibling-graphs for models not in active profile.

    Scans model directories from ovms.repository and removes graph.<name>.pbtxt
    files whose <name> is not in active_models.

    Returns list of cleaned-up paths.
    """
    cleaned_up: list[str] = []
    seen_dirs: set[Path] = set()

    for repo_name, identity in ovms.repository.items():
        model_dir = store / identity.hf
        if model_dir in seen_dirs or not model_dir.is_dir():
            continue
        seen_dirs.add(model_dir)

        for sibling_graph in model_dir.glob("graph.*.pbtxt"):
            # Extract model name from filename: graph.<name>.pbtxt
            # Use Path.stem to remove .pbtxt, then removeprefix to get name.
            stem = sibling_graph.stem  # removes .pbtxt
            name = stem.removeprefix("graph.")

            # Skip if name is empty (malformed filename).
            if not name:
                logger.debug("[cleanup] skipping malformed sibling-graph: %s", sibling_graph)
                continue

            # If model not in active_models, remove the sibling-graph.
            if name not in active_models:
                try:
                    sibling_graph.unlink()
                    cleaned_up.append(str(sibling_graph))
                    logger.debug("[cleanup] removed obsolete sibling-graph: %s", sibling_graph)
                except OSError as exc:
                    logger.warning("[cleanup] failed to remove %s: %s", sibling_graph, exc)

    return cleaned_up


def _backup_file(src: Path, timestamp: str) -> None:
    """Create a backup of src file with suffix: src.bak.<timestamp>.

    Backup stored next to original (same directory).
    timestamp format: YYYYMMDDTHHMMSS (UTC, no colons).
    """
    if not src.exists():
        return
    dest = src.parent / f"{src.name}.bak.{timestamp}"
    shutil.copy2(src, dest)
    logger.debug("[backup] %s -> %s", src, dest)
