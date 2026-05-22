"""Apply stage: copy pristine graph.pbtxt to sibling + patch + register.

Steps per served entry:
1. Resolve target model directory (store/<hf_org>/<hf_repo>).
2. Read pristine graph.pbtxt (never mutated).
3. Create sibling copy graph.<served_name>.pbtxt.
4. Patch sibling copy (device, draft_device, draft_models_path).
5. Merge generation_config.json overrides if declared on model.
6. Register endpoint via direct config.json JSON write (not ovms CLI).
7. Backup config.json after registration (live run only).
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import (
    ConfigError,
    load_local,
    load_ovms,
)
from ovms_rig.env import build_env
from ovms_rig.probes import ovms_binary
from ovms_rig.stages.apply.generation import merge as merge_generation
from ovms_rig.stages.apply.paths import model_dir, relative_posix
from ovms_rig.stages.apply.pbtxt import collect_pbtxt_fields, patch
from ovms_rig.stages.apply.registry import register_mediapipe_entry

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

    env = build_env(binary.parent)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    failures: list[str] = []

    for entry in ovms.served:
        model_identity = ovms.models[entry.model]
        target_dir = model_dir(store, model_identity.hf)

        if not target_dir.is_dir():
            logger.error(
                "[apply] %s: model directory not found at %s; run fetch first",
                entry.name, target_dir,
            )
            failures.append(entry.name)
            continue

        pristine_pbtxt = target_dir / "graph.pbtxt"
        if not pristine_pbtxt.exists():
            logger.error(
                "[apply] %s: graph.pbtxt not found at %s",
                entry.name, pristine_pbtxt,
            )
            failures.append(entry.name)
            continue

        # Resolve draft path if declared.
        draft_rel: str | None = None
        if entry.graph.draft_model is not None:
            draft_identity = ovms.models[entry.graph.draft_model]
            draft_dir = model_dir(store, draft_identity.hf)
            draft_rel = relative_posix(target_dir, draft_dir)

        fields = collect_pbtxt_fields(
            entry.graph, draft_rel, cache_dir=local.runtime.cache_dir,
        )

        # Compute destination path for sibling-copy (live or build/).
        # Sibling naming: graph.<served_name>.pbtxt in the same directory as pristine.
        if dry_run:
            sibling_pbtxt = _BUILD_DIR / model_identity.hf / f"graph.{entry.name}.pbtxt"
        else:
            sibling_pbtxt = target_dir / f"graph.{entry.name}.pbtxt"

        # Read pristine (never mutate).
        try:
            pristine_content = pristine_pbtxt.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("[apply] %s: failed to read pristine graph.pbtxt: %s", entry.name, exc)
            failures.append(entry.name)
            continue

        # Patch the pristine content (goes to sibling, not to pristine itself).
        try:
            patched_content = patch(pristine_content, fields)
        except (ValueError, OSError) as exc:
            logger.error("[apply] %s: pbtxt patch failed: %s", entry.name, exc)
            failures.append(entry.name)
            continue

        sibling_pbtxt.parent.mkdir(parents=True, exist_ok=True)
        sibling_pbtxt.write_text(patched_content, encoding="utf-8")
        logger.info("[apply] %s: graph.%s.pbtxt written to %s", entry.name, entry.name, sibling_pbtxt)

        # Handle generation_config.json if overrides are declared on the entry.
        overrides = entry.generation
        if overrides:
            genconfig_path = target_dir / "generation_config.json"
            if not genconfig_path.exists():
                logger.error(
                    "[apply] %s: generation_config.json not found at %s",
                    entry.name, genconfig_path,
                )
                failures.append(entry.name)
                continue

            # Compute destination (live or build/).
            if dry_run:
                dest_genconfig = _BUILD_DIR / model_identity.hf / "generation_config.json"
            else:
                dest_genconfig = genconfig_path

            # Backup before write (live run only).
            if not dry_run:
                _backup_file(genconfig_path, timestamp)

            # Merge and write.
            try:
                existing_text = genconfig_path.read_text(encoding="utf-8")
                new_genconfig = merge_generation(existing_text, overrides)
            except (ValueError, OSError) as exc:
                logger.error(
                    "[apply] %s: generation_config merge failed: %s",
                    entry.name, exc,
                )
                failures.append(entry.name)
                continue

            dest_genconfig.parent.mkdir(parents=True, exist_ok=True)
            dest_genconfig.write_text(new_genconfig, encoding="utf-8")
            logger.info(
                "[apply] %s: generation_config.json written to %s",
                entry.name, dest_genconfig,
            )

        # Step 6: register in config.json via direct JSON write.
        if dry_run:
            config_json_path = _BUILD_DIR / "config.json"
        else:
            config_json_path = store / "config.json"

        config_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Backup config.json before mutation (live run only).
        if not dry_run and config_json_path.exists():
            _backup_file(config_json_path, timestamp)

        # Register mediapipe entry with sibling-copy graph path.
        # graph_path is relative from base_path (model directory).
        try:
            register_mediapipe_entry(
                config_path=config_json_path,
                entry_name=entry.name,
                base_path=target_dir.resolve(),
                graph_path=f"graph.{entry.name}.pbtxt",
            )
        except (OSError, ValueError) as exc:
            logger.error("[apply] %s: failed to register in config.json: %s", entry.name, exc)
            failures.append(entry.name)
            continue

    if failures:
        logger.error("apply failed for: %s", ", ".join(failures))
        return 1

    mode = "dry-run -> build/" if dry_run else "live"
    logger.info("[apply] done (%s)", mode)
    return 0


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
