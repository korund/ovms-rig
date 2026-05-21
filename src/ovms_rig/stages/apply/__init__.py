"""Apply stage: patch graph.pbtxt + register config.json from the declaration.

Steps per served entry:
1. Resolve target model directory (store/<hf_org>/<hf_repo>).
2. Warn if pbtxt mtime is newer than last-apply marker.
3. Backup live files (unless dry-run).
4. Patch graph.pbtxt fields from declaration (device, draft_device,
   draft_models_path).
5. Write patched pbtxt to live store or build/ (dry-run).
6. Register endpoint via `ovms --add_to_config` (or emulate for dry-run).
7. Update apply marker (live run only).
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
from ovms_rig.stages.apply import marker as apply_marker
from ovms_rig.stages.apply.paths import model_dir, relative_posix
from ovms_rig.stages.apply.pbtxt import collect_pbtxt_fields, patch
from ovms_rig.stages.apply.registry import add_to_config

logger = logging.getLogger(__name__)

_BUILD_DIR = Path("build")
_BACKUP_ROOT = Path(".backup")


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
    marker = apply_marker.load()
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

        pbtxt_path = target_dir / "graph.pbtxt"
        if not pbtxt_path.exists():
            logger.error(
                "[apply] %s: graph.pbtxt not found at %s",
                entry.name, pbtxt_path,
            )
            failures.append(entry.name)
            continue

        # Step 2: warn if pbtxt was regenerated since last apply.
        apply_marker.warn_if_stale(entry.model, pbtxt_path, marker)

        # Resolve draft path if declared.
        draft_rel: str | None = None
        if entry.graph.draft_model is not None:
            draft_identity = ovms.models[entry.graph.draft_model]
            draft_dir = model_dir(store, draft_identity.hf)
            draft_rel = relative_posix(target_dir, draft_dir)

        fields = collect_pbtxt_fields(
            entry.graph, draft_rel, cache_dir=local.runtime.cache_dir,
        )

        # Compute the destination for this pbtxt (live or build/).
        if dry_run:
            dest_pbtxt = _BUILD_DIR / model_identity.hf / "graph.pbtxt"
        else:
            dest_pbtxt = pbtxt_path

        # Step 3: backup before any write (live run only).
        if not dry_run:
            _backup_file(pbtxt_path, timestamp)

        # Step 4+5: patch and write.
        try:
            new_content = patch(pbtxt_path.read_text(encoding="utf-8"), fields)
        except (ValueError, OSError) as exc:
            logger.error("[apply] %s: pbtxt patch failed: %s", entry.name, exc)
            failures.append(entry.name)
            continue

        dest_pbtxt.parent.mkdir(parents=True, exist_ok=True)
        dest_pbtxt.write_text(new_content, encoding="utf-8")
        logger.info("[apply] %s: graph.pbtxt written to %s", entry.name, dest_pbtxt)

        # Step 6: register in config.json.
        if dry_run:
            config_json_path = _BUILD_DIR / "config.json"
        else:
            config_json_path = store / "config.json"
            _backup_file(config_json_path, timestamp)

        config_json_path.parent.mkdir(parents=True, exist_ok=True)
        # For dry-run, if build/config.json doesn't yet exist, seed it so
        # ovms --add_to_config has a file to update.
        if dry_run and not config_json_path.exists():
            if (store / "config.json").exists():
                shutil.copy2(store / "config.json", config_json_path)

        # model_path for --add_to_config: OVMS expects the absolute path to
        # the model directory (containing graph.pbtxt).
        model_path_for_registry = (
            dest_pbtxt.parent if dry_run else target_dir
        )
        rc = add_to_config(
            binary=binary,
            env=env,
            config_path=config_json_path,
            model_name=entry.name,
            model_path=model_path_for_registry.resolve(),
            extras=extras,
        )
        if rc != 0:
            failures.append(entry.name)
            continue

        # Step 7: update marker (live run only).
        if not dry_run:
            apply_marker.update(entry.model, pbtxt_path, marker)

    # Persist marker after all entries (live run only).
    if not dry_run and not failures:
        apply_marker.save(marker)

    if failures:
        logger.error("apply failed for: %s", ", ".join(failures))
        return 1

    mode = "dry-run -> build/" if dry_run else "live"
    logger.info("[apply] done (%s)", mode)
    return 0


def _backup_file(src: Path, timestamp: str) -> None:
    """Copy src into .backup/<timestamp>/<relative_to_cwd> if src exists."""
    if not src.exists():
        return
    try:
        rel = src.relative_to(Path.cwd())
    except ValueError:
        rel = Path(*src.parts[-2:])  # fallback: last two segments
    dest = _BACKUP_ROOT / timestamp / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    logger.debug("[backup] %s -> %s", src, dest)
