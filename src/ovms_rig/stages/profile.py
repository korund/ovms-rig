"""Profile activation/deactivation stage.

Updates ovms.yaml active fields and rebuilds config.json + sibling graphs
via apply stage.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_local, load_ovms
from ovms_rig.stages import apply

logger = logging.getLogger(__name__)


def set_active_profile(ctx: dict, target: str | None) -> int:
    """Sets active profile to target (or none if target is None).

    Updates ovms.yaml (active fields), backs it up, then rebuilds
    config.json and sibling graphs via apply stage.

    Args:
        ctx: CLI context dict with config_path, local_path, log_level, ovms_path.
        target: Profile name to activate, or None to deactivate all.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    config_path = Path(ctx["config_path"])
    cli_level: str | None = ctx.get("log_level")

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        ovms = load_ovms(config_path)
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    # Validation: if target specified, check it exists in profiles.
    if target is not None and target not in ovms.profiles:
        available = ", ".join(sorted(ovms.profiles.keys())) if ovms.profiles else "(none)"
        logger.error(
            "profile '%s' not found (available: %s)",
            target, available,
        )
        return 1

    # Read ovms.yaml as raw data so we can preserve structure and rewrite.
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.error("failed to read ovms.yaml: %s", e)
        return 1

    # Ensure profiles section exists for rewriting (in case it was missing).
    if "profiles" not in data:
        data["profiles"] = {}

    # Set active status for each profile.
    for profile_name in data.get("profiles", {}):
        if profile_name == target:
            data["profiles"][profile_name]["active"] = True
        else:
            # Deactivate (set to false or delete the key; we choose false for consistency).
            data["profiles"][profile_name]["active"] = False

    # Backup ovms.yaml before write.
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = config_path.parent / f"{config_path.name}.bak.{timestamp}"
    try:
        with open(config_path, "r", encoding="utf-8") as src:
            backup_path.write_text(src.read(), encoding="utf-8")
        logger.info("[profile] backed up ovms.yaml to %s", backup_path)
    except OSError as e:
        logger.error("failed to backup ovms.yaml: %s", e)
        return 1

    # Serialize new YAML to string.
    try:
        new_yaml_str = yaml.dump(data, sort_keys=False, allow_unicode=False)
    except yaml.YAMLError as e:
        logger.error("failed to serialize ovms.yaml: %s", e)
        return 1

    # Write to temp file, then atomically replace.
    temp_path = config_path.parent / f"{config_path.name}.tmp.{timestamp}"
    try:
        temp_path.write_text(new_yaml_str, encoding="utf-8")
        logger.debug("[profile] wrote temp file: %s", temp_path)
        # Atomic replace: os.replace is atomic on the same filesystem.
        os.replace(temp_path, config_path)
        logger.info("[profile] updated ovms.yaml")
    except (OSError, yaml.YAMLError) as e:
        logger.error("failed to write ovms.yaml: %s", e)
        # Clean up temp file if it exists.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return 1

    # Call apply.run to rebuild config.json and sibling graphs.
    # Pass dry_run=False and extras=[] (no extras needed for apply).
    apply_ctx = dict(ctx)
    apply_ctx["dry_run"] = False
    apply_ctx["extras"] = []

    rc = apply.run(apply_ctx)
    if rc != 0:
        logger.error("[profile] apply failed while rebuilding config (rc=%d)", rc)
        # Rollback: restore ovms.yaml from backup.
        try:
            shutil.copy(backup_path, config_path)
            logger.info("[profile] apply failed (rc=%d), rolled back ovms.yaml from %s", rc, backup_path)
        except OSError as e:
            logger.error("[profile] failed to rollback ovms.yaml: %s", e)
        return rc

    # Log final state.
    if target is not None:
        logger.info("[profile] '%s' is now active", target)
    else:
        logger.info("[profile] no profile is active")

    return 0
