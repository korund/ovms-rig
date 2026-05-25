"""Profile activation/deactivation stage.

Updates ovms.yaml active fields and rebuilds config.json + sibling graphs
via apply submodule.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_declaration
from ovms_rig.stages.activation import apply
from ovms_rig.probes import smoke_load

logger = logging.getLogger(__name__)


def set_active_profile(ctx: dict, target: str | None, *, backup: bool = False) -> int:
    """Sets active profile to target (or none if target is None).

    Updates ovms.yaml (active fields), then rebuilds config.json and
    sibling graphs via apply stage. On apply failure (mid-write crash),
    ovms.yaml is rolled back from an in-memory snapshot and apply restores
    derived files from its own snapshot. On smoke-load failure, files are
    left as written: the rendered config is what OVMS rejected, fixing it
    requires editing ovms.yaml and re-running activate. The on-disk
    ovms.yaml.bak is opt-in via `backup` (overwrites any previous .bak).

    Args:
        ctx: CLI context dict with config_path, local_path, log_level, ovms_path.
        target: Profile name to activate, or None to deactivate all.
        backup: If True, write ovms.yaml.bak next to ovms.yaml before overwrite.

    Returns:
        Exit code (0 on success, 1 on error).
    """
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    cli_level: str | None = ctx.get("log_level")

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        decl = load_declaration(config_path, local_path)
        ovms = decl.ovms
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

    # Read ovms.yaml as raw text + parsed data. The text snapshot lets us
    # roll back in-memory if apply fails; the parsed data is what we mutate.
    try:
        original_text = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(original_text)
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

    # Opt-in disk backup. Fixed name, overwrites any previous .bak.
    if backup:
        backup_path = config_path.parent / f"{config_path.name}.bak"
        try:
            backup_path.write_text(original_text, encoding="utf-8")
            logger.info("[activation] backed up ovms.yaml to %s", backup_path)
        except OSError as e:
            logger.error("failed to backup ovms.yaml: %s", e)
            return 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

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
        logger.debug("[activation] wrote temp file: %s", temp_path)
        # Atomic replace: os.replace is atomic on the same filesystem.
        os.replace(temp_path, config_path)
        logger.info("[activation] updated ovms.yaml")
    except (OSError, yaml.YAMLError) as e:
        logger.error("failed to write ovms.yaml: %s", e)
        # Clean up temp file if it exists.
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return 1

    # Call apply.run to rebuild config.json and sibling graphs.
    # Pass dry_run=False and extras=[] (no extras needed).
    apply_ctx = dict(ctx)
    apply_ctx["dry_run"] = False
    apply_ctx["extras"] = []

    rc = apply.run(apply_ctx)
    if rc != 0:
        logger.error("[activation] apply failed while rebuilding config (rc=%d)", rc)
        # Rollback ovms.yaml from in-memory snapshot.
        try:
            config_path.write_text(original_text, encoding="utf-8")
            logger.info("[activation] rolled back ovms.yaml from snapshot")
        except OSError as e:
            logger.error("[activation] failed to rollback ovms.yaml: %s", e)
        return rc

    # Smoke-load validation: probe OVMS with generated config.
    try:
        decl = load_declaration(config_path, local_path, cli_override=Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None)
    except ConfigError as e:
        logger.error("[activation] config reload for smoke-load failed: %s", e)
        return 1

    result = smoke_load.check(decl)

    if result.status == "error":
        logger.error("[activation] smoke-load validation failed: %s", result.summary)
        if result.details.get("fail_markers"):
            for marker in result.details["fail_markers"]:
                logger.error("  %s", marker)
        if result.details.get("log_tail"):
            logger.error("[activation] last log lines:")
            for line in result.details["log_tail"]:
                logger.error("  %s", line)
        return 1

    if result.status == "warn":
        logger.warning("[activation] smoke-load warning: %s", result.summary)
        if result.hint:
            logger.info("  hint: %s", result.hint)

    # Log final state.
    if target is not None:
        logger.info("[activation] '%s' is now active", target)
    else:
        logger.info("[activation] no profile is active")

    return 0
