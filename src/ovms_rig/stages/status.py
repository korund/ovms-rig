"""State check: declaration vs world. Read-only, no side effects."""

from __future__ import annotations

import logging
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.probes import live_config, model_store, ovms_binary, port
from ovms_rig.config import ConfigError, LocalConfig, load_local, load_ovms
from ovms_rig.report import CheckResult

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")

    # Logging is bootstrapped at default level so config-load failures still
    # surface; once the declaration is loaded we re-apply the declared level.
    logging_setup.configure((cli_level or "INFO").upper())

    try:
        ovms = load_ovms(config_path)
        local = load_local(local_path) if local_path.exists() else LocalConfig()
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)
    logger.debug("log level: %s (source: %s)",
                 level, "cli" if cli_level else "ovms.yaml")
    logger.debug("config: %s", config_path)
    logger.debug("local:  %s", local_path)

    results: list[CheckResult] = [
        _declaration_ok(config_path, local_path),
        ovms_binary.check(ovms_override, local),
        model_store.check_destination(local),
        model_store.check_inventory(ovms, local),
        port.check(ovms),
        live_config.check(ovms, local),
    ]

    for r in results:
        _emit(r)

    return 1 if any(r.status == "error" for r in results) else 0


def _declaration_ok(config_path: Path, local_path: Path) -> CheckResult:
    return CheckResult(
        name="declaration",
        status="ok",
        summary="ovms.yaml + local.yaml parsed and references resolved",
        details={
            "config": str(config_path),
            "local": str(local_path) if local_path.exists() else "absent (defaults)",
        },
    )


def _emit(r: CheckResult) -> None:
    level = logging.ERROR if r.status == "error" else logging.INFO
    logger.log(level, "[%s] %s -- %s", r.status.upper(), r.name, r.summary)
    if r.hint:
        logger.info("  hint: %s", r.hint)
    if r.details:
        logger.debug("  details: %s", r.details)
