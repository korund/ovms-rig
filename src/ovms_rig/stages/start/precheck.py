"""Precheck: run status internally and split results into hard/soft.

Hard errors -> return nonzero immediately.
Soft warnings -> log and continue.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_declaration
from ovms_rig.probes import live_config, repository, ovms_binary, port
from ovms_rig.report import CheckResult

logger = logging.getLogger(__name__)

# Names of checks whose failure is considered a hard (blocking) error.
_HARD_ERROR_NAMES = frozenset(
    {
        "declaration",
        "ovms binary",
        "rest port",
    }
)


def run(ctx: dict) -> int:
    """Run all status checks and print results.

    Returns 0 if no hard errors, 1 if any hard error was found.
    Soft errors (warn status) are emitted as warnings.
    """
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        ovms, local = load_declaration(config_path, local_path)
    except ConfigError as e:
        logger.error("precheck: config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)

    results: list[CheckResult] = [
        _declaration_ok(config_path, local_path),
        ovms_binary.check(ovms_override, local),
        repository.check_destination(local),
        repository.check_inventory(ovms, local),
        port.check(ovms),
        live_config.check(ovms, local),
    ]

    hard_failure = False
    for r in results:
        _emit(r)
        if r.status == "error" and r.name in _HARD_ERROR_NAMES:
            hard_failure = True

    if hard_failure:
        return 1
    return 0


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
    if r.status == "error":
        level = logging.ERROR
    elif r.status == "warn":
        level = logging.WARNING
    else:
        level = logging.INFO
    logger.log(level, "[%s] %s -- %s", r.status.upper(), r.name, r.summary)
    if r.hint:
        logger.info("  hint: %s", r.hint)
    if r.details:
        logger.debug("  details: %s", r.details)
