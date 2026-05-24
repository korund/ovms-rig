"""State check: declaration vs world. Read-only, no side effects."""

from __future__ import annotations

import logging
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_declaration
from ovms_rig.probes import registry

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    cli_level: str | None = ctx.get("log_level")

    logging_setup.configure((cli_level or "INFO").upper())

    try:
        ovms, _ = load_declaration(config_path, local_path)
        level = (cli_level or ovms.runtime.log_level).upper()
        logging_setup.configure(level)
        logger.debug("log level: %s (source: %s)",
                     level, "cli" if cli_level else "ovms.yaml")
    except ConfigError:
        pass

    logger.debug("config: %s", config_path)
    logger.debug("local:  %s", local_path)

    report = registry.run(ctx, registry.Preset.DIAGNOSTIC)
    report.print()

    return 1 if report.has_errors() else 0
