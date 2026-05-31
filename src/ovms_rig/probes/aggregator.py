from __future__ import annotations

import logging
from dataclasses import dataclass

from ovms_rig.probes.registry import Probe
from ovms_rig.report import CheckResult

logger = logging.getLogger(__name__)


@dataclass
class Report:
    entries: list[tuple[Probe, CheckResult]]

    def has_errors(self) -> bool:
        return any(result.status == "error" for _, result in self.entries)

    def print(self) -> None:
        for _, result in self.entries:
            level = logging.ERROR if result.status == "error" else logging.INFO
            logger.log(level, "[%s] %s -- %s", result.status.upper(), result.name, result.summary)
            if result.hint:
                logger.info("  hint: %s", result.hint)
            if result.details:
                logger.debug("  details: %s", result.details)
