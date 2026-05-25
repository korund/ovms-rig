from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ovms_rig.config import ConfigError, Declaration, load_declaration
from ovms_rig.probes import live_config, models, ovms_binary, port, profiles, repository
from ovms_rig.report import CheckResult

ProbeFn = Callable[[Declaration], CheckResult]


@dataclass(frozen=True)
class Probe:
    name: str
    fn: ProbeFn


# Direct probe functions that all accept Declaration
def _declaration(decl: Declaration) -> CheckResult:
    """Check that declaration files loaded and references resolved."""
    return CheckResult(
        name="declaration",
        status="ok",
        summary="ovms.yaml + local.yaml parsed and references resolved",
        details={
            "config": str(decl.ovms),
            "local": "configured" if decl.local else "defaults",
        },
    )


PROBES: dict[str, Probe] = {
    "declaration": Probe("declaration", _declaration),
    "ovms_binary": Probe("ovms binary", ovms_binary.check),
    "repository.destination": Probe("repository destination", repository.check_destination),
    "repository.inventory": Probe("repository inventory", repository.check_inventory),
    "models": Probe("models", models.check),
    "profiles": Probe("profiles", profiles.check),
    "port": Probe("rest port", port.check),
    "live_config": Probe("live config", live_config.check),
}


class Preset(str, Enum):
    DIAGNOSTIC = "diagnostic"
    BLOCKING = "blocking"


PRESETS: dict[Preset, tuple[str, ...]] = {
    Preset.DIAGNOSTIC: tuple(PROBES.keys()),
    Preset.BLOCKING: ("declaration", "ovms_binary", "models", "port"),
}


def run(ctx: dict, preset: Preset) -> "Report":
    from ovms_rig.probes.report import Report

    # Load declaration once per run
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    cli_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None

    try:
        decl = load_declaration(config_path, local_path, cli_override=cli_override)
    except ConfigError as e:
        # If declaration load fails, return a single error entry
        return Report(
            [
                (
                    PROBES["declaration"],
                    CheckResult(
                        name="declaration",
                        status="error",
                        summary=f"config load failed: {e}",
                    ),
                )
            ]
        )

    entries = []
    for key in PRESETS[preset]:
        probe = PROBES[key]
        entries.append((probe, probe.fn(decl)))
    return Report(entries)
