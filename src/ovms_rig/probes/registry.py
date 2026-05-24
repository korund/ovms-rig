from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ovms_rig.config import ConfigError, OvmsConfig, LocalConfig, load_declaration
from ovms_rig.probes import live_config, models, ovms_binary, port, profiles, repository
from ovms_rig.report import CheckResult

ProbeFn = Callable[[dict], CheckResult]


@dataclass(frozen=True)
class Probe:
    name: str
    fn: ProbeFn


def _get_declaration(ctx: dict) -> tuple[OvmsConfig, LocalConfig] | None:
    if "_probe_declaration_cache" not in ctx:
        config_path = Path(ctx["config_path"])
        local_path = Path(ctx["local_path"])
        try:
            decl = load_declaration(config_path, local_path)
            ctx["_probe_declaration_cache"] = (decl, None)
        except ConfigError as e:
            ctx["_probe_declaration_cache"] = (None, str(e))
    decl, error = ctx["_probe_declaration_cache"]
    return decl if error is None else None


def _declaration_error(ctx: dict) -> str | None:
    if "_probe_declaration_cache" not in ctx:
        _get_declaration(ctx)
    _, error = ctx["_probe_declaration_cache"]
    return error


def _declaration(ctx: dict) -> CheckResult:
    error_msg = _declaration_error(ctx)
    if error_msg is not None:
        return CheckResult(
            name="declaration",
            status="error",
            summary=f"config load failed: {error_msg}",
        )
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    return CheckResult(
        name="declaration",
        status="ok",
        summary="ovms.yaml + local.yaml parsed and references resolved",
        details={
            "config": str(config_path),
            "local": str(local_path) if local_path.exists() else "absent (defaults)",
        },
    )


def _ovms_binary(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="ovms binary",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    return ovms_binary.check(ovms_override, local)


def _repo_destination(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="repository destination",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return repository.check_destination(local)


def _repo_inventory(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="repository inventory",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return repository.check_inventory(ovms, local)


def _models(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="models",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return models.check(ovms)


def _profiles(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="profiles",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return profiles.check(ovms)


def _port(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="rest port",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return port.check(ovms)


def _live_config(ctx: dict) -> CheckResult:
    decl = _get_declaration(ctx)
    if decl is None:
        return CheckResult(
            name="live config",
            status="error",
            summary="skipped (config load failed)",
        )
    ovms, local = decl
    return live_config.check(ovms, local)


PROBES: dict[str, Probe] = {
    "declaration": Probe("declaration", _declaration),
    "ovms_binary": Probe("ovms binary", _ovms_binary),
    "repository.destination": Probe("repository destination", _repo_destination),
    "repository.inventory": Probe("repository inventory", _repo_inventory),
    "models": Probe("models", _models),
    "profiles": Probe("profiles", _profiles),
    "port": Probe("rest port", _port),
    "live_config": Probe("live config", _live_config),
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

    entries = []
    for key in PRESETS[preset]:
        probe = PROBES[key]
        entries.append((probe, probe.fn(ctx)))
    return Report(entries)
