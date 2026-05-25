"""Build the ovms command-line for a foreground start."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ovms_rig.config.schema import LocalRuntime, Runtime


# Mapping from Runtime field names to ovms CLI flag names.
_RUNTIME_FLAGS: dict[str, str] = {
    "rest_port": "--rest_port",
    "log_level": "--log_level",
}

# Flags that must not appear in extras (managed by rig, not user).
_MANAGED_FLAGS: frozenset[str] = frozenset({
    "--log_level",
    "--log_path",
    "--config_path",
})


def _flags_present(extras: list[str]) -> frozenset[str]:
    """Return the set of '--flag' tokens already present in extras."""
    return frozenset(tok for tok in extras if tok.startswith("--"))


def build(
    binary: Path,
    config_json: Path,
    runtime: "Runtime",
    local_runtime: "LocalRuntime",
    extras: list[str],
    log_level_override: str | None = None,
) -> list[str]:
    """Return the argv list to pass to Popen.

    Precedence: CLI extras > YAML > ovms defaults.
    Implemented via explicit guard: if a flag already appears in extras it is
    not emitted from YAML, so the duplicate is never passed to ovms.  This
    avoids any reliance on ovms flag-precedence behaviour.

    Args:
        binary: Absolute path to the ovms executable.
        config_json: Path to the config.json produced by `apply`.
        runtime: Parsed Runtime section of ovms.yaml (rest_port, log_level).
        local_runtime: Parsed runtime section of local.yaml; supplies
                       per-machine flags such as cache_dir.
        extras: Extra CLI tokens forwarded verbatim from the rig start
                command (e.g. ['--port', '9000', '--log_level', 'DEBUG']).
        log_level_override: If provided, overrides runtime.log_level. Used by
                           probes to force DEBUG regardless of YAML config.

    Returns a list of strings ready for subprocess.Popen(args=...).

    Raises:
        ValueError: If extras contains managed flags (--log_level, --log_path,
                   --config_path).
    """
    already_in_extras = _flags_present(extras)

    # Guard against managed flags in extras.
    for managed_flag in _MANAGED_FLAGS:
        if managed_flag in already_in_extras:
            if managed_flag == "--log_level":
                raise ValueError(
                    "--log_level is not allowed in extras; "
                    "use the global --log-level flag on rig"
                )
            elif managed_flag == "--log_path":
                raise ValueError(
                    "--log_path is managed by rig; ovms logs flow through rig's stdio"
                )
            elif managed_flag == "--config_path":
                raise ValueError(
                    "--config_path is managed by rig; "
                    "the config is rendered from the declaration"
                )

    cmd: list[str] = [str(binary), "--config_path", str(config_json)]

    for field, flag in _RUNTIME_FLAGS.items():
        if flag in already_in_extras:
            # CLI extra takes precedence; skip the YAML value.
            continue
        if flag == "--log_level":
            # Use override if provided; otherwise fall back to YAML.
            value = log_level_override if log_level_override is not None else getattr(runtime, field, None)
        else:
            value = getattr(runtime, field, None)
        if value is not None:
            cmd.extend([flag, str(value)])

    # cache_dir lives on the per-machine local.yaml, not ovms.yaml: where the
    # OpenVINO compile cache lands is a machine concern (disk, NVMe vs HDD),
    # not part of the declared deployment.
    if "--cache_dir" not in already_in_extras:
        cache_dir = local_runtime.cache_dir
        if cache_dir is not None:
            cmd.extend(["--cache_dir", str(cache_dir)])

    cmd.extend(extras)
    return cmd
