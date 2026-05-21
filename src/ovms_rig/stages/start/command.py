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


def _flags_present(extras: list[str]) -> frozenset[str]:
    """Return the set of '--flag' tokens already present in extras."""
    return frozenset(tok for tok in extras if tok.startswith("--"))


def build(
    binary: Path,
    config_json: Path,
    runtime: "Runtime",
    local_runtime: "LocalRuntime",
    extras: list[str],
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

    Returns a list of strings ready for subprocess.Popen(args=...).
    """
    cmd: list[str] = [str(binary), "--config_path", str(config_json)]

    already_in_extras = _flags_present(extras)

    for field, flag in _RUNTIME_FLAGS.items():
        if flag in already_in_extras:
            # CLI extra takes precedence; skip the YAML value.
            continue
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
