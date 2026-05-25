"""Tests for ovms command builder.

Unit tests cover:
- command builder (correct argv with/without extras)
- builder does not emit --log_path (probes manage it separately)
- managed flag guards (--log_level, --log_path, --config_path)
- log_level_override parameter
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ovms_rig.config.schema import LocalRuntime, Runtime
from ovms_rig.command import build as build_command


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runtime(rest_port: int = 8000, log_level: str = "INFO") -> Runtime:
    return Runtime(rest_port=rest_port, log_level=log_level)


def _local_runtime(
    ovms_path: Path | None = None,
    cache_dir: Path | None = None,
) -> LocalRuntime:
    return LocalRuntime(ovms_path=ovms_path, cache_dir=cache_dir)


# ---------------------------------------------------------------------------
# Unit: command builder
# ---------------------------------------------------------------------------

def test_build_command_no_extras():
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(binary, config_json, _runtime(), _local_runtime(), [])
    assert cmd[0] == str(binary)
    assert "--config_path" in cmd
    assert cmd[cmd.index("--config_path") + 1] == str(config_json)


def test_build_command_yaml_flags_appear_when_no_extras():
    """YAML rest_port and log_level are injected when extras don't override them."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(
        binary, config_json,
        _runtime(rest_port=8000, log_level="WARNING"),
        _local_runtime(),
        [],
    )
    assert "--rest_port" in cmd
    assert cmd[cmd.index("--rest_port") + 1] == "8000"
    assert "--log_level" in cmd
    assert cmd[cmd.index("--log_level") + 1] == "WARNING"


def test_build_command_extras_rest_port_wins():
    """CLI --rest_port overrides YAML value."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--rest_port", "9000"]
    cmd = build_command(binary, config_json, _runtime(rest_port=8000), _local_runtime(), extras)
    # Only one --rest_port in the command, and it must be 9000.
    indices = [i for i, tok in enumerate(cmd) if tok == "--rest_port"]
    assert len(indices) == 1
    assert cmd[indices[0] + 1] == "9000"


def test_build_command_extras_log_level_rejected():
    """Extras containing --log_level raises ValueError (A3 guard)."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--log_level", "DEBUG"]
    with pytest.raises(ValueError, match="--log-level"):
        build_command(binary, config_json, _runtime(log_level="INFO"), _local_runtime(), extras)


def test_build_command_with_extras():
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--port", "9000"]
    cmd = build_command(binary, config_json, _runtime(), _local_runtime(), extras)
    # Extras appended after YAML flags; --config_path is second token.
    assert cmd[1] == "--config_path"
    assert cmd[2] == str(config_json)
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "9000"


def test_build_command_config_path_value_is_string_not_posix():
    """config_path forwarded as str(path) so ovms sees the native separator."""
    binary = Path("/bin/ovms")
    config_json = Path("/a/b/config.json")
    cmd = build_command(binary, config_json, _runtime(), _local_runtime(), [])
    idx = cmd.index("--config_path")
    assert cmd[idx + 1] == str(config_json)


def test_build_command_cache_dir_from_local_yaml():
    """local.runtime.cache_dir is forwarded to ovms as --cache_dir."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cache = Path("/var/ovms/cache")
    cmd = build_command(
        binary, config_json, _runtime(), _local_runtime(cache_dir=cache), []
    )
    assert "--cache_dir" in cmd
    assert cmd[cmd.index("--cache_dir") + 1] == str(cache)


def test_build_command_extras_cache_dir_wins():
    """CLI --cache_dir overrides the YAML value."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--cache_dir", "/override/cache"]
    cmd = build_command(
        binary, config_json,
        _runtime(),
        _local_runtime(cache_dir=Path("/yaml/cache")),
        extras,
    )
    indices = [i for i, tok in enumerate(cmd) if tok == "--cache_dir"]
    assert len(indices) == 1
    assert cmd[indices[0] + 1] == "/override/cache"


def test_build_command_no_cache_dir_when_unset():
    """Absent cache_dir in YAML and extras -> no --cache_dir in argv."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(binary, config_json, _runtime(), _local_runtime(), [])
    assert "--cache_dir" not in cmd


def test_build_command_no_log_path_emitted():
    """Builder does not emit --log_path by default."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(binary, config_json, _runtime(), _local_runtime(), [])
    assert "--log_path" not in cmd


def test_build_command_rejects_log_level_in_extras():
    """Extras containing --log_level raises ValueError."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--log_level", "DEBUG"]
    with pytest.raises(ValueError, match="--log_level is not allowed in extras"):
        build_command(binary, config_json, _runtime(), _local_runtime(), extras)


def test_build_command_rejects_log_path_in_extras():
    """Extras containing --log_path raises ValueError."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--log_path", "/tmp/custom.log"]
    with pytest.raises(ValueError, match="--log_path is managed by rig"):
        build_command(binary, config_json, _runtime(), _local_runtime(), extras)


def test_build_command_rejects_config_path_in_extras():
    """Extras containing --config_path raises ValueError."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--config_path", "/tmp/alt-config.json"]
    with pytest.raises(ValueError, match="--config_path is managed by rig"):
        build_command(binary, config_json, _runtime(), _local_runtime(), extras)


def test_build_command_log_level_override():
    """log_level_override wins over runtime.log_level."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(
        binary, config_json,
        _runtime(log_level="INFO"),
        _local_runtime(),
        [],
        log_level_override="DEBUG"
    )
    idx = cmd.index("--log_level")
    assert cmd[idx + 1] == "DEBUG"


def test_build_command_log_level_override_none_uses_yaml():
    """log_level_override=None uses runtime.log_level from YAML."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    cmd = build_command(
        binary, config_json,
        _runtime(log_level="WARNING"),
        _local_runtime(),
        [],
        log_level_override=None
    )
    idx = cmd.index("--log_level")
    assert cmd[idx + 1] == "WARNING"
