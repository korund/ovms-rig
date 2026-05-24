"""Tests for the start stage.

Unit tests cover:
- command builder (correct argv with/without extras)
- signal forwarder (mock subprocess, verify terminate() is called on SIGTERM)

Integration tests cover:
- blocking probe hard error -> nonzero exit, no Popen called
- diagnostic probes not checked in start (soft warns do not block launch)
"""

from __future__ import annotations

import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ovms_rig.config.schema import LocalRuntime, Runtime
from ovms_rig.stages.start.command import build as build_command
from ovms_rig.stages.start.signals import GRACEFUL_TIMEOUT_S, _wait_or_kill


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


def test_build_command_extras_log_level_wins():
    """CLI --log_level overrides YAML value."""
    binary = Path("/opt/ovms/bin/ovms")
    config_json = Path("/store/config.json")
    extras = ["--log_level", "DEBUG"]
    cmd = build_command(binary, config_json, _runtime(log_level="INFO"), _local_runtime(), extras)
    indices = [i for i, tok in enumerate(cmd) if tok == "--log_level"]
    assert len(indices) == 1
    assert cmd[indices[0] + 1] == "DEBUG"


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


# ---------------------------------------------------------------------------
# Unit: signal forwarder / _wait_or_kill
# ---------------------------------------------------------------------------

def _make_proc(returncode=0):
    # Plain MagicMock, no spec=subprocess.Popen.  When tests patch
    # `ovms_rig.stages.start.launch.subprocess.Popen` they reach into the
    # real `subprocess` module (launch does `import subprocess`), so
    # `subprocess.Popen` itself becomes a Mock and cannot be used as a spec.
    proc = MagicMock()
    proc.pid = 12345
    proc.returncode = returncode
    proc.wait.return_value = returncode
    return proc


def test_wait_or_kill_exits_within_timeout():
    proc = _make_proc()
    _wait_or_kill(proc)
    proc.wait.assert_called_once_with(timeout=GRACEFUL_TIMEOUT_S)
    proc.kill.assert_not_called()


def test_wait_or_kill_kills_after_timeout():
    proc = _make_proc()
    # First wait raises TimeoutExpired, second (after kill) returns normally.
    proc.wait.side_effect = [subprocess.TimeoutExpired(cmd="ovms", timeout=GRACEFUL_TIMEOUT_S), None]
    _wait_or_kill(proc)
    proc.kill.assert_called_once()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_posix_sigterm_calls_send_signal():
    from ovms_rig.stages.start.signals import _install_posix

    proc = _make_proc()
    _install_posix(proc)

    # Simulate SIGTERM delivery.
    handler = signal.getsignal(signal.SIGTERM)
    handler(signal.SIGTERM, None)

    proc.send_signal.assert_called_once_with(signal.SIGTERM)
    proc.wait.assert_called()

    # Restore default handler.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals only")
def test_posix_sigint_calls_send_signal():
    from ovms_rig.stages.start.signals import _install_posix

    proc = _make_proc()
    _install_posix(proc)

    handler = signal.getsignal(signal.SIGINT)
    handler(signal.SIGINT, None)

    proc.send_signal.assert_called_once_with(signal.SIGINT)

    signal.signal(signal.SIGINT, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# Integration: blocking probe error -> Popen not called
# ---------------------------------------------------------------------------

OVMS_YAML = """
runtime:
  rest_port: 8765
  log_level: DEBUG

repository:
  main:
    hf: org/main-int8-ov
    task: text_generation
  draft:
    hf: org/draft-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
      draft_model: draft
      draft_device: CPU
"""

LOCAL_YAML = """
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""


@pytest.fixture
def rig(tmp_path: Path) -> dict:
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()
    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")
    return {
        "config_path": str(cfg),
        "local_path": str(loc),
        "ovms_path": None,
        "log_level": None,
        "extras": [],
    }


def test_blocking_probe_error_prevents_launch(rig: dict, tmp_path: Path) -> None:
    """Binary not found (blocking probe error) -> launch.run returns 1, Popen not called."""
    # Force a hard error: point --ovms-path at a file that does not exist.
    rig["ovms_path"] = str(tmp_path / "does-not-exist-ovms")
    with patch("ovms_rig.stages.start.launch.subprocess.Popen") as mock_popen:
        from ovms_rig.stages.start.launch import run
        rc = run(rig)
    assert rc != 0
    mock_popen.assert_not_called()


def test_diagnostic_probes_not_checked_in_start(rig: dict, tmp_path: Path) -> None:
    """Diagnostic probes (live config, models inventory) are not checked before start."""
    # Provide a real binary path (sys.executable) so blocking checks pass.
    rig["ovms_path"] = sys.executable

    # Seed a config.json so the store path is found (apply would create it).
    store = Path(rig["local_path"]).parent / "store"
    config_json = store / "config.json"
    config_json.write_text('{"model_config_list": []}', encoding="utf-8")

    fake_proc = _make_proc(returncode=0)

    with patch("ovms_rig.stages.start.launch.subprocess.Popen", return_value=fake_proc) as mock_popen:
        from ovms_rig.stages.start.launch import run
        rc = run(rig)

    # Popen was called (launch proceeded even though diagnostic probe would fail).
    mock_popen.assert_called_once()
    # Exit code mirrors the fake process.
    assert rc == 0


def test_start_cli_passes_extras_to_command(rig: dict, tmp_path: Path) -> None:
    """Extra args from CLI end up in the Popen argv after --config_path."""
    rig["ovms_path"] = sys.executable
    rig["extras"] = ["--port", "9999"]

    store = Path(rig["local_path"]).parent / "store"
    config_json = store / "config.json"
    config_json.write_text('{"model_config_list": []}', encoding="utf-8")

    captured: list[list[str]] = []

    def fake_popen(args, **kwargs):
        captured.append(list(args))
        return _make_proc(returncode=0)

    with patch("ovms_rig.stages.start.launch.subprocess.Popen", side_effect=fake_popen):
        from ovms_rig.stages.start.launch import run
        run(rig)

    assert captured, "Popen was never called"
    cmd = captured[0]
    assert "--port" in cmd
    assert cmd[cmd.index("--port") + 1] == "9999"
    assert "--config_path" in cmd


def test_launch_creates_cache_dir_when_missing(rig: dict, tmp_path: Path) -> None:
    """If local.runtime.cache_dir points at a nonexistent path, launch mkdirs it."""
    rig["ovms_path"] = sys.executable

    store = Path(rig["local_path"]).parent / "store"
    (store / "config.json").write_text('{"model_config_list": []}', encoding="utf-8")

    cache = tmp_path / "ov_cache" / "nested"  # parents missing on purpose
    assert not cache.exists()

    # Rewrite local.yaml to include cache_dir.
    Path(rig["local_path"]).write_text(
        "runtime:\n"
        "  ovms_path: null\n"
        f"  cache_dir: {cache.as_posix()}\n"
        "models:\n"
        f"  repository_path: {store.as_posix()}\n",
        encoding="utf-8",
    )

    captured: list[list[str]] = []

    def fake_popen(args, **kwargs):
        captured.append(list(args))
        return _make_proc(returncode=0)

    with patch("ovms_rig.stages.start.launch.subprocess.Popen", side_effect=fake_popen):
        from ovms_rig.stages.start.launch import run
        run(rig)

    assert cache.is_dir(), "launch should have created the cache directory"
    cmd = captured[0]
    assert "--cache_dir" in cmd
    assert cmd[cmd.index("--cache_dir") + 1] == str(cache)
