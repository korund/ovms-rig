"""Tests for smoke-load probe validation.

Tests verify:
- Success: OVMS accepts config, all N graphs initialize
- Fail: libprotobuf ERROR, LOADING_PRECONDITION_FAILED, mediapipe parse fail
- Timeout: 30s elapsed, no graphs initialized
- Empty profile: no-op, status ok
- OVMS binary not resolved: status warn
- Process cleanup on exception
- Command builder is called with log_level_override="DEBUG"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from ovms_rig.config import Declaration, LocalConfig, LocalModels, LocalRuntime, OvmsConfig, Runtime
from ovms_rig.config.schema import Graph, ModelEntry, ModelIdentity, Profile
from ovms_rig.probes import smoke_load


def mock_config_exists():
    """Patch that makes config.json appear to exist for tests."""
    original_is_file = Path.is_file

    def patched_is_file(self):
        if self.name == "config.json":
            return True
        return original_is_file(self)

    return patch.object(Path, "is_file", patched_is_file)


def _decl(num_models: int = 0, active_profile: str | None = None) -> Declaration:
    """Create a Declaration for testing."""
    ovms = OvmsConfig(
        runtime=Runtime(rest_port=8000),
        repository={
            "main": ModelIdentity(hf="test/model", task="text_generation"),
        },
        models={
            f"model{i}": ModelEntry(
                source="main",
                graph=Graph(device="GPU"),
                generation=None,
            )
            for i in range(num_models)
        },
        profiles={
            "default": Profile(
                models=[f"model{i}" for i in range(num_models)],
                active=(active_profile == "default"),
            ),
        },
    )
    local = LocalConfig(
        runtime=LocalRuntime(ovms_path=None, cache_dir=None),
        models=LocalModels(repository_path=Path("/store")),
    )
    return Declaration(ovms=ovms, local=local, cli_override=None)


def test_empty_profile_returns_ok():
    """No active profile -> status ok, no probe."""
    decl = _decl(num_models=0, active_profile=None)
    result = smoke_load.check(decl)
    assert result.status == "ok"
    assert "no active profile" in result.summary


def test_binary_not_resolved_returns_warn(monkeypatch):
    """OVMS binary not resolved -> status warn."""
    decl = _decl(num_models=1, active_profile="default")
    monkeypatch.setattr("ovms_rig.probes.smoke_load.ovms_binary.resolve", lambda *a, **k: (None, None))
    result = smoke_load.check(decl)
    assert result.status == "warn"
    assert "binary not resolved" in result.summary


def test_success_single_graph():
    """Single graph initializes -> status ok."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "[2026-05-25 10:00:01] MediapipeGraphDefinition initializing graph nodes",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "ok"
    assert "graph(s) validated" in result.summary


def test_success_multiple_graphs():
    """Two graphs both initialize -> status ok."""
    decl = _decl(num_models=2, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "[2026-05-25 10:00:01] MediapipeGraphDefinition initializing graph nodes",
        "[2026-05-25 10:00:02] MediapipeGraphDefinition initializing graph nodes",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, None, None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "ok"


def test_fail_on_libprotobuf_error():
    """libprotobuf ERROR -> status error with fail_markers."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        '[libprotobuf ERROR] Error parsing "kv_cache_precision"',
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "error"
    assert "rejected" in result.summary.lower()
    assert "fail_markers" in result.details
    assert any("[libprotobuf ERROR" in m for m in result.details["fail_markers"])


def test_fail_on_loading_precondition_failed():
    """LOADING_PRECONDITION_FAILED -> status error."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "[2026-05-25 10:00:01] state changed to: LOADING_PRECONDITION_FAILED",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "error"
    assert any("LOADING_PRECONDITION_FAILED" in m for m in result.details["fail_markers"])


def test_fail_on_mediapipe_parse_error():
    """Mediapipe graph parse failure -> status error."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "Trying to parse mediapipe graph definition: graph.test.pbtxt - failed",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "error"
    assert any("mediapipe" in m.lower() for m in result.details["fail_markers"])


def test_timeout_no_graphs_initialized():
    """Timeout when no graphs initialized -> status error."""
    decl = _decl(num_models=1, active_profile="default")

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    call_count = [0]
                    def read_data(*args, **kwargs):
                        class FakeFile:
                            def __enter__(self):
                                return self
                            def __exit__(self, *args):
                                pass
                            def readline(self):
                                call_count[0] += 1
                                if call_count[0] <= 35:
                                    return f"[DEBUG] Waiting {call_count[0]}...\n"
                                return ""
                        return FakeFile()

                    with patch("builtins.open", side_effect=read_data):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.return_value = None
                            mock_proc.wait = MagicMock()
                            mock_popen_cls.return_value = mock_proc

                            with patch("subprocess.run") as mock_run:
                                mock_run.return_value = MagicMock(returncode=0)

                                with patch("ovms_rig.probes.smoke_load.time") as mock_time:
                                    time_vals = [0]
                                    def advancing_time():
                                        val = time_vals[0]
                                        time_vals[0] += 0.5
                                        return float(val)
                                    mock_time.time.side_effect = advancing_time
                                    mock_time.sleep = MagicMock()

                                    result = smoke_load.check(decl)

    assert result.status == "error"
    assert "timed out" in result.summary
    assert "log_tail" in result.details
    assert len(result.details["log_tail"]) > 0


def test_log_tail_attached_on_timeout():
    """Timeout result includes log_tail with recent non-marker lines."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[INFO] loading model X.bin",
        "[INFO] model loading started",
        "[ERROR] field 'foo' unknown",
        "[ERROR] config parsing failed",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    call_count = [0]
                    def read_data(*args, **kwargs):
                        class FakeFile:
                            def __enter__(self):
                                return self
                            def __exit__(self, *args):
                                pass
                            def readline(self):
                                call_count[0] += 1
                                if call_count[0] <= len(output_lines):
                                    return output_lines[call_count[0] - 1] + "\n"
                                return ""
                        return FakeFile()

                    with patch("builtins.open", side_effect=read_data):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.return_value = None
                            mock_proc.wait = MagicMock()
                            mock_popen_cls.return_value = mock_proc

                            with patch("subprocess.run") as mock_run:
                                mock_run.return_value = MagicMock(returncode=0)

                                with patch("ovms_rig.probes.smoke_load.time") as mock_time:
                                    time_vals = [0]
                                    def advancing_time():
                                        val = time_vals[0]
                                        time_vals[0] += 0.5
                                        return float(val)
                                    mock_time.time.side_effect = advancing_time
                                    mock_time.sleep = MagicMock()

                                    result = smoke_load.check(decl)

    assert result.status == "error"
    assert "timed out" in result.summary
    assert "log_tail" in result.details
    assert result.details["log_tail"] == output_lines


def test_process_cleaned_up_on_exception():
    """Process is killed even if exception occurs during read."""
    decl = _decl(num_models=1, active_profile="default")

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    def read_error(*args, **kwargs):
                        class FailingFile:
                            def __enter__(self):
                                return self
                            def __exit__(self, *args):
                                pass
                            def readline(self):
                                raise RuntimeError("Read error")
                        return FailingFile()

                    with patch("builtins.open", side_effect=read_error):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.return_value = None
                            mock_proc.wait = MagicMock()
                            mock_popen_cls.return_value = mock_proc

                            with patch("subprocess.run") as mock_run:
                                mock_run.return_value = MagicMock(returncode=0)

                                result = smoke_load.check(decl)

    assert result.status == "error"
    assert mock_proc.poll.called


def test_fail_after_partial_success():
    """N=2 expected, 1 initialized, then fail-marker -> error."""
    decl = _decl(num_models=2, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "[2026-05-25 10:00:01] MediapipeGraphDefinition initializing graph nodes",
        "[2026-05-25 10:00:02] state changed to: LOADING_PRECONDITION_FAILED",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen") as mock_popen_cls:
                            mock_proc = MagicMock()
                            mock_proc.pid = 12345
                            mock_proc.poll.side_effect = [None, None, 0]
                            mock_popen_cls.return_value = mock_proc

                            result = smoke_load.check(decl)

    assert result.status == "error"
    assert "fail_markers" in result.details


def test_probe_command_includes_log_level_debug():
    """Probe command includes --log_level DEBUG (override) and probe-added --log_path."""
    decl = _decl(num_models=1, active_profile="default")
    output_lines = [
        "[2026-05-25 10:00:00] Loading config",
        "[2026-05-25 10:00:01] MediapipeGraphDefinition initializing graph nodes",
    ]

    mock_binary = MagicMock(spec=Path)
    mock_binary.is_file.return_value = True

    captured_popen_args = []

    def capture_popen(args, **kwargs):
        captured_popen_args.append({"args": list(args), "kwargs": kwargs})
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0
        return mock_proc

    with mock_config_exists():
        with patch("ovms_rig.probes.smoke_load.ovms_binary.resolve", return_value=(mock_binary, "test")):
            with patch("ovms_rig.probes.smoke_load.build_env", return_value={}):
                with patch("tempfile.NamedTemporaryFile") as mock_tempfile:
                    temp_file = MagicMock()
                    temp_file.name = "/tmp/test.log"
                    mock_tempfile.return_value = temp_file

                    with patch("builtins.open", mock_open(read_data="\n".join(output_lines) + "\n")):
                        with patch("subprocess.Popen", side_effect=capture_popen):
                            result = smoke_load.check(decl)

    assert result.status == "ok"
    assert len(captured_popen_args) == 1
    cmd = captured_popen_args[0]["args"]
    assert "--log_level" in cmd
    idx = cmd.index("--log_level")
    assert cmd[idx + 1] == "DEBUG"
    assert "--log_path" in cmd
    idx = cmd.index("--log_path")
    assert cmd[idx + 1] == "/tmp/test.log"
