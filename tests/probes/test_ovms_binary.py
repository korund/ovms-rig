"""Resolver for the ovms binary path."""

from __future__ import annotations

import sys
from pathlib import Path

from ovms_rig.probes.ovms_binary import check
from ovms_rig.config import LocalConfig, LocalModels, LocalRuntime


def _local(ovms_path: Path | None = None) -> LocalConfig:
    return LocalConfig(
        runtime=LocalRuntime(ovms_path=ovms_path),
        models=LocalModels(repository_path=Path("C:/unused-by-binary-probe")),
    )


def test_cli_override_wins(tmp_path: Path) -> None:
    exe = Path(sys.executable)
    result = check(cli_override=exe, local=_local(ovms_path=tmp_path / "ignored"))
    assert result.status == "ok"
    assert result.details["source"] == "cli"
    assert result.summary == str(exe)


def test_local_yaml_used_when_no_cli(tmp_path: Path) -> None:
    exe = Path(sys.executable)
    result = check(cli_override=None, local=_local(ovms_path=exe))
    assert result.status == "ok"
    assert result.details["source"] == "local.yaml"


def test_error_when_path_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = check(cli_override=None, local=_local())
    assert result.status == "error"
    assert "not found" in result.summary


def test_error_when_resolved_path_is_directory(tmp_path: Path) -> None:
    result = check(cli_override=tmp_path, local=_local())
    assert result.status == "error"
    assert "not a file" in result.summary


def test_path_lookup_via_which(tmp_path: Path, monkeypatch) -> None:
    """CLI override and local.yaml absent, but shutil.which finds ovms via PATH."""
    binary = tmp_path / "ovms"
    binary.write_text("", encoding="utf-8")
    binary.chmod(0o755)  # make executable; may be no-op on Windows but harmless

    monkeypatch.setattr("shutil.which", lambda _: str(binary))
    result = check(cli_override=None, local=_local())
    assert result.status == "ok"
    assert result.details["source"] == "PATH"
    assert result.summary == str(binary)


def test_error_when_binary_not_executable(tmp_path: Path, monkeypatch) -> None:
    """Binary exists and is a file, but not executable -> error."""
    binary_file = tmp_path / "ovms"
    binary_file.write_text("not executable", encoding="utf-8")

    # Mock os.access to return False for X_OK check.
    def mock_access(path, mode):
        # Only return False for X_OK checks on our file.
        if path == binary_file and mode == 1:  # os.X_OK == 1
            return False
        # For exists checks and other modes, delegate to real os.access.
        import os
        return os.access(path, mode)

    monkeypatch.setattr("os.access", mock_access)
    result = check(cli_override=binary_file, local=_local())
    assert result.status == "error"
    assert "executable" in result.summary.lower() or "executable" in (result.hint or "").lower()
