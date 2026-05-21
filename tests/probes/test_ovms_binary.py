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
