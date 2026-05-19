"""Tests for env.bootstrap: 4 combos (win32/linux x python-on/off)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ovms_rig.env import build_env


def _layout(root: Path, *, platform: str, python_on: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if platform == "win32":
        if python_on:
            (root / "python" / "Scripts").mkdir(parents=True, exist_ok=True)
    else:
        (root / "bin").mkdir(exist_ok=True)
        (root / "lib").mkdir(exist_ok=True)
        if python_on:
            (root / "lib" / "python").mkdir(exist_ok=True)
    return root


def test_win32_python_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    ovms = _layout(tmp_path / "ovms", platform="win32", python_on=True)
    env = build_env(ovms)

    assert env["PYTHONHOME"] == str(ovms / "python")
    path_head = env["PATH"].split(os.pathsep)[:3]
    assert path_head == [str(ovms), str(ovms / "python"), str(ovms / "python" / "Scripts")]


def test_win32_python_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.platform", "win32")
    ovms = _layout(tmp_path / "ovms", platform="win32", python_on=False)
    env = build_env(ovms)

    assert "PYTHONHOME" not in env or env.get("PYTHONHOME") == os.environ.get("PYTHONHOME")
    assert env["PATH"].split(os.pathsep)[0] == str(ovms)


def test_linux_python_on(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    ovms = _layout(tmp_path / "ovms", platform="linux", python_on=True)
    env = build_env(ovms)

    assert env["PYTHONPATH"] == str(ovms / "lib" / "python")
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(ovms / "lib")
    assert env["PATH"].split(os.pathsep)[0] == str(ovms / "bin")


def test_linux_python_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    ovms = _layout(tmp_path / "ovms", platform="linux", python_on=False)
    env = build_env(ovms)

    # Bundled python absent -> PYTHONPATH not added by build_env.
    assert env.get("PYTHONPATH") == os.environ.get("PYTHONPATH")
    assert env["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(ovms / "lib")
    assert env["PATH"].split(os.pathsep)[0] == str(ovms / "bin")


def test_path_prepends_not_replaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    ovms = _layout(tmp_path / "ovms", platform="linux", python_on=False)
    env = build_env(ovms)
    assert env["PATH"] == f"{ovms / 'bin'}{os.pathsep}/usr/bin:/bin"
