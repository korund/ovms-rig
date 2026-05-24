"""Fetch stage tests: single repository entry pull.

Tests verify fetch pulls a single repository entry with correct args.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from ovms_rig.cli import main

OVMS_YAML = """
runtime:
  rest_port: 8000
  log_level: DEBUG

repository:
  qwen-14b:
    hf: org/qwen-14b-int8-ov
    task: text_generation
  qwen-draft:
    hf: org/qwen-draft-int8-ov
    task: text_generation
"""

LOCAL_YAML = """
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""


@dataclass
class Recorder:
    calls: list[list[str]] = field(default_factory=list)
    returncode: int = 0

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(args=args, returncode=self.returncode)


@pytest.fixture
def rig(tmp_path: Path) -> dict:
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()
    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")
    return {"config": cfg, "local": loc, "store": store}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr("ovms_rig.stages.fetch.subprocess.run", rec)
    return rec


def _invoke(rig: dict, *args: str) -> "object":
    runner = CliRunner()
    return runner.invoke(
        main,
        ["--config", str(rig["config"]),
         "--local", str(rig["local"]),
         "--ovms-path", sys.executable,
         "fetch",
         *args],
        catch_exceptions=False,
    )


def test_fetch_pulls_single_entry(rig: dict, recorder: Recorder) -> None:
    """Fetch pulls a single repository entry with correct args."""
    result = _invoke(rig, "qwen-14b")
    assert result.exit_code == 0
    assert len(recorder.calls) == 1

    call = recorder.calls[0]
    assert "--pull" in call
    assert "--source_model" in call
    assert "org/qwen-14b-int8-ov" in call
    assert "--task" in call
    assert "text_generation" in call
    assert "--model_repository_path" in call
    assert str(rig["store"]) in call


def test_fetch_already_present_skips(rig: dict, recorder: Recorder) -> None:
    """Fetch skips if model directory already exists."""
    # Create the model directory beforehand.
    model_dir = rig["store"] / "org" / "qwen-14b-int8-ov"
    model_dir.mkdir(parents=True)

    result = _invoke(rig, "qwen-14b")
    assert result.exit_code == 0
    assert len(recorder.calls) == 0  # No pull call made.


def test_fetch_unknown_entry_fails(rig: dict, recorder: Recorder) -> None:
    """Fetch fails if repository entry doesn't exist."""
    result = _invoke(rig, "nonexistent")
    assert result.exit_code == 1
    assert len(recorder.calls) == 0  # No pull call made.
    assert "not found" in result.output or "nonexistent" in result.output


def test_fetch_passes_through_extras(rig: dict, recorder: Recorder) -> None:
    """Fetch forwards extra args to ovms --pull."""
    result = _invoke(rig, "qwen-14b", "--", "--foo", "bar", "--baz")
    assert result.exit_code == 0
    assert len(recorder.calls) == 1

    call = recorder.calls[0]
    assert "--foo" in call
    assert "bar" in call
    assert "--baz" in call
