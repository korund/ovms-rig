"""Fetch stage tests: command composition + idempotency.

Subprocess is mocked; we never actually invoke ovms. Tests assert what
args fetch would pass to `ovms --pull`, in what order, and per which
model.
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
  ovms_version: ">=2025"
  rest_port: 8000
  log_level: DEBUG

models:
  main:
    hf: org/main-int8-ov
    task: text_generation
  draft:
    hf: org/draft-int8-ov
    task: text_generation

served:
  - name: ep
    model: main
    graph:
      device: GPU
      max_num_seqs: 256
      enable_prefix_caching: true
      draft_model: draft
      draft_device: CPU
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


def _invoke(rig: dict, *extras: str) -> "object":
    runner = CliRunner()
    return runner.invoke(
        main,
        ["--config", str(rig["config"]),
         "--local", str(rig["local"]),
         "--ovms-path", sys.executable,
         "fetch",
         *extras],
        catch_exceptions=False,
    )


def test_fetch_pulls_target_with_pull_flags_and_draft_bare(
    rig: dict, recorder: Recorder
) -> None:
    result = _invoke(rig)
    assert result.exit_code == 0
    assert len(recorder.calls) == 2

    # Calls are sorted by model short name -> draft first, then main.
    draft_call, main_call = recorder.calls
    assert "--source_model" in draft_call
    assert draft_call[draft_call.index("--source_model") + 1] == "org/draft-int8-ov"
    assert "--task" in draft_call
    assert draft_call[draft_call.index("--task") + 1] == "text_generation"
    # Draft is pulled bare: pull-bucket flags from served.graph must NOT leak in.
    assert "--max_num_seqs" not in draft_call
    assert "--enable_prefix_caching" not in draft_call

    # Target carries pull-bucket flags. Pbtxt-only fields (device, draft_*) stay out.
    assert main_call[main_call.index("--source_model") + 1] == "org/main-int8-ov"
    assert main_call[main_call.index("--max_num_seqs") + 1] == "256"
    assert main_call[main_call.index("--enable_prefix_caching") + 1] == "true"
    assert "--device" not in main_call
    assert "--draft_device" not in main_call
    assert "--draft_model" not in main_call
    assert "--draft_source_model" not in main_call


def test_fetch_skips_already_present_models(
    rig: dict, recorder: Recorder
) -> None:
    # Pre-create the target dir at the HF path -- inventory should see it.
    (rig["store"] / "org" / "main-int8-ov").mkdir(parents=True)

    result = _invoke(rig)
    assert result.exit_code == 0
    assert len(recorder.calls) == 1
    only_call = recorder.calls[0]
    assert only_call[only_call.index("--source_model") + 1] == "org/draft-int8-ov"


def test_fetch_returns_nonzero_when_pull_fails(
    rig: dict, recorder: Recorder
) -> None:
    recorder.returncode = 7
    result = _invoke(rig)
    assert result.exit_code == 1


def test_fetch_forwards_extra_args_to_pull(
    rig: dict, recorder: Recorder
) -> None:
    result = _invoke(rig, "--overwrite_models")
    assert result.exit_code == 0
    for call in recorder.calls:
        assert call[-1] == "--overwrite_models"


def test_fetch_errors_when_store_path_unset(tmp_path: Path, recorder: Recorder) -> None:
    # repository_path missing now fails at config load, before any pull runs.
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(
        "runtime:\n  ovms_path: null\nmodels: {}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--config", str(cfg),
         "--local", str(loc),
         "--ovms-path", sys.executable,
         "fetch"],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert recorder.calls == []
