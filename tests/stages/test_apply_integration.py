"""Integration tests for the apply stage.

Subprocess (ovms --add_to_config) is mocked throughout. Tests verify:
- dry-run writes to build/, does not touch live files.
- live run writes to store and takes backup.
- mtime warning is emitted when pbtxt is newer than marker.
- pass-through extras reach ovms invocation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from click.testing import CliRunner

from ovms_rig.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OVMS_YAML = """
runtime:
  ovms_version: ">=2025"
  rest_port: 8000
  log_level: DEBUG

models:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation
  draft:
    hf: OpenVINO/draft-int8-ov
    task: text_generation

served:
  - name: ep
    model: main
    graph:
      device: GPU
      draft_model: draft
      draft_device: CPU
"""

LOCAL_YAML = """\
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""

# Minimal pbtxt content similar to real OVMS output.
GRAPH_PBTXT = """\
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
        }
    }
"""


@dataclass
class Recorder:
    calls: list[list[str]] = field(default_factory=list)
    returncode: int = 0

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        return subprocess.CompletedProcess(
            args=args, returncode=self.returncode,
            stdout="", stderr="",
        )


@pytest.fixture
def rig(tmp_path: Path) -> dict:
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # Create model directories with graph.pbtxt.
    for hf_path in ("OpenVINO/main-int8-ov", "OpenVINO/draft-int8-ov"):
        d = store / hf_path
        d.mkdir(parents=True)
        (d / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")
    return {"config": cfg, "local": loc, "store": store, "tmp": tmp_path}


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr("ovms_rig.stages.apply.registry.subprocess.run", rec)
    return rec


def _invoke(rig: dict, *extra_args: str) -> object:
    runner = CliRunner()
    return runner.invoke(
        main,
        [
            "--config", str(rig["config"]),
            "--local", str(rig["local"]),
            "--ovms-path", sys.executable,
            "apply",
            *extra_args,
        ],
        catch_exceptions=False,
    )


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_to_build_not_live(self, rig: dict, recorder: Recorder,
                                               monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig, "--dry-run")
        assert result.exit_code == 0

        # build/ pbtxt must exist
        build_pbtxt = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        assert build_pbtxt.exists(), f"Expected {build_pbtxt} to exist"

        # Live pbtxt must be UNCHANGED (still has device: "CPU")
        live_pbtxt = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        assert 'device: "CPU"' in live_pbtxt.read_text(encoding="utf-8")

    def test_dry_run_no_backup(self, rig: dict, recorder: Recorder,
                                monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        backup_root = rig["tmp"] / ".backup"
        # No backup directory should be created in dry-run mode.
        if backup_root.exists():
            import os
            entries = list(backup_root.rglob("*"))
            assert entries == [], f"Expected no backup files, found: {entries}"

    def test_dry_run_pbtxt_has_gpu_device(self, rig: dict, recorder: Recorder,
                                           monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        build_pbtxt = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        content = build_pbtxt.read_text(encoding="utf-8")
        assert 'device: "GPU"' in content

    def test_dry_run_pbtxt_has_draft_path(self, rig: dict, recorder: Recorder,
                                           monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        build_pbtxt = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        content = build_pbtxt.read_text(encoding="utf-8")
        # draft_models_path must be present and point to draft dir
        assert "draft_models_path" in content
        assert "draft-int8-ov" in content

    def test_dry_run_ovms_called_with_build_config(self, rig: dict, recorder: Recorder,
                                                    monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        # --add_to_config path should be inside build/
        add_idx = call.index("--add_to_config") + 1
        assert "build" in call[add_idx]


# ---------------------------------------------------------------------------
# Live run tests
# ---------------------------------------------------------------------------

class TestLiveRun:
    def test_live_run_patches_pbtxt_in_store(self, rig: dict, recorder: Recorder,
                                              monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig)
        assert result.exit_code == 0
        live_pbtxt = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        content = live_pbtxt.read_text(encoding="utf-8")
        assert 'device: "GPU"' in content

    def test_live_run_creates_backup(self, rig: dict, recorder: Recorder,
                                     monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig)
        backup_root = rig["tmp"] / ".backup"
        assert backup_root.exists()
        pbtxt_backups = list(backup_root.rglob("graph.pbtxt"))
        assert len(pbtxt_backups) >= 1

    def test_live_run_ovms_called_with_store_config(self, rig: dict, recorder: Recorder,
                                                     monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig)
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        add_idx = call.index("--add_to_config") + 1
        # config path must be the store config.json, not build/
        assert "build" not in call[add_idx]

    def test_live_run_extras_forwarded_to_ovms(self, rig: dict, recorder: Recorder,
                                                monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig, "--some_extra_flag", "42")
        assert result.exit_code == 0
        call = recorder.calls[0]
        assert "--some_extra_flag" in call
        assert "42" in call

    def test_live_run_fails_if_model_dir_missing(self, rig: dict, recorder: Recorder,
                                                  monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        import shutil
        shutil.rmtree(rig["store"] / "OpenVINO" / "main-int8-ov")
        result = _invoke(rig)
        assert result.exit_code == 1

    def test_live_run_returns_nonzero_when_registry_fails(
            self, rig: dict, recorder: Recorder, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        recorder.returncode = 1
        result = _invoke(rig)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Mtime warning test
# ---------------------------------------------------------------------------

class TestMtimeWarning:
    def test_warns_when_pbtxt_newer_than_marker(self, rig: dict, recorder: Recorder,
                                                 monkeypatch: pytest.MonkeyPatch,
                                                 caplog: pytest.LogCaptureFixture):
        monkeypatch.chdir(rig["tmp"])

        # Write a marker with a very old timestamp for the main model.
        cache_dir = rig["tmp"] / ".cache"
        cache_dir.mkdir()
        marker_data = {"main": 1000.0}  # epoch 1970 -- always older than actual mtime
        (cache_dir / "last_apply.json").write_text(
            json.dumps(marker_data), encoding="utf-8"
        )

        # Monkeypatch marker module to use our tmp .cache dir.
        import ovms_rig.stages.apply.marker as marker_mod
        monkeypatch.setattr(marker_mod, "_CACHE_DIR", cache_dir)
        monkeypatch.setattr(marker_mod, "_MARKER_FILE", cache_dir / "last_apply.json")

        import logging
        with caplog.at_level(logging.WARNING, logger="ovms_rig.stages.apply.marker"):
            result = _invoke(rig)

        assert result.exit_code == 0
        # Warning must mention re-applying.
        warning_texts = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("regenerated" in t or "re-applying" in t for t in warning_texts), (
            f"Expected stale-pbtxt warning, got: {warning_texts}"
        )
