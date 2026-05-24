"""Integration tests for the apply stage.

Tests verify:
- pristine graph.pbtxt is never mutated.
- sibling copy (graph.<model_name>.pbtxt) is created with patches applied.
- config.json is written with mediapipe_config_list entries.
- dry-run writes to build/, does not touch live files.
- live run writes to store and backs up config.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ovms_rig.cli import main

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OVMS_YAML = """
runtime:
  rest_port: 8000
  log_level: DEBUG

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation
  draft:
    hf: OpenVINO/draft-int8-ov
    task: text_generation

models:
  ep:
    source: main
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

# Sample generation_config.json for a HuggingFace model.
GENERATION_CONFIG = """\
{
  "architectures": ["QwenForCausalLM"],
  "temperature": 1.0,
  "top_p": 0.999,
  "max_length": 4096
}
"""


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
    def test_dry_run_writes_sibling_to_build_not_live(self, rig: dict,
                                                      monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig, "--dry-run")
        assert result.exit_code == 0

        # Sibling-copy in build/ must exist with served name.
        build_sibling = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        assert build_sibling.exists(), f"Expected {build_sibling} to exist"

        # Pristine pbtxt must be UNCHANGED in live store.
        pristine = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        pristine_content = pristine.read_text(encoding="utf-8")
        assert 'device: "CPU"' in pristine_content, "Pristine should not be modified"

    def test_dry_run_no_backup(self, rig: dict,
                               monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        # Dry-run must not create backups in store.
        store = rig["store"]
        backups = list(store.glob("*.bak.*"))
        assert backups == [], f"Expected no backups in store, found: {backups}"

    def test_dry_run_sibling_has_gpu_device(self, rig: dict,
                                            monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        sibling = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        content = sibling.read_text(encoding="utf-8")
        assert 'device: "GPU"' in content

    def test_dry_run_sibling_has_draft_path(self, rig: dict,
                                            monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        sibling = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        content = sibling.read_text(encoding="utf-8")
        # draft_models_path must be present and point to draft dir
        assert "draft_models_path" in content
        assert "draft-int8-ov" in content

    def test_dry_run_creates_config_json(self, rig: dict,
                                         monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        _invoke(rig, "--dry-run")
        config_path = rig["tmp"] / "build" / "config.json"
        assert config_path.exists(), f"Expected {config_path} to exist"
        # Verify JSON structure.
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert "mediapipe_config_list" in cfg
        entries = cfg["mediapipe_config_list"]
        assert len(entries) == 1
        assert entries[0]["name"] == "ep"
        assert "graph.ep.pbtxt" in entries[0]["graph_path"]


# ---------------------------------------------------------------------------
# Live run tests
# ---------------------------------------------------------------------------

class TestLiveRun:
    def test_live_run_creates_sibling_copy(self, rig: dict,
                                           monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig)
        assert result.exit_code == 0
        # Sibling-copy in store must exist.
        sibling = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        assert sibling.exists(), f"Expected {sibling} to exist"
        content = sibling.read_text(encoding="utf-8")
        assert 'device: "GPU"' in content

    def test_live_run_pristine_unchanged(self, rig: dict,
                                         monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        # Record pristine content before apply.
        pristine_path = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        pristine_before = pristine_path.read_text(encoding="utf-8")

        _invoke(rig)

        # Pristine must be completely unchanged.
        pristine_after = pristine_path.read_text(encoding="utf-8")
        assert pristine_before == pristine_after, "Pristine graph.pbtxt should not be modified"

    def test_live_run_registers_in_config_json(self, rig: dict,
                                               monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig)
        assert result.exit_code == 0
        config_path = rig["store"] / "config.json"
        assert config_path.exists()
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        assert "mediapipe_config_list" in cfg
        entries = cfg["mediapipe_config_list"]
        assert len(entries) == 1
        assert entries[0]["name"] == "ep"
        assert "graph.ep.pbtxt" in entries[0]["graph_path"]

    def test_live_run_config_backup_created(self, rig: dict,
                                            monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        # Pre-populate config.json with known content (original state).
        store = rig["store"]
        config_path = store / "config.json"
        original_content = '{"model_config_list": []}'
        config_path.write_text(original_content, encoding="utf-8")

        result = _invoke(rig)
        assert result.exit_code == 0
        # Backup file must be created next to config.json with suffix.
        backups = list(store.glob("config.json.bak.*"))
        assert len(backups) == 1, f"Expected exactly 1 backup, found: {backups}"
        # Backup content must match original (before apply mutation).
        backup_path = backups[0]
        backup_content = backup_path.read_text(encoding="utf-8")
        assert backup_content == original_content, (
            f"Backup should contain original content, got: {backup_content}"
        )

    def test_live_run_fails_if_model_dir_missing(self, rig: dict,
                                                 monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        import shutil
        shutil.rmtree(rig["store"] / "OpenVINO" / "main-int8-ov")
        result = _invoke(rig)
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Generation config tests
# ---------------------------------------------------------------------------

class TestGenerationConfig:
    def test_generation_config_merge_in_live_run(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
        """Model with generation overrides merges into generation_config.json."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Create model directories with graph.pbtxt and generation_config.json.
        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")

        # Config with generation overrides on the model.
        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
    generation:
      temperature: 0.5
      top_p: 0.95
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config", str(cfg),
                "--local", str(loc),
                "--ovms-path", sys.executable,
                "apply",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # generation_config.json must exist and contain the overrides.
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.exists()
        content = json.loads(live_genconfig.read_text(encoding="utf-8"))
        assert content["temperature"] == 0.5
        assert content["top_p"] == 0.95
        # Original keys must be preserved.
        assert content["architectures"] == ["QwenForCausalLM"]

    def test_generation_config_backup_created(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
        """Backup of generation_config.json is created on live run."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")
        original_genconfig_content = GENERATION_CONFIG

        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
    generation:
      temperature: 0.3
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config", str(cfg),
                "--local", str(loc),
                "--ovms-path", sys.executable,
                "apply",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # Backup must exist next to original with suffix.
        genconfig_backups = list(model_dir.glob("generation_config.json.bak.*"))
        assert len(genconfig_backups) == 1, f"Expected 1 backup, found: {genconfig_backups}"
        # Backup content must match original (before apply mutation).
        backup_path = genconfig_backups[0]
        backup_content = backup_path.read_text(encoding="utf-8")
        assert backup_content == original_genconfig_content, (
            f"Backup should contain original, got: {backup_content}"
        )

    def test_generation_config_missing_file_error(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
        """Apply fails when generation_config.json is missing but overrides declared."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Model directory without generation_config.json.
        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
    generation:
      temperature: 0.5
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config", str(cfg),
                "--local", str(loc),
                "--ovms-path", sys.executable,
                "apply",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 1

    def test_generation_config_none_skipped(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch):
        """Model with generation=None (no overrides) skips generation_config.json."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        original_genconfig = GENERATION_CONFIG
        (model_dir / "generation_config.json").write_text(original_genconfig, encoding="utf-8")

        # Config without generation field on the model.
        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config", str(cfg),
                "--local", str(loc),
                "--ovms-path", sys.executable,
                "apply",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # generation_config.json must be unchanged.
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.read_text(encoding="utf-8") == original_genconfig

    def test_generation_config_empty_dict_skipped(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
        """Model with empty generation dict {} skips generation_config.json."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        original_genconfig = GENERATION_CONFIG
        (model_dir / "generation_config.json").write_text(original_genconfig, encoding="utf-8")

        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU
    generation: {}
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--config", str(cfg),
                "--local", str(loc),
                "--ovms-path", sys.executable,
                "apply",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # generation_config.json must be unchanged.
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.read_text(encoding="utf-8") == original_genconfig
