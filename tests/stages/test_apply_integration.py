"""Integration tests for the profile activation stage.

Tests verify:
- pristine graph.pbtxt is never mutated.
- sibling copy (graph.<model_name>.pbtxt) is created with patches applied.
- config.json is written with mediapipe_config_list entries.
- dry-run writes to build/, does not touch live files (via apply internal).
- live run writes to store and backs up config.json.
- Profile activation/deactivation works correctly.
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

profiles:
  default:
    models: [ep]
    active: true
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
            *extra_args,
        ],
        catch_exceptions=False,
    )


def _invoke_apply_directly(rig: dict, dry_run: bool = False) -> int:
    """Call apply.run() directly with prepared context."""
    from ovms_rig.stages.activation import apply
    ctx = {
        "config_path": str(rig["config"]),
        "local_path": str(rig["local"]),
        "ovms_path": sys.executable,
        "log_level": None,
        "dry_run": dry_run,
        "extras": [],
    }
    return apply.run(ctx)


# ---------------------------------------------------------------------------
# Dry-run tests (via activate with internal apply --dry-run analogue)
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_writes_sibling_to_build_not_live(self, rig: dict,
                                                      monkeypatch: pytest.MonkeyPatch):
        """Dry-run writes sibling to build/, not to live store."""
        monkeypatch.chdir(rig["tmp"])
        rc = _invoke_apply_directly(rig, dry_run=True)
        assert rc == 0

        # Sibling-copy in build/ must exist.
        build_sibling = rig["tmp"] / "build" / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        assert build_sibling.exists(), f"Expected {build_sibling} in build/"

        # Sibling must NOT be in live store.
        live_sibling = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        assert not live_sibling.exists(), f"Sibling should not exist in live store during dry-run"

        # Pristine pbtxt must be UNCHANGED.
        pristine = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        pristine_content = pristine.read_text(encoding="utf-8")
        assert 'device: "CPU"' in pristine_content, "Pristine should not be modified"

    def test_dry_run_no_config_json_in_store(self, rig: dict,
                                             monkeypatch: pytest.MonkeyPatch):
        """Dry-run does not create config.json in live store."""
        monkeypatch.chdir(rig["tmp"])
        rc = _invoke_apply_directly(rig, dry_run=True)
        assert rc == 0

        # Live config.json must NOT be created.
        live_config = rig["store"] / "config.json"
        assert not live_config.exists(), "config.json should not be created in live store during dry-run"

        # Build config.json should exist.
        build_config = rig["tmp"] / "build" / "config.json"
        assert build_config.exists(), "config.json should be in build/"

    def test_dry_run_no_backups_created(self, rig: dict,
                                        monkeypatch: pytest.MonkeyPatch):
        """Dry-run does not create backup files."""
        monkeypatch.chdir(rig["tmp"])
        rc = _invoke_apply_directly(rig, dry_run=True)
        assert rc == 0

        # No backups in store.
        backups = list(rig["store"].glob("*.bak.*"))
        assert backups == [], f"No backups should be created during dry-run, found: {backups}"

    def test_dry_run_with_generation_overrides_missing_file(self, tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch):
        """Dry-run with generation overrides and missing pristine generation_config.json doesn't crash."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Create model directory with graph.pbtxt but NO generation_config.json.
        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: DEBUG

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

profiles:
  default:
    models: [ep]
    active: true
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)

        # Call apply directly with dry_run=True.
        from ovms_rig.stages.activation import apply
        ctx = {
            "config_path": str(cfg),
            "local_path": str(loc),
            "ovms_path": sys.executable,
            "log_level": None,
            "dry_run": True,
            "extras": [],
        }
        rc = apply.run(ctx)
        assert rc == 0, "Dry-run should succeed even without generation_config.json"

        # No files should be written to live store.
        assert not (model_dir / "generation_config.json").exists()
        # But build/ can have the proposed config.
        build_path = tmp_path / "build"
        # (may or may not exist depending on whether any files were written to build)


# ---------------------------------------------------------------------------
# Live run tests (via activate)
# ---------------------------------------------------------------------------

class TestLiveRun:
    def test_live_run_creates_sibling_copy(self, rig: dict,
                                          monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig, "activate", "default")
        assert result.exit_code == 0
        # Sibling-copy in store must exist.
        sibling = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.ep.pbtxt"
        assert sibling.exists(), f"Expected {sibling} to exist"
        content = sibling.read_text(encoding="utf-8")
        assert 'device: "GPU"' in content

    def test_live_run_pristine_unchanged(self, rig: dict,
                                        monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        # Record pristine content before activate.
        pristine_path = rig["store"] / "OpenVINO" / "main-int8-ov" / "graph.pbtxt"
        pristine_before = pristine_path.read_text(encoding="utf-8")

        _invoke(rig, "activate", "default")

        # Pristine must be completely unchanged.
        pristine_after = pristine_path.read_text(encoding="utf-8")
        assert pristine_before == pristine_after, "Pristine graph.pbtxt should not be modified"

    def test_live_run_registers_in_config_json(self, rig: dict,
                                              monkeypatch: pytest.MonkeyPatch):
        monkeypatch.chdir(rig["tmp"])
        result = _invoke(rig, "activate", "default")
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

        result = _invoke(rig, "activate", "default")
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
        result = _invoke(rig, "activate", "default")
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Generation config tests (via activate)
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # generation_config.json must be unchanged.
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.read_text(encoding="utf-8") == original_genconfig

    def test_generation_config_empty_dict_skipped(self, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch):
        """Model with empty generation dict {} skips generation_config.json (no overrides)."""
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

profiles:
  default:
    models: [ep]
    active: true
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
                "activate",
                "default",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

        # generation_config.json must be unchanged (empty dict is treated as no overrides).
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.read_text(encoding="utf-8") == original_genconfig

        # No backup should be created (because no overrides were applied).
        genconfig_backups = list(model_dir.glob("generation_config.json.bak.*"))
        assert genconfig_backups == [], "No backup should be created for empty generation dict"


# ---------------------------------------------------------------------------
# Profile-aware activation/deactivation tests
# ---------------------------------------------------------------------------

def test_deactivate_produces_empty_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Deactivate command sets no active profile -> config.json has empty mediapipe_config_list."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

    cfg.write_text(OVMS_YAML, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config", str(cfg),
            "--local", str(loc),
            "--ovms-path", sys.executable,
            "deactivate",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    config_json = store / "config.json"
    assert config_json.exists()
    data = json.loads(config_json.read_text(encoding="utf-8"))
    assert data.get("mediapipe_config_list") == []


def test_apply_partial_failure_missing_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Partial failure: one model missing, config.json reflects only successful models."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # Create only main model directory (draft will be missing).
    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

    # YAML with two models but only one directory exists.
    ovms_yaml = """\
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
  draft_ep:
    source: draft
    graph:
      device: CPU

profiles:
  default:
    models: [ep, draft_ep]
    active: true
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    from ovms_rig.stages.activation import apply
    ctx = {
        "config_path": str(cfg),
        "local_path": str(loc),
        "ovms_path": sys.executable,
        "log_level": None,
        "dry_run": False,
        "extras": [],
    }
    rc = apply.run(ctx)

    # Should fail (rc=1) because one model directory is missing.
    assert rc == 1

    # Check config.json state: apply may have reconciled before failure or skipped it.
    # After audit fix, apply should fail early on missing model_dir, before reconcile.
    config_json = store / "config.json"
    if config_json.exists():
        # If config.json exists, it should only contain the successful model.
        data = json.loads(config_json.read_text(encoding="utf-8"))
        entries = data.get("mediapipe_config_list", [])
        # Should be empty or contain only 'ep', not 'draft_ep'.
        names = {e["name"] for e in entries if "name" in e}
        assert "draft_ep" not in names, "Failed model should not be in config.json"


def test_activate_different_profile_cleans_up_old_sibling_graphs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Activate different profile removes sibling graphs from previous profile."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

    # YAML with two profiles: default active, other inactive.
    ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: DEBUG

repository:
  main:
    hf: OpenVINO/main-int8-ov
    task: text_generation

models:
  ep:
    source: main
    graph:
      device: GPU

profiles:
  default:
    models: [ep]
    active: true
  other:
    models: [ep]
    active: false
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    # First activate default (already active, but do it explicitly).
    result = runner.invoke(
        main,
        [
            "--config", str(cfg),
            "--local", str(loc),
            "--ovms-path", sys.executable,
            "activate",
            "default",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # Sibling created for 'default' profile.
    assert (model_dir / "graph.ep.pbtxt").exists()

    # Create an obsolete sibling graph (from hypothetical past activation).
    (model_dir / "graph.old_model.pbtxt").write_text("obsolete", encoding="utf-8")
    (model_dir / "graph.another.pbtxt").write_text("obsolete", encoding="utf-8")

    # Now activate 'other' profile.
    result = runner.invoke(
        main,
        [
            "--config", str(cfg),
            "--local", str(loc),
            "--ovms-path", sys.executable,
            "activate",
            "other",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0

    # Obsolete sibling graphs should be deleted.
    assert not (model_dir / "graph.old_model.pbtxt").exists()
    assert not (model_dir / "graph.another.pbtxt").exists()
    # Current sibling graph should still exist (both profiles have [ep]).
    assert (model_dir / "graph.ep.pbtxt").exists()
