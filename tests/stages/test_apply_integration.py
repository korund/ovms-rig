"""Integration tests for the profile activation stage.

Tests verify:
- pristine graph.pbtxt is never mutated.
- sibling copy (graph.<model_name>.pbtxt) is created with patches applied.
- config.json is written with mediapipe_config_list entries.
- dry-run writes to build/, does not touch live files (via apply internal).
- live run writes to store and derives generation_config.json from .orig.
- Profile activation/deactivation works correctly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ovms_rig.cli import main
from ovms_rig.report import CheckResult

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
    device: GPU
    graph:
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
    # Mock smoke_load.check so it always succeeds without actually running OVMS.
    from ovms_rig.report import CheckResult

    def ok_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="ok",
            summary="validation passed",
        )

    with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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
    device: GPU
    graph: {}
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
    def test_generation_config_reads_from_orig(self, tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch):
        """Apply reads generation_config.json from .orig (pristine) and merges overrides."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Create model directories with graph.pbtxt and generation_config.json.
        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")
        # Create .orig (simulating C1 fetch behavior).
        (model_dir / "generation_config.json.orig").write_text(GENERATION_CONFIG, encoding="utf-8")

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
    device: GPU
    graph: {}
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

        def ok_smoke_check(decl):
            return CheckResult(
                name="smoke-load",
                status="ok",
                summary="validation passed",
            )

        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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

        # generation_config.json must exist and contain the overrides (derived from .orig).
        live_genconfig = model_dir / "generation_config.json"
        assert live_genconfig.exists()
        content = json.loads(live_genconfig.read_text(encoding="utf-8"))
        assert content["temperature"] == 0.5
        assert content["top_p"] == 0.95
        # Original keys from .orig must be preserved.
        assert content["architectures"] == ["QwenForCausalLM"]

    def test_generation_config_missing_orig_fails(self, tmp_path: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
        """Apply fails when generation_config.json.orig is missing (requires fetch first)."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Model directory without .orig (and without generation_config.json too).
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
    device: GPU
    graph: {}
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
        # Don't patch smoke_load here, we expect this to fail during apply.
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
        # Error message should mention .orig and fetch.
        assert "generation_config.json.orig" in result.output or "fetch" in result.output.lower()

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
    device: GPU
    graph: {}

profiles:
  default:
    models: [ep]
    active: true
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        def ok_smoke_check(decl):
            return CheckResult(
                name="smoke-load",
                status="ok",
                summary="validation passed",
            )

        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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
    device: GPU
    graph: {}
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

        def ok_smoke_check(decl):
            return CheckResult(
                name="smoke-load",
                status="ok",
                summary="validation passed",
            )

        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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

    def test_generation_config_idempotent(self, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch):
        """Repeated activate with same profile derives identical generation_config.json (no drift)."""
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        model_dir = store / "OpenVINO" / "main-int8-ov"
        model_dir.mkdir(parents=True)
        (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
        (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")
        (model_dir / "generation_config.json.orig").write_text(GENERATION_CONFIG, encoding="utf-8")

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
    device: GPU
    graph: {}
    generation:
      temperature: 0.7

profiles:
  default:
    models: [ep]
    active: true
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)

        def ok_smoke_check(decl):
            return CheckResult(
                name="smoke-load",
                status="ok",
                summary="validation passed",
            )

        # First activate.
        runner = CliRunner()
        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
            result1 = runner.invoke(
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
        assert result1.exit_code == 0
        content_after_first = (model_dir / "generation_config.json").read_text(encoding="utf-8")

        # Second activate (same profile).
        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
            result2 = runner.invoke(
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
        assert result2.exit_code == 0
        content_after_second = (model_dir / "generation_config.json").read_text(encoding="utf-8")

        # Content must be identical (idempotent, no drift).
        assert content_after_first == content_after_second, \
            "Repeated activate should produce identical generation_config.json (derived from .orig)"


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

    def ok_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="ok",
            summary="validation passed",
        )

    with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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
    config_path = store / "config.json"
    assert config_path.exists()
    cfg_data = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg_data.get("mediapipe_config_list") == []


def test_deactivate_restores_generation_config_from_orig(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch):
    """Deactivate restores generation_config.json from .orig for models with overrides."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
    # Create pristine and live generation_config.json files.
    (model_dir / "generation_config.json.orig").write_text(GENERATION_CONFIG, encoding="utf-8")
    # Live file is modified (simulating activate having applied overrides).
    modified_genconfig = """\
{
  "architectures": ["QwenForCausalLM"],
  "temperature": 0.5,
  "top_p": 0.95,
  "max_length": 4096
}
"""
    (model_dir / "generation_config.json").write_text(modified_genconfig, encoding="utf-8")

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
    device: GPU
    graph: {}
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

    def ok_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="ok",
            summary="validation passed",
        )

    with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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

    # After deactivate, live generation_config.json should be restored to pristine.
    live_genconfig = model_dir / "generation_config.json"
    restored_content = json.loads(live_genconfig.read_text(encoding="utf-8"))
    # Should match .orig (pristine), not the modified version.
    assert restored_content["temperature"] == 1.0, "Should be restored to .orig pristine value"
    assert restored_content["top_p"] == 0.999, "Should be restored to .orig pristine value"


def test_deactivate_missing_orig_warns_no_fail(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch):
    """Deactivate warns if .orig is missing (old model, pre-C1), but doesn't fail."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
    # NO .orig file (simulating old fetch before C1).
    (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")

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
    device: GPU
    graph: {}
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

    def ok_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="ok",
            summary="validation passed",
        )

    with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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

    # Should succeed (exit code 0), not fail.
    assert result.exit_code == 0
    # Live generation_config.json should not be modified (no .orig to restore from).
    live_genconfig = model_dir / "generation_config.json"
    content = live_genconfig.read_text(encoding="utf-8")
    assert content == GENERATION_CONFIG


def test_deactivate_no_overrides_does_not_touch_genconfig(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch):
    """Deactivate does not touch generation_config.json if model has no overrides."""
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")
    (model_dir / "generation_config.json").write_text(GENERATION_CONFIG, encoding="utf-8")
    (model_dir / "generation_config.json.orig").write_text(GENERATION_CONFIG, encoding="utf-8")

    # Config WITHOUT generation overrides on the model.
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
    device: GPU
    graph: {}

profiles:
  default:
    models: [ep]
    active: true
"""
    cfg.write_text(ovms_yaml, encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    def ok_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="ok",
            summary="validation passed",
        )

    original_content = GENERATION_CONFIG

    with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
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
    # Live generation_config.json should be unchanged (no overrides, so not restored).
    live_genconfig = model_dir / "generation_config.json"
    assert live_genconfig.read_text(encoding="utf-8") == original_content


class TestPlainModel:
    """Plain (non-task) models register via model_config_list, no graph.pbtxt."""

    def test_activate_plain_model(self, tmp_path: Path,
                                  monkeypatch: pytest.MonkeyPatch):
        cfg = tmp_path / "ovms.yaml"
        loc = tmp_path / "local.yaml"
        store = tmp_path / "store"
        store.mkdir()

        # Plain model: directory present, no graph.pbtxt / generation_config.
        model_dir = store / "pp-doclayout-m"
        model_dir.mkdir(parents=True)
        (model_dir / "model.onnx").write_text("stub", encoding="utf-8")

        ovms_yaml = """\
runtime:
  rest_port: 8000
  log_level: INFO

repository:
  pp-doclayout-m:
    hf: pp-doclayout-m

models:
  doclayout:
    source: pp-doclayout-m
    device: NPU

profiles:
  layout:
    models: [doclayout]
    active: true
"""
        cfg.write_text(ovms_yaml, encoding="utf-8")
        loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        runner = CliRunner()

        def ok_smoke_check(decl):
            return CheckResult(name="smoke-load", status="ok", summary="ok")

        with patch("ovms_rig.probes.smoke_load.check", side_effect=ok_smoke_check):
            result = runner.invoke(
                main,
                ["--config", str(cfg), "--local", str(loc),
                 "--ovms-path", sys.executable, "activate", "layout"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0

        # No sibling graph created for a plain model.
        assert not list(model_dir.glob("graph.*.pbtxt"))

        # config.json registers it in model_config_list, not mediapipe.
        data = json.loads((store / "config.json").read_text(encoding="utf-8"))
        assert data["mediapipe_config_list"] == []
        assert data["model_config_list"] == [
            {
                "config": {
                    "name": "doclayout",
                    "base_path": str(model_dir.resolve()),
                    "target_device": "NPU",
                }
            },
        ]
