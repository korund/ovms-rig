"""Tests for profile activation/deactivation stage.

Tests verify:
- activate <unknown> fails with rc=1
- activate <name> when already active is idempotent
- activate <other> when <first> active switches active status
- deactivate when active sets no active profile
- deactivate when none active is no-op
- backup created as .bak.<timestamp> after activate
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
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

models:
  ep:
    source: main
    graph:
      device: GPU

profiles:
  default:
    models: [ep]
    active: true
  bench:
    models: [ep]
    active: false
"""

LOCAL_YAML = """\
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""

GRAPH_PBTXT = """\
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs: 256,
            device: "CPU",
            models_path: "./",
        }
    }
"""


@pytest.fixture
def rig(tmp_path: Path) -> dict:
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()

    # Create model directory with graph.pbtxt.
    model_dir = store / "OpenVINO" / "main-int8-ov"
    model_dir.mkdir(parents=True)
    (model_dir / "graph.pbtxt").write_text(GRAPH_PBTXT, encoding="utf-8")

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_activate_unknown_profile_fails(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate <unknown> -> rc=1, yaml unchanged, config unchanged."""
    monkeypatch.chdir(rig["tmp"])

    # Record initial state.
    cfg_before = rig["config"].read_text(encoding="utf-8")

    result = _invoke(rig, "activate", "nonexistent")
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "nonexistent" in result.output

    # Config file must be unchanged.
    cfg_after = rig["config"].read_text(encoding="utf-8")
    assert cfg_before == cfg_after, "Config should not change on error"

    # No config.json should be created (apply was not called).
    config_json = rig["store"] / "config.json"
    assert not config_json.exists(), "config.json should not be created on activation failure"


def test_activate_already_active_is_idempotent(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate <name> when already active -> idempotent rc=0."""
    monkeypatch.chdir(rig["tmp"])

    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0

    # config.json should be created.
    config_json = rig["store"] / "config.json"
    assert config_json.exists()

    # Activate again (should be idempotent).
    result2 = _invoke(rig, "activate", "default")
    assert result2.exit_code == 0

    # config.json should still exist and be functional.
    assert config_json.exists()


def test_activate_switches_active_status(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Activate <other> when <first> active -> first inactive, other active in yaml."""
    monkeypatch.chdir(rig["tmp"])

    # Initially default is active.
    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0

    cfg_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    assert cfg_data["profiles"]["default"]["active"] is True
    assert cfg_data["profiles"]["bench"]["active"] is False

    # Now activate bench.
    result = _invoke(rig, "activate", "bench")
    assert result.exit_code == 0

    cfg_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    assert cfg_data["profiles"]["default"]["active"] is False
    assert cfg_data["profiles"]["bench"]["active"] is True


def test_deactivate_clears_active_profile(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deactivate when active -> no active profile, config empty."""
    monkeypatch.chdir(rig["tmp"])

    # Activate first.
    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0

    # Now deactivate.
    result = _invoke(rig, "deactivate")
    assert result.exit_code == 0

    # Check yaml: no profile should be active.
    cfg_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    for profile in cfg_data["profiles"].values():
        assert profile["active"] is False, "All profiles should be inactive after deactivate"

    # config.json should have empty mediapipe_config_list.
    import json
    config_json = rig["store"] / "config.json"
    assert config_json.exists()
    config_data = json.loads(config_json.read_text(encoding="utf-8"))
    assert config_data.get("mediapipe_config_list") == []


def test_deactivate_when_none_active_is_noop(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Deactivate when none active -> no-op rc=0, config empty."""
    monkeypatch.chdir(rig["tmp"])

    # Start with all profiles inactive.
    cfg_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    cfg_data["profiles"]["default"]["active"] = False
    cfg_data["profiles"]["bench"]["active"] = False
    rig["config"].write_text(yaml.dump(cfg_data), encoding="utf-8")

    # Deactivate (should be no-op).
    result = _invoke(rig, "deactivate")
    assert result.exit_code == 0

    # Still no active profiles.
    cfg_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    for profile in cfg_data["profiles"].values():
        assert profile["active"] is False

    # config.json should have empty list.
    import json
    config_json = rig["store"] / "config.json"
    config_data = json.loads(config_json.read_text(encoding="utf-8"))
    assert config_data.get("mediapipe_config_list") == []


def test_backup_created_on_activate(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backup ovms.yaml created as .bak.<timestamp> after activate."""
    monkeypatch.chdir(rig["tmp"])

    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0

    # Look for backup files.
    backups = list(rig["config"].parent.glob("ovms.yaml.bak.*"))
    assert len(backups) >= 1, f"Expected at least 1 backup, found: {backups}"

    # Backup should contain original content (both profiles exist).
    backup_path = backups[0]
    backup_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert "default" in backup_data.get("profiles", {})
    assert "bench" in backup_data.get("profiles", {})
