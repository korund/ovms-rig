"""Tests for profile activation/deactivation stage.

Tests verify:
- activate <unknown> fails with rc=1
- activate <name> when already active is idempotent
- activate <other> when <first> active switches active status
- deactivate when active sets no active profile
- deactivate when none active is no-op
- --backup writes ovms.yaml.bak (fixed name, overwrites existing); no backup by default
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

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


def test_no_backup_by_default(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --backup, activate must not create any ovms.yaml.bak* files."""
    monkeypatch.chdir(rig["tmp"])

    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0

    leftovers = list(rig["config"].parent.glob("ovms.yaml.bak*"))
    assert leftovers == [], f"Expected no backups, found: {leftovers}"


def test_backup_flag_writes_fixed_name(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """With --backup, write ovms.yaml.bak (no timestamp) with original content."""
    monkeypatch.chdir(rig["tmp"])

    result = _invoke(rig, "activate", "default", "--backup")
    assert result.exit_code == 0

    backup_path = rig["config"].parent / "ovms.yaml.bak"
    assert backup_path.exists()
    # No timestamped variants alongside it.
    timestamped = list(rig["config"].parent.glob("ovms.yaml.bak.*"))
    assert timestamped == [], f"Expected no timestamped backups, found: {timestamped}"

    backup_data = yaml.safe_load(backup_path.read_text(encoding="utf-8"))
    assert "default" in backup_data.get("profiles", {})
    assert "bench" in backup_data.get("profiles", {})


def test_backup_flag_overwrites_existing_backup(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated --backup runs overwrite ovms.yaml.bak; no accumulation."""
    monkeypatch.chdir(rig["tmp"])

    backup_path = rig["config"].parent / "ovms.yaml.bak"
    backup_path.write_text("stale-backup-content\n", encoding="utf-8")

    result = _invoke(rig, "activate", "default", "--backup")
    assert result.exit_code == 0

    assert backup_path.exists()
    assert backup_path.read_text(encoding="utf-8") != "stale-backup-content\n"
    # Still only one .bak file.
    assert list(rig["config"].parent.glob("ovms.yaml.bak*")) == [backup_path]


def test_activate_atomic_write_preserves_original_on_write_failure(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.replace failure leaves original ovms.yaml unchanged."""
    monkeypatch.chdir(rig["tmp"])

    # Record original content before any activation attempt.
    original_content = rig["config"].read_text(encoding="utf-8")

    # Mock os.replace to raise OSError.
    import os
    original_replace = os.replace
    call_count = [0]

    def mock_replace(src, dst):
        call_count[0] += 1
        # Fail on the config_path replace (not on other paths).
        if str(dst) == str(rig["config"]):
            raise OSError("Simulated write failure")
        return original_replace(src, dst)

    monkeypatch.setattr("os.replace", mock_replace)

    result = _invoke(rig, "activate", "default")
    # Should fail due to write failure.
    assert result.exit_code == 1

    # Original content must be completely unchanged.
    current_content = rig["config"].read_text(encoding="utf-8")
    assert current_content == original_content, "Original ovms.yaml should be unchanged after write failure"


def test_activate_rolls_back_yaml_when_apply_fails(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """When apply.run fails, ovms.yaml is rolled back to original state."""
    monkeypatch.chdir(rig["tmp"])

    # Record original state: default should be active initially.
    original_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    assert original_data["profiles"]["default"]["active"] is True
    assert original_data["profiles"]["bench"]["active"] is False

    # Mock apply.run to fail.
    monkeypatch.setattr("ovms_rig.stages.activation.apply.run", lambda ctx: 1)

    # Attempt to activate bench (which will fail when apply runs).
    result = _invoke(rig, "activate", "bench")
    assert result.exit_code == 1

    # ovms.yaml should be rolled back to original state (default active, bench inactive).
    rolled_back_data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    assert rolled_back_data["profiles"]["default"]["active"] is True, "Should rollback to original state"
    assert rolled_back_data["profiles"]["bench"]["active"] is False, "Should rollback to original state"

    # No disk backup expected (rollback uses in-memory snapshot).
    assert list(rig["config"].parent.glob("ovms.yaml.bak*")) == []

    # Log should mention rollback.
    assert "rolled back" in result.output.lower() or "apply failed" in result.output.lower()


def test_activate_leaves_files_on_smoke_load_failure(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke-load failure exits non-zero but leaves rendered files in place.

    Rationale: rolling ovms.yaml back to its pre-activate snapshot and reapplying
    derived files does not restore a known-good state -- the snapshot captures
    whatever the user hand-edited, which is typically what caused the failure.
    Honest behavior: report the failure, leave the rejected config on disk for
    the user to inspect and fix.
    """
    monkeypatch.chdir(rig["tmp"])

    from ovms_rig.report import CheckResult

    def fail_smoke_check(decl):
        return CheckResult(
            name="smoke-load",
            status="error",
            summary="test failure",
            details={"fail_markers": ["test marker"]},
        )

    runner = CliRunner()
    with patch("ovms_rig.probes.smoke_load.check", side_effect=fail_smoke_check):
        result = runner.invoke(
            main,
            [
                "--config", str(rig["config"]),
                "--local", str(rig["local"]),
                "--ovms-path", sys.executable,
                "activate",
                "bench",
            ],
            catch_exceptions=False,
        )

    assert result.exit_code == 1

    # ovms.yaml reflects the requested flip -- not rolled back.
    data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    assert data["profiles"]["default"]["active"] is False
    assert data["profiles"]["bench"]["active"] is True

    # config.json was rendered (and left as-is) for the requested profile.
    config_json = rig["store"] / "config.json"
    assert config_json.exists()
    config_data = json.loads(config_json.read_text(encoding="utf-8"))
    assert len(config_data.get("mediapipe_config_list", [])) == 1


def test_bare_activate_reapplies_without_touching_yaml(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """`activate` without a profile name re-renders derived files but leaves yaml unchanged."""
    monkeypatch.chdir(rig["tmp"])

    # default is active in fixture. Capture yaml verbatim.
    yaml_before = rig["config"].read_text(encoding="utf-8")

    # Render config.json once via normal activate, then delete it to prove reapply re-creates it.
    result = _invoke(rig, "activate", "default")
    assert result.exit_code == 0
    config_json = rig["store"] / "config.json"
    assert config_json.exists()
    config_json.unlink()

    # Yaml may have been re-serialized by activate; refresh baseline.
    yaml_before = rig["config"].read_text(encoding="utf-8")

    # Bare activate: no profile name.
    result = _invoke(rig, "activate")
    assert result.exit_code == 0

    # Yaml byte-identical -- bare activate must not touch it.
    yaml_after = rig["config"].read_text(encoding="utf-8")
    assert yaml_after == yaml_before, "bare activate must not modify ovms.yaml"

    # Derived files were re-rendered.
    assert config_json.exists()
    config_data = json.loads(config_json.read_text(encoding="utf-8"))
    assert len(config_data.get("mediapipe_config_list", [])) == 1


def test_bare_activate_with_no_active_profile_renders_empty(rig: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bare `activate` when no profile is active renders an empty config (== deactivate)."""
    monkeypatch.chdir(rig["tmp"])

    # Flip all profiles inactive by hand.
    data = yaml.safe_load(rig["config"].read_text(encoding="utf-8"))
    for p in data["profiles"].values():
        p["active"] = False
    rig["config"].write_text(yaml.dump(data), encoding="utf-8")
    yaml_before = rig["config"].read_text(encoding="utf-8")

    result = _invoke(rig, "activate")
    assert result.exit_code == 0

    assert rig["config"].read_text(encoding="utf-8") == yaml_before
    config_json = rig["store"] / "config.json"
    assert config_json.exists()
    config_data = json.loads(config_json.read_text(encoding="utf-8"))
    assert config_data.get("mediapipe_config_list") == []
