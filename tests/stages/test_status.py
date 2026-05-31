"""End-to-end smoke test for the `status` command via the click CLI."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from ovms_rig.cli import main

OVMS_YAML = """
runtime:
  rest_port: {port}
  log_level: DEBUG

repository:
  main:
    hf: org/main-int8-ov
    task: text_generation
  draft:
    hf: org/draft-int8-ov
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

LOCAL_YAML = """
runtime:
  ovms_path: null
models:
  repository_path: {store}
"""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def rig(tmp_path: Path) -> dict:
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    store = tmp_path / "store"
    store.mkdir()
    cfg.write_text(OVMS_YAML.format(port=_free_port()), encoding="utf-8")
    loc.write_text(LOCAL_YAML.format(store=store.as_posix()), encoding="utf-8")
    return {"config": cfg, "local": loc, "store": store}


def _invoke(rig: dict, ovms_path: str) -> "object":
    runner = CliRunner()
    return runner.invoke(
        main,
        [
            "--config", str(rig["config"]),
            "--local", str(rig["local"]),
            "--ovms-path", ovms_path,
            "status",
        ],
        catch_exceptions=False,
    )


def test_status_succeeds_on_fresh_rig(rig: dict) -> None:
    result = _invoke(rig, ovms_path=sys.executable)
    assert result.exit_code == 0
    assert "declaration" in result.output
    assert "ovms binary" in result.output
    assert "model store destination" in result.output
    assert "declared models on disk" in result.output
    assert "models (endpoints)" in result.output
    assert "profiles" in result.output
    assert "rest port" in result.output
    assert "live ovms config" in result.output


def test_status_fails_when_ovms_binary_missing(rig: dict, tmp_path: Path) -> None:
    result = _invoke(rig, ovms_path=str(tmp_path / "does-not-exist"))
    assert result.exit_code == 1
    assert "ERROR" in result.output


def test_status_reports_missing_models_as_ok_with_hint(rig: dict) -> None:
    result = _invoke(rig, ovms_path=sys.executable)
    assert result.exit_code == 0
    assert "0/2 present" in result.output
    assert "fetch" in result.output


def test_status_fails_when_repository_path_unset(tmp_path: Path) -> None:
    # repository_path is required by schema, so omitting it surfaces at
    # config-load time rather than as a downstream warning.
    cfg = tmp_path / "ovms.yaml"
    loc = tmp_path / "local.yaml"
    cfg.write_text(OVMS_YAML.format(port=_free_port()), encoding="utf-8")
    loc.write_text("runtime:\n  ovms_path: null\nmodels: {}\n",
                   encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "--config", str(cfg),
            "--local", str(loc),
            "--ovms-path", sys.executable,
            "status",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 1
    assert "config load failed" in result.output
    assert "repository_path" in result.output
