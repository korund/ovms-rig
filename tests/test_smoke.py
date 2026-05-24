"""Smoke tests that prove the package imports and the CLI is wired up."""

from __future__ import annotations

from click.testing import CliRunner

from ovms_rig import __version__
from ovms_rig.cli import main


def test_package_has_version():
    assert isinstance(__version__, str) and __version__


def test_cli_help_exits_cleanly():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Declarative loader" in result.output


def test_cli_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    for cmd in ("status", "fetch", "activate", "deactivate", "start"):
        assert cmd in result.output
