from __future__ import annotations

import pytest

from ovms_rig.probes.registry import PROBES, PRESETS, Preset, run


class TestPresets:
    def test_presets_cover_known_keys(self):
        for preset in Preset:
            probe_keys = PRESETS[preset]
            for key in probe_keys:
                assert key in PROBES, f"preset {preset.value} references unknown probe key: {key}"

    def test_blocking_subset_of_diagnostic(self):
        diagnostic_keys = set(PRESETS[Preset.DIAGNOSTIC])
        blocking_keys = set(PRESETS[Preset.BLOCKING])
        assert blocking_keys.issubset(diagnostic_keys), \
            f"BLOCKING is not a subset of DIAGNOSTIC"

    def test_blocking_preset_contents(self):
        expected = {"declaration", "ovms_binary", "models", "port"}
        actual = set(PRESETS[Preset.BLOCKING])
        assert actual == expected, \
            f"BLOCKING preset mismatch. Expected {expected}, got {actual}"


class TestRun:
    def test_run_returns_report_with_matching_entries(self, tmp_path, monkeypatch):
        config_file = tmp_path / "ovms.yaml"
        config_file.write_text(
            "runtime:\n"
            "  rest_port: 8001\n"
            "repository: {}\n"
            "models: {}\n"
            "profiles: {}\n",
            encoding="utf-8"
        )
        local_file = tmp_path / "local.yaml"

        ctx = {
            "config_path": str(config_file),
            "local_path": str(local_file),
        }

        report = run(ctx, Preset.BLOCKING)

        assert len(report.entries) == 4, \
            f"BLOCKING preset should have 4 entries, got {len(report.entries)}"

        probe_names = [probe.name for probe, _ in report.entries]
        expected_names = [
            "declaration",
            "ovms binary",
            "models",
            "rest port",
        ]
        assert probe_names == expected_names, \
            f"probe order mismatch. Expected {expected_names}, got {probe_names}"
