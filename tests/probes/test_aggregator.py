from __future__ import annotations

import logging

import pytest

from ovms_rig.probes.registry import Probe
from ovms_rig.probes.aggregator import Report
from ovms_rig.report import CheckResult


class TestReportHasErrors:
    def test_has_errors_true_on_any_error(self):
        entries = [
            (
                Probe("p1", lambda ctx: None),
                CheckResult(name="p1", status="ok", summary="ok"),
            ),
            (
                Probe("p2", lambda ctx: None),
                CheckResult(name="p2", status="error", summary="error"),
            ),
        ]
        report = Report(entries)
        assert report.has_errors() is True

    def test_has_errors_false_when_all_ok(self):
        entries = [
            (
                Probe("p1", lambda ctx: None),
                CheckResult(name="p1", status="ok", summary="ok"),
            ),
            (
                Probe("p2", lambda ctx: None),
                CheckResult(name="p2", status="ok", summary="ok"),
            ),
        ]
        report = Report(entries)
        assert report.has_errors() is False

    def test_has_errors_false_when_all_warn(self):
        entries = [
            (
                Probe("p1", lambda ctx: None),
                CheckResult(name="p1", status="warn", summary="warn"),
            ),
        ]
        report = Report(entries)
        assert report.has_errors() is False

    def test_has_errors_false_when_mixed_ok_warn(self):
        entries = [
            (
                Probe("p1", lambda ctx: None),
                CheckResult(name="p1", status="ok", summary="ok"),
            ),
            (
                Probe("p2", lambda ctx: None),
                CheckResult(name="p2", status="warn", summary="warn"),
            ),
        ]
        report = Report(entries)
        assert report.has_errors() is False


class TestReportPrint:
    def test_print_logs_each_entry(self, caplog):
        entries = [
            (
                Probe("p1", lambda ctx: None),
                CheckResult(
                    name="p1",
                    status="ok",
                    summary="all is well",
                ),
            ),
            (
                Probe("p2", lambda ctx: None),
                CheckResult(
                    name="p2",
                    status="error",
                    summary="something failed",
                    hint="try this",
                    details={"key": "value"},
                ),
            ),
        ]
        report = Report(entries)

        with caplog.at_level(logging.DEBUG):
            report.print()

        assert "[OK] p1 -- all is well" in caplog.text
        assert "[ERROR] p2 -- something failed" in caplog.text
        assert "hint: try this" in caplog.text
        assert "details:" in caplog.text
