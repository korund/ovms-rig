"""Unit tests for apply/paths.py: relative POSIX path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from ovms_rig.stages.activation.paths import relative_posix


class TestSameOrg:
    def test_sibling_repos(self, tmp_path: Path):
        target = tmp_path / "OpenVINO" / "Qwen3-14B-int8-ov"
        draft = tmp_path / "OpenVINO" / "Qwen3-0.6B-int8-ov"
        target.mkdir(parents=True)
        draft.mkdir(parents=True)
        rel = relative_posix(target, draft)
        assert rel == "../Qwen3-0.6B-int8-ov"

    def test_sibling_repos_forward_slashes(self, tmp_path: Path):
        target = tmp_path / "Org" / "TargetModel"
        draft = tmp_path / "Org" / "DraftModel"
        target.mkdir(parents=True)
        draft.mkdir(parents=True)
        rel = relative_posix(target, draft)
        assert "/" in rel
        assert "\\" not in rel


class TestCrossOrg:
    def test_different_orgs(self, tmp_path: Path):
        target = tmp_path / "OrgA" / "main-model"
        draft = tmp_path / "OrgB" / "draft-model"
        target.mkdir(parents=True)
        draft.mkdir(parents=True)
        rel = relative_posix(target, draft)
        assert rel == "../../OrgB/draft-model"

    def test_different_orgs_no_backslash(self, tmp_path: Path):
        target = tmp_path / "Foo" / "A"
        draft = tmp_path / "Bar" / "B"
        target.mkdir(parents=True)
        draft.mkdir(parents=True)
        rel = relative_posix(target, draft)
        assert "\\" not in rel


class TestEdgeCases:
    def test_same_directory(self, tmp_path: Path):
        d = tmp_path / "Org" / "Repo"
        d.mkdir(parents=True)
        rel = relative_posix(d, d)
        # Relative path from a directory to itself is "."
        assert rel in (".", "")

    def test_three_levels(self, tmp_path: Path):
        target = tmp_path / "A" / "B" / "target"
        draft = tmp_path / "X" / "Y" / "draft"
        target.mkdir(parents=True)
        draft.mkdir(parents=True)
        rel = relative_posix(target, draft)
        # Navigating 3 levels up then into X/Y/draft
        assert rel.startswith("../")
        assert "draft" in rel
        assert "\\" not in rel
