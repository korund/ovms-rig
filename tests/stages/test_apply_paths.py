"""Unit tests for apply/paths.py: relative POSIX path resolver and model dir resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from ovms_rig.stages.activation.paths import relative_posix, resolve_model_dir


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


class TestResolveModelDir:
    """Tests for resolve_model_dir: resolves hf or dir source to absolute path."""

    def test_hf_source_returns_store_slash_hf_id(self, tmp_path: Path) -> None:
        # HF source: store / hf_id
        store = tmp_path / "models"
        store.mkdir()
        hf_id = "OpenVINO/Qwen3-14B"
        result = resolve_model_dir(store, hf_id=hf_id, local_dir=None)
        expected = store / hf_id
        assert result == expected

    def test_dir_source_relative_resolves_to_store_slash_dir(self, tmp_path: Path) -> None:
        # dir source with relative path: resolved against store, then normalized.
        store = tmp_path / "models"
        store.mkdir()
        local_dir = "local/qwen"
        result = resolve_model_dir(store, hf_id=None, local_dir=local_dir)
        expected = (store / local_dir).resolve()
        assert result == expected

    def test_dir_source_absolute_resolves_to_itself(self, tmp_path: Path) -> None:
        # dir source with absolute path: used as-is (after resolve() for normalization).
        abs_dir = tmp_path / "opt" / "models" / "qwen"
        abs_dir.mkdir(parents=True)
        store = tmp_path / "models"
        store.mkdir()
        result = resolve_model_dir(store, hf_id=None, local_dir=str(abs_dir))
        assert result == abs_dir.resolve()

    def test_dir_source_relative_nested_path(self, tmp_path: Path) -> None:
        # dir source with nested relative path.
        store = tmp_path / "store"
        store.mkdir()
        local_dir = "nested/path/to/model"
        result = resolve_model_dir(store, hf_id=None, local_dir=local_dir)
        expected = (store / local_dir).resolve()
        assert result == expected
        assert "nested" in str(result)
        assert "path" in str(result)
        assert "to" in str(result)
        assert "model" in str(result)

    def test_neither_hf_nor_dir_raises_error(self, tmp_path: Path) -> None:
        # Both hf_id and local_dir are None: error.
        store = tmp_path / "models"
        with pytest.raises(ValueError, match="both.*are None"):
            resolve_model_dir(store, hf_id=None, local_dir=None)
