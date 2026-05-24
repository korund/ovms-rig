"""Unit tests for generation.merge()."""

from __future__ import annotations

import json

import pytest

from ovms_rig.stages.activation.generation import merge


class TestMerge:
    def test_merge_into_empty_dict(self):
        """Merging overrides into empty config returns just the overrides."""
        existing = "{}"
        overrides = {"temperature": 0.5, "top_p": 0.9}
        result = merge(existing, overrides)
        parsed = json.loads(result)
        assert parsed == overrides
        assert result.endswith("\n")

    def test_merge_with_overlapping_keys(self):
        """Overrides replace existing keys."""
        existing = '{"temperature": 1.0, "top_k": 40}'
        overrides = {"temperature": 0.5}
        result = merge(existing, overrides)
        parsed = json.loads(result)
        assert parsed["temperature"] == 0.5
        assert parsed["top_k"] == 40

    def test_merge_with_disjoint_keys(self):
        """Both original and override keys are present."""
        existing = '{"top_k": 40}'
        overrides = {"temperature": 0.5}
        result = merge(existing, overrides)
        parsed = json.loads(result)
        assert parsed == {"top_k": 40, "temperature": 0.5}

    def test_merge_empty_overrides(self):
        """Empty overrides dict returns unchanged content."""
        existing = '{"temperature": 0.5}'
        overrides = {}
        result = merge(existing, overrides)
        parsed = json.loads(result)
        assert parsed == {"temperature": 0.5}

    def test_merge_preserves_trailing_newline(self):
        """Output always has trailing newline."""
        existing = '{"a": 1}'
        overrides = {"b": 2}
        result = merge(existing, overrides)
        assert result.endswith("\n")

    def test_merge_2_space_indent(self):
        """Output uses 2-space indentation."""
        existing = '{"a": 1}'
        overrides = {"b": 2}
        result = merge(existing, overrides)
        # Result should be pretty-printed with 2-space indent.
        assert "  " in result

    def test_merge_invalid_json_raises(self):
        """Malformed JSON raises ValueError."""
        existing = "{ not valid json"
        overrides = {"temp": 0.5}
        with pytest.raises(ValueError, match="Invalid JSON"):
            merge(existing, overrides)

    def test_merge_multiple_types(self):
        """Overrides can be int, float, bool, str, list."""
        existing = '{"old": "value"}'
        overrides = {
            "temperature": 0.5,
            "max_new_tokens": 100,
            "do_sample": True,
            "name": "test",
            "indices": [1, 2, 3],
        }
        result = merge(existing, overrides)
        parsed = json.loads(result)
        assert parsed["temperature"] == 0.5
        assert parsed["max_new_tokens"] == 100
        assert parsed["do_sample"] is True
        assert parsed["name"] == "test"
        assert parsed["indices"] == [1, 2, 3]
        assert parsed["old"] == "value"
