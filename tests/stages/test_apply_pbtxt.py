"""Unit tests for pbtxt.py: field patching round-trips."""

from __future__ import annotations

import pytest

from ovms_rig.stages.apply.pbtxt import patch

# Minimal pbtxt that mirrors real OVMS output (without draft fields).
_BASE = """\
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
        }
    }
    }
"""

# Same but with draft fields already present.
_BASE_WITH_DRAFT = """\
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "CPU",
            models_path: "./",
            draft_models_path: "old-draft",
            draft_device: "CPU",
        }
    }
"""


class TestPatchReplaceExisting:
    """Replace a field that already exists in the block."""

    def test_device_replaced(self):
        result = patch(_BASE, {"device": "GPU"})
        assert 'device: "GPU"' in result
        # Old value gone
        assert 'device: "CPU"' not in result

    def test_other_fields_preserved(self):
        result = patch(_BASE, {"device": "GPU"})
        assert "max_num_seqs" in result
        assert "models_path" in result
        assert "enable_prefix_caching" in result

    def test_multiple_fields_replaced(self):
        result = patch(_BASE_WITH_DRAFT, {"device": "GPU", "draft_device": "NPU"})
        assert 'device: "GPU"' in result
        assert 'draft_device: "NPU"' in result

    def test_draft_models_path_replaced(self):
        result = patch(_BASE_WITH_DRAFT, {"draft_models_path": "../new-draft"})
        assert '"../new-draft"' in result
        assert '"old-draft"' not in result


class TestPatchAppendMissing:
    """Append a field that does not yet exist in the block."""

    def test_draft_models_path_appended(self):
        result = patch(_BASE, {"draft_models_path": "../some-draft"})
        assert '"../some-draft"' in result

    def test_draft_device_appended(self):
        result = patch(_BASE, {"draft_device": "CPU"})
        assert 'draft_device: "CPU"' in result

    def test_appended_field_inside_block(self):
        # The appended field must appear before the closing brace of the block.
        result = patch(_BASE, {"draft_models_path": "../d"})
        block_start = result.index("LLMCalculatorOptions")
        block_end = result.index("}", block_start)
        field_pos = result.index('"../d"')
        assert block_start < field_pos < block_end


class TestPatchIdempotent:
    """Applying the same patch twice yields the same result."""

    def test_replace_idempotent(self):
        fields = {"device": "GPU"}
        once = patch(_BASE, fields)
        twice = patch(once, fields)
        assert once == twice

    def test_append_then_replace_idempotent(self):
        fields = {"draft_models_path": "../draft", "draft_device": "CPU"}
        once = patch(_BASE, fields)
        twice = patch(once, fields)
        assert once == twice


class TestPatchMissingBlock:
    def test_raises_on_missing_block(self):
        with pytest.raises(ValueError, match="LLMCalculatorOptions"):
            patch("node_options: {}", {"device": "GPU"})


class TestPatchStringFormatting:
    def test_string_value_quoted(self):
        result = patch(_BASE, {"device": "GPU"})
        assert 'device: "GPU"' in result

    def test_string_appended_quoted(self):
        result = patch(_BASE, {"draft_models_path": "../x"})
        assert 'draft_models_path: "../x"' in result
