"""Unit tests for pbtxt.py: field patching round-trips."""

from __future__ import annotations

import pytest

from pathlib import Path

from ovms_rig.stages.activation.pbtxt import (
    collect_pbtxt_fields,
    format_plugin_config,
    patch,
)

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


class TestFormatPluginConfig:
    """Serialization of plugin_config dict to the pbtxt literal."""

    def test_simple_dict_serialized(self):
        lit = format_plugin_config({"PERFORMANCE_HINT": "LATENCY"})
        # Single-quoted JSON document.
        assert lit.startswith("'") and lit.endswith("'")
        assert '"PERFORMANCE_HINT":"LATENCY"' in lit

    def test_backslashes_converted_to_forward_slashes(self):
        # Windows-style path must become POSIX before json.dumps so the pbtxt
        # parser does not mis-escape the embedded JSON.
        lit = format_plugin_config({"CACHE_DIR": r"C:\Forge\cache"})
        assert "C:/Forge/cache" in lit
        assert "\\" not in lit

    def test_path_value_normalized(self):
        lit = format_plugin_config({"CACHE_DIR": Path("C:/Forge/cache")})
        assert "C:/Forge/cache" in lit

    def test_deterministic_key_order(self):
        a = format_plugin_config({"B": "2", "A": "1"})
        b = format_plugin_config({"A": "1", "B": "2"})
        assert a == b


class TestPatchPluginConfig:
    """Patching plugin_config into LLMCalculatorOptions."""

    def test_plugin_config_appended(self):
        lit = format_plugin_config({"CACHE_DIR": "C:/cache"})
        result = patch(_BASE, {"plugin_config": lit})
        assert "plugin_config: '" in result
        assert '"CACHE_DIR":"C:/cache"' in result

    def test_plugin_config_replaced(self):
        # First apply, then re-apply with a different value.
        once = patch(_BASE, {"plugin_config": format_plugin_config({"CACHE_DIR": "C:/a"})})
        twice = patch(once, {"plugin_config": format_plugin_config({"CACHE_DIR": "C:/b"})})
        assert '"CACHE_DIR":"C:/b"' in twice
        assert '"CACHE_DIR":"C:/a"' not in twice
        # Exactly one plugin_config line in the file.
        assert twice.count("plugin_config:") == 1

    def test_plugin_config_idempotent(self):
        fields = {"plugin_config": format_plugin_config({"CACHE_DIR": "C:/cache"})}
        once = patch(_BASE, fields)
        twice = patch(once, fields)
        assert once == twice


class _StubGraph:
    """Minimal stand-in for config.schema.Graph (LLM-only fields)."""

    def __init__(
        self,
        draft_device=None,
        max_num_seqs=None,
        enable_prefix_caching=None,
        cache_size=None,
        dynamic_split_fuse=None,
    ):
        self.draft_device = draft_device
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_caching = enable_prefix_caching
        self.cache_size = cache_size
        self.dynamic_split_fuse = dynamic_split_fuse


class _StubEntry:
    """Minimal stand-in for config.schema.ModelEntry used by bridge tests.

    device and plugin_config are shared entry-level knobs; the remaining
    LLMCalculatorOptions fields live on the nested graph.
    """

    def __init__(
        self,
        device="GPU",
        plugin_config=None,
        draft_device=None,
        max_num_seqs=None,
        enable_prefix_caching=None,
        cache_size=None,
        dynamic_split_fuse=None,
    ):
        self.device = device
        self.plugin_config = plugin_config
        self.graph = _StubGraph(
            draft_device=draft_device,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            cache_size=cache_size,
            dynamic_split_fuse=dynamic_split_fuse,
        )


class TestLLMCalculatorOptionsFields:
    """All declared LLMCalculatorOptions fields reach the patch dict."""

    def test_all_declared_fields_collected(self):
        entry = _StubEntry(
            max_num_seqs=128,
            enable_prefix_caching=True,
            cache_size=0,
            dynamic_split_fuse=True,
        )
        fields = collect_pbtxt_fields(entry, draft_models_path=None, cache_dir=None)
        assert fields["max_num_seqs"] == 128
        assert fields["enable_prefix_caching"] is True
        assert fields["cache_size"] == 0
        assert fields["dynamic_split_fuse"] is True

    def test_unset_fields_skipped(self):
        entry = _StubEntry(max_num_seqs=64)
        fields = collect_pbtxt_fields(entry, draft_models_path=None, cache_dir=None)
        assert fields["max_num_seqs"] == 64
        assert "enable_prefix_caching" not in fields
        assert "cache_size" not in fields
        assert "dynamic_split_fuse" not in fields


class TestCacheDirBridge:
    """local.runtime.cache_dir bridged into plugin_config.CACHE_DIR."""

    def test_cache_dir_injected_when_plugin_config_absent(self):
        entry = _StubEntry(plugin_config=None)
        fields = collect_pbtxt_fields(
            entry, draft_models_path=None, cache_dir=Path("C:/ov/cache"),
        )
        assert "plugin_config" in fields
        assert '"CACHE_DIR":"C:/ov/cache"' in str(fields["plugin_config"])

    def test_cache_dir_injected_when_key_missing(self):
        entry = _StubEntry(plugin_config={"PERFORMANCE_HINT": "LATENCY"})
        fields = collect_pbtxt_fields(
            entry, draft_models_path=None, cache_dir=Path("C:/ov/cache"),
        )
        rendered = str(fields["plugin_config"])
        assert '"CACHE_DIR":"C:/ov/cache"' in rendered
        assert '"PERFORMANCE_HINT":"LATENCY"' in rendered

    def test_explicit_cache_dir_wins(self):
        entry = _StubEntry(plugin_config={"CACHE_DIR": "D:/user/choice"})
        fields = collect_pbtxt_fields(
            entry, draft_models_path=None, cache_dir=Path("C:/ignored"),
        )
        rendered = str(fields["plugin_config"])
        assert '"CACHE_DIR":"D:/user/choice"' in rendered
        assert "ignored" not in rendered

    def test_no_plugin_config_when_neither_set(self):
        entry = _StubEntry(plugin_config=None)
        fields = collect_pbtxt_fields(
            entry, draft_models_path=None, cache_dir=None,
        )
        assert "plugin_config" not in fields
