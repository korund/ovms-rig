"""Textual patcher for graph.pbtxt files.

Operates on the LLMCalculatorOptions block inside node_options. The block is
identified by its type URL line:

    [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
        ...
    }

Fields inside are simple key: value, pairs (OVMS generates one field per
line). The patcher:
- Replaces an existing field value in-place (preserving surrounding text).
- Appends a missing field before the closing brace of the block.

String values are written with double quotes; all other values are written
bare (booleans, numbers). The patcher does NOT parse the protobuf schema;
it treats every value it receives as already-formatted or delegates
formatting to _fmt().
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence


# Matches the opening of the LLMCalculatorOptions block (with optional
# spaces around '/').
_BLOCK_OPEN = re.compile(
    r"\[type\.googleapis\.com\s*/\s*mediapipe\.LLMCalculatorOptions\]:\s*\{"
)

# Matches a key: value, line inside the block.
# Group 1 = leading whitespace, Group 2 = key, Group 3 = ': ' separator,
# Group 4 = value (double-quoted, single-quoted, or bare), Group 5 = optional
# comma and trailing whitespace/comment.
#
# Single-quoted values are needed for fields like `plugin_config` whose value
# is a JSON document (contains double quotes and commas); wrapping in single
# quotes lets us keep the whole JSON as one regex-captured token without
# tripping the bare-value comma terminator.
_FIELD_RE = re.compile(
    r"^(\s*)(\w+)(\s*:\s*)(\"[^\"]*\"|'[^']*'|[^,\n#]+?)(,?\s*(?:#[^\n]*)?)$",
    re.MULTILINE,
)


class _Literal(str):
    """Marker: value already formatted as a pbtxt literal; emit as-is.

    Used for `plugin_config`, whose value is a JSON document wrapped in
    single quotes (so it survives the field regex intact).
    """


def _fmt(value: object) -> str:
    """Format a Python value as a pbtxt literal."""
    if isinstance(value, _Literal):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def _normalize_plugin_config_value(value: object) -> str:
    """Convert backslashes to forward-slashes for path-like plugin_config values.

    OVMS' pbtxt-to-JSON parser interprets backslashes inside the embedded JSON
    as escape sequences (`\\C` is not a valid JSON escape), so plugin_config
    paths must use forward slashes even on Windows. OpenVINO Core accepts
    forward-slash Windows paths.
    """
    if isinstance(value, Path):
        return str(value).replace("\\", "/")
    if isinstance(value, str) and "\\" in value:
        return value.replace("\\", "/")
    return str(value)


def format_plugin_config(values: dict[str, object]) -> _Literal:
    """Serialize a plugin_config dict to the pbtxt literal `'<json>'`.

    Wrapped in single quotes so the embedded JSON (which contains double
    quotes) survives the pbtxt field-line regex as a single token. All
    path-like values are forced to forward slashes before serialization --
    see _normalize_plugin_config_value.
    """
    normalized = {k: _normalize_plugin_config_value(v) for k, v in values.items()}
    # sort_keys for deterministic output (idempotent apply).
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return _Literal(f"'{payload}'")


def _find_block(text: str) -> tuple[int, int] | None:
    """Return (start, end) byte offsets of the LLMCalculatorOptions block content.

    start points to the character after '{', end points to the matching '}'.
    Returns None if the block is not found.
    """
    m = _BLOCK_OPEN.search(text)
    if m is None:
        return None

    # Walk forward from the opening brace to find the matching closing brace.
    depth = 0
    i = m.end() - 1  # position of '{'
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                # text[m.end():i] is the block body
                return m.end(), i
        i += 1
    return None  # malformed: no matching brace


def patch(text: str, fields: dict[str, object]) -> str:
    """Return a new pbtxt string with the given fields set in LLMCalculatorOptions.

    Existing fields are replaced in-place. Missing fields are appended before
    the closing brace of the block. Order of appended fields is stable
    (sorted by key for determinism).

    Raises ValueError if the LLMCalculatorOptions block is not found.
    """
    span = _find_block(text)
    if span is None:
        raise ValueError(
            "LLMCalculatorOptions block not found in pbtxt; "
            "is this a valid OVMS graph.pbtxt?"
        )
    block_start, block_end = span
    block_body = text[block_start:block_end]

    remaining = dict(fields)  # keys we still need to handle

    def _replace_field(m: re.Match) -> str:
        key = m.group(2)
        if key not in remaining:
            return m.group(0)
        new_val = _fmt(remaining.pop(key))
        sep = m.group(3)
        trailing = m.group(5)
        indent = m.group(1)
        return f"{indent}{key}{sep}{new_val}{trailing}"

    new_body = _FIELD_RE.sub(_replace_field, block_body)

    # Append any fields that were not found in the block.
    if remaining:
        # Detect indentation from the first existing field line, fallback to 8
        # spaces (matching typical OVMS pbtxt style).
        indent = _detect_indent(new_body)
        additions = "".join(
            f"{indent}{k}: {_fmt(v)},\n"
            for k, v in sorted(remaining.items())
        )
        # Insert before the closing brace.
        new_body = new_body.rstrip("\n") + "\n" + additions

    return text[:block_start] + new_body + text[block_end:]


def _detect_indent(block_body: str) -> str:
    """Return the indentation string used by the first field line in the block."""
    for line in block_body.splitlines():
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#") and ":" in stripped:
            return line[: len(line) - len(stripped)]
    return "        "  # 8 spaces fallback


def patch_file(path: Path, fields: dict[str, object]) -> str:
    """Read path, apply patch(), return new content. Does not write."""
    text = path.read_text(encoding="utf-8")
    return patch(text, fields)


def collect_pbtxt_fields(
    entry_graph,  # config.schema.Graph instance
    draft_models_path: str | None,
    cache_dir: Path | None = None,
) -> dict[str, object]:
    """Build the dict of fields to patch into LLMCalculatorOptions.

    Covers device, draft_device, draft_models_path, and plugin_config.

    cache_dir bridge: ovms 2026.1 LLM continuous-batching pipeline does not
    forward the global `--cache_dir` flag to the device plugin (the LLM
    servable initializer skips `set_property(ov::cache_dir)`). We work around
    by injecting CACHE_DIR via plugin_config, which the LLMCalculator does
    propagate. If the user has already set plugin_config.CACHE_DIR explicitly,
    their value wins.
    """
    fields: dict[str, object] = {}
    if entry_graph.device is not None:
        fields["device"] = entry_graph.device
    if entry_graph.draft_device is not None:
        fields["draft_device"] = entry_graph.draft_device
    if draft_models_path is not None:
        fields["draft_models_path"] = draft_models_path

    plugin_config: dict[str, object] = dict(entry_graph.plugin_config or {})
    if cache_dir is not None and "CACHE_DIR" not in plugin_config:
        plugin_config["CACHE_DIR"] = cache_dir
    if plugin_config:
        fields["plugin_config"] = format_plugin_config(plugin_config)
    return fields
