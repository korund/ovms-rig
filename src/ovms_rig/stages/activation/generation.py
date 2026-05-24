"""Top-level merge of user overrides into generation_config.json."""

from __future__ import annotations

import json


def merge(existing_json_text: str, overrides: dict) -> str:
    try:
        config = json.loads(existing_json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    config.update(overrides)
    return json.dumps(config, indent=2, sort_keys=False) + "\n"
