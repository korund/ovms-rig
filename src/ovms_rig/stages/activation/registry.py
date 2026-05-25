"""Direct manipulation of config.json: rig owns the file.

config.json is rewritten as an exact projection of the active profile.
Any pre-existing content (model_config_list left over from another tool,
stray mediapipe entries, unknown keys) is discarded. Declarative contract:
what is declared in ovms.yaml, is what runs.

Entry structure:
  {
    "name": "<model_name>",
    "base_path": "<absolute_path_to_model_directory>",
    "graph_path": "graph.<model_name>.pbtxt"  (relative from base_path)
  }
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def render_mediapipe_entries(
    config_path: Path,
    desired_entries: dict[str, tuple[Path, str]],
) -> None:
    """Rewrite config.json as exact projection of desired_entries.

    desired_entries: dict mapping model_name -> (base_path, graph_path).
    If desired_entries is empty, mediapipe_config_list becomes [].

    Raises OSError if file I/O fails.
    """
    rendered = [
        {
            "name": model_name,
            "base_path": str(base_path),
            "graph_path": graph_path,
        }
        for model_name, (base_path, graph_path) in desired_entries.items()
    ]

    data = {
        "model_config_list": [],
        "mediapipe_config_list": rendered,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("[registry] rendered config.json: desired=%s", list(desired_entries.keys()))
