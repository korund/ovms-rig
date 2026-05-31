"""Direct manipulation of config.json: rig owns the file.

config.json is rewritten as an exact projection of the active profile.
Any pre-existing content (model_config_list left over from another tool,
stray mediapipe entries, unknown keys) is discarded. Declarative contract:
what is declared in ovms.yaml, is what runs.

Two entry kinds, by model type:
  mediapipe (task-based LLM):
    {"name": "<model_name>",
     "base_path": "<abs model dir>",
     "graph_path": "graph.<model_name>.pbtxt"}
  plain (model_config_list):
    {"config": {"name": "<model_name>",
                "base_path": "<abs model dir>",
                "target_device": "<device>"}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def render_config(
    config_path: Path,
    mediapipe_entries: dict[str, tuple[Path, str]],
    model_entries: dict[str, tuple[Path, str, dict[str, str] | None, dict[str, object] | None]],
) -> None:
    """Rewrite config.json as exact projection of the desired entries.

    mediapipe_entries: model_name -> (base_path, graph_path).
    model_entries:     model_name -> (base_path, target_device, plugin_config, plain).
                       plugin_config (OpenVINO device properties) is emitted only
                       when non-empty; for LLM models it travels via graph.pbtxt.
                       plain (model_config_list options) is emitted only when non-empty.
    Empty dicts render empty lists.

    Raises OSError if file I/O fails.
    """
    mediapipe_rendered = [
        {
            "name": model_name,
            "base_path": str(base_path),
            "graph_path": graph_path,
        }
        for model_name, (base_path, graph_path) in mediapipe_entries.items()
    ]
    model_rendered = []
    for model_name, (base_path, target_device, plugin_config, plain) in model_entries.items():
        # Start with plain options (user-supplied model_config_list fields).
        config: dict[str, object] = dict(plain or {})
        # Set rig-owned structural keys last so they always win.
        config["name"] = model_name
        config["base_path"] = str(base_path)
        config["target_device"] = target_device
        if plugin_config:
            config["plugin_config"] = dict(plugin_config)
        model_rendered.append({"config": config})

    data = {
        "model_config_list": model_rendered,
        "mediapipe_config_list": mediapipe_rendered,
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug(
        "[registry] rendered config.json: mediapipe=%s model=%s",
        list(mediapipe_entries.keys()), list(model_entries.keys()),
    )
