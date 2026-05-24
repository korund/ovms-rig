"""Direct manipulation of config.json to register mediapipe_config_list entries.

Instead of shelling out to `ovms --add_to_config`, this module reads/writes
the config.json directly. Two modes of operation:
- register_mediapipe_entry: upsert a single entry
- reconcile_mediapipe_entries: reconcile the entire list (keep only specified entries)

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


def register_mediapipe_entry(
    config_path: Path,
    entry_name: str,
    base_path: Path,
    graph_path: str,
) -> None:
    """Register a mediapipe_config_list entry in config.json.

    config_path: path to config.json (created if missing).
    entry_name: name of the model entry.
    base_path: absolute path to model directory.
    graph_path: relative path from base_path to the graph file
                (typically "graph.<model_name>.pbtxt").

    Raises OSError if file I/O fails, ValueError if JSON is malformed.
    """
    # Load existing config or start with empty dict.
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    else:
        data = {}

    # Ensure mediapipe_config_list exists.
    if "mediapipe_config_list" not in data:
        data["mediapipe_config_list"] = []

    entries = data["mediapipe_config_list"]

    # Find and update or append.
    found = False
    for entry in entries:
        if entry.get("name") == entry_name:
            entry["base_path"] = str(base_path)
            entry["graph_path"] = graph_path
            found = True
            break

    if not found:
        entries.append({
            "name": entry_name,
            "base_path": str(base_path),
            "graph_path": graph_path,
        })

    # Write back with nice formatting.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("[registry] registered mediapipe entry: name=%s, base_path=%s, graph_path=%s",
                 entry_name, base_path, graph_path)


def reconcile_mediapipe_entries(
    config_path: Path,
    desired_entries: dict[str, tuple[Path, str]],
) -> None:
    """Reconcile mediapipe_config_list to contain exactly the desired entries.

    desired_entries: dict mapping model_name -> (base_path, graph_path).
    Removes any entries not in desired_entries, updates existing ones,
    and adds missing ones.

    If desired_entries is empty, mediapipe_config_list becomes [].

    Raises OSError if file I/O fails, ValueError if JSON is malformed.
    """
    # Load existing config or start with empty dict.
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    else:
        data = {}

    # Ensure mediapipe_config_list exists.
    if "mediapipe_config_list" not in data:
        data["mediapipe_config_list"] = []

    entries = data["mediapipe_config_list"]

    # Build map of existing entries by name for efficient lookup.
    # Skip entries without a name (warn if found).
    existing_by_name = {}
    for e in entries:
        entry_name = e.get("name")
        if entry_name is None:
            logger.warning("[registry] skipping entry without name in existing config.json: %s", e)
            continue
        existing_by_name[entry_name] = e

    # Reconcile: keep only entries in desired_entries, update or add as needed.
    reconciled = []
    for model_name, (base_path, graph_path) in desired_entries.items():
        if model_name in existing_by_name:
            # Update existing entry.
            entry = existing_by_name[model_name]
            entry["base_path"] = str(base_path)
            entry["graph_path"] = graph_path
            reconciled.append(entry)
        else:
            # Add new entry.
            reconciled.append({
                "name": model_name,
                "base_path": str(base_path),
                "graph_path": graph_path,
            })

    data["mediapipe_config_list"] = reconciled

    # Write back with nice formatting.
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.debug("[registry] reconciled mediapipe entries: desired=%s", list(desired_entries.keys()))
