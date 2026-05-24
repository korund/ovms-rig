"""Cleanup of obsolete sibling-graph files from previous activations."""

from __future__ import annotations

import logging
from pathlib import Path

from ovms_rig.config import OvmsConfig

logger = logging.getLogger(__name__)


def cleanup_obsolete_sibling_graphs(store: Path, active_models: set[str], ovms: OvmsConfig) -> list[str]:
    """Remove sibling-graphs for models not in active profile.

    Scans model directories from ovms.repository and removes graph.<name>.pbtxt
    files whose <name> is not in active_models.

    Returns list of cleaned-up paths.
    """
    cleaned_up: list[str] = []
    seen_dirs: set[Path] = set()

    for repo_name, identity in ovms.repository.items():
        model_dir = store / identity.hf
        if model_dir in seen_dirs or not model_dir.is_dir():
            continue
        seen_dirs.add(model_dir)

        for sibling_graph in model_dir.glob("graph.*.pbtxt"):
            # Extract model name from filename: graph.<name>.pbtxt
            # Use Path.stem to remove .pbtxt, then removeprefix to get name.
            stem = sibling_graph.stem  # removes .pbtxt
            name = stem.removeprefix("graph.")

            # Skip if name is empty (malformed filename).
            if not name:
                logger.debug("[cleanup] skipping malformed sibling-graph: %s", sibling_graph)
                continue

            # If model not in active_models, remove the sibling-graph.
            if name not in active_models:
                try:
                    sibling_graph.unlink()
                    cleaned_up.append(str(sibling_graph))
                    logger.debug("[cleanup] removed obsolete sibling-graph: %s", sibling_graph)
                except OSError as exc:
                    logger.warning("[cleanup] failed to remove %s: %s", sibling_graph, exc)

    return cleaned_up
