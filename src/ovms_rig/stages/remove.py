"""Remove artifacts for a single repository entry (inverse of fetch).

Removes:
- Model directory on disk
- Entry from config.json mediapipe_config_list

Never touches ovms.yaml (repository, models, profiles).
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from ovms_rig import log as logging_setup
from ovms_rig.config import ConfigError, load_declaration
from ovms_rig.stages.activation.paths import model_dir

logger = logging.getLogger(__name__)


def run(ctx: dict) -> int:
    """Remove artifacts for repository_name.

    Args:
        ctx: dict with config_path, local_path, ovms_path, log_level, repository_name, force.

    Returns:
        0 on success or nothing-to-do, 1 on error.
    """
    config_path = Path(ctx["config_path"])
    local_path = Path(ctx["local_path"])
    ovms_override = Path(ctx["ovms_path"]) if ctx.get("ovms_path") else None
    cli_level: str | None = ctx.get("log_level")
    repository_name = ctx["repository_name"]
    force = ctx.get("force", False)

    # Logging bootstrapped at default level.
    logging_setup.configure((cli_level or "INFO").upper())

    try:
        decl = load_declaration(config_path, local_path, cli_override=ovms_override)
        ovms = decl.ovms
        local = decl.local
    except ConfigError as e:
        logger.error("config load failed: %s", e)
        return 1

    level = (cli_level or ovms.runtime.log_level).upper()
    logging_setup.configure(level)
    logger.debug("log level: %s (source: %s)",
                 level, "cli" if cli_level else "ovms.yaml")
    logger.debug("config: %s", config_path)
    logger.debug("local:  %s", local_path)

    # Check that repository_name exists in ovms.repository.
    if repository_name not in ovms.repository:
        logger.error("unknown repository: %s", repository_name)
        return 1

    store = local.models.repository_path
    model_identity = ovms.repository[repository_name]
    model_path = model_dir(store, model_identity.hf)

    # Check that directory exists (nothing to do if not fetched).
    if not model_path.is_dir():
        logger.info("nothing to do: %s not fetched", repository_name)
        return 0

    # Pre-check: scan for references in ovms.yaml (all profiles).
    references = _find_references(ovms, repository_name)
    if references and not force:
        logger.error("cannot remove: %s", repository_name)
        logger.error("  referenced by:")
        for profile_name, ref_types in sorted(references.items()):
            for ref_type in sorted(ref_types):
                logger.error("    - profile %s: %s", profile_name, ref_type[1])
        logger.error("use --force to override")
        return 1

    # Success path: snapshot, remove from config.json, delete directory.
    try:
        # Snapshot for reporting.
        size_bytes = _dir_size(model_path)
        size_str = _format_size(size_bytes)

        # Remove from config.json mediapipe_config_list.
        config_json_path = store / "config.json"
        config_entry_removed = False
        if config_json_path.exists():
            config_entry_removed = _remove_from_config(config_json_path, repository_name)

        # Delete the model directory.
        shutil.rmtree(model_path)
        logger.info("[remove] %s: deleted directory", repository_name)

        # Clean up empty parent directories (cosmetic).
        _cleanup_empty_parents(model_path.parent, store)

        # Report success.
        logger.info("removed: %s", repository_name)
        logger.info("  path: %s", model_path)
        logger.info("  size: %s", size_str)
        if config_entry_removed:
            logger.info("  config.json: entry removed (mediapipe_config_list)")

        # Warn about orphan draft references.
        orphans = _find_orphan_drafts(ovms, repository_name)
        for model_name, draft_model in orphans:
            logger.warning(
                "warning: orphan draft reference -- models.%s.graph.draft_model = \"%s\"",
                model_name, draft_model
            )

        return 0

    except (OSError, ValueError) as e:
        logger.error("remove failed: %s", e)
        return 1


def _find_references(ovms, repository_name: str) -> dict[str, set[tuple[str, str]]]:
    """Find all references to repository_name in ovms.yaml.

    Returns dict mapping profile_name -> set of (ref_type, description).
    ref_type: "models entry" or "draft_model"
    """
    references: dict[str, set[tuple[str, str]]] = {}

    for profile_name, profile in ovms.profiles.items():
        # Check if repository_name is a source of any model in this profile.
        for model_name in profile.models:
            model_entry = ovms.models[model_name]
            if model_entry.source == repository_name:
                if profile_name not in references:
                    references[profile_name] = set()
                references[profile_name].add(
                    ("models entry", f"models.{model_name}.source")
                )

            # Check if repository_name is a draft_model in this profile.
            if model_entry.graph.draft_model == repository_name:
                if profile_name not in references:
                    references[profile_name] = set()
                references[profile_name].add(
                    ("draft_model", f"models.{model_name}.graph.draft_model")
                )

    return references


def _find_orphan_drafts(ovms, removed_name: str) -> list[tuple[str, str]]:
    """Find models that reference removed_name as draft_model.

    Returns list of (model_name, draft_model) tuples.
    """
    orphans = []
    for model_name, model_entry in ovms.models.items():
        if model_entry.graph.draft_model == removed_name:
            orphans.append((model_name, removed_name))
    return orphans


def _dir_size(path: Path) -> int:
    """Calculate total size in bytes of directory and contents."""
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _format_size(size_bytes: int) -> str:
    """Format bytes to human-readable string (GB, MB, etc.)."""
    for unit in ("B", "MB", "GB", "TB"):
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes} {unit}"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _remove_from_config(config_json_path: Path, repository_name: str) -> bool:
    """Remove entry with name=repository_name from mediapipe_config_list.

    Returns True if an entry was actually removed, False otherwise.
    """
    try:
        data = json.loads(config_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # If config.json doesn't exist or is malformed, ignore.
        return False

    if "mediapipe_config_list" not in data:
        return False

    entries = data["mediapipe_config_list"]

    # Keep entries that don't match repository_name.
    reconciled = [
        e for e in entries
        if e.get("name") != repository_name
    ]

    if len(reconciled) < len(entries):
        # Something was removed; write back.
        data["mediapipe_config_list"] = reconciled
        config_json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("[remove] config.json: entry '%s' removed", repository_name)
        return True
    return False


def _cleanup_empty_parents(directory: Path, stop_at: Path) -> None:
    """Remove empty parent directories up to (but not including) stop_at.

    Walks up from directory, removing directories if empty, until hitting
    stop_at or a non-empty directory.
    """
    current = directory
    stop_at = stop_at.resolve()

    while current != stop_at and current != current.parent:
        if not current.exists():
            current = current.parent
            continue

        try:
            # Try to remove; will fail if not empty.
            current.rmdir()
            logger.debug("[remove] cleaned up empty directory: %s", current)
            current = current.parent
        except OSError:
            # Directory not empty or other error; stop here.
            break
