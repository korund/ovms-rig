"""Filesystem path helpers for the apply stage.

Computes the relative POSIX path from a target model directory to a draft
model directory, covering both same-org and cross-org layouts.

OVMS --pull layout (flat siblings under store):
    <store>/<hf_org>/<hf_repo>/

Examples
--------
Same org:
    target: store/OpenVINO/Qwen3-14B-int8-ov
    draft:  store/OpenVINO/Qwen3-0.6B-int8-ov
    rel:    ../Qwen3-0.6B-int8-ov

Cross org:
    target: store/OrgA/main-model
    draft:  store/OrgB/draft-model
    rel:    ../../OrgB/draft-model
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


def relative_posix(target_dir: Path, draft_dir: Path) -> str:
    """Return a POSIX-style relative path from target_dir to draft_dir.

    Both paths must be absolute. The result uses forward slashes and starts
    with '..' when navigating upward (never starts with '/').
    """
    # Use PurePosixPath for string building after computing parts.
    target = target_dir.resolve()
    draft = draft_dir.resolve()

    # Find common ancestor.
    try:
        rel = draft.relative_to(target)
        # draft is inside target -- uncommon but handle it.
        return rel.as_posix()
    except ValueError:
        pass

    # Walk up from target until we can express draft relative to that ancestor.
    ups = 0
    ancestor = target
    while True:
        try:
            down = draft.relative_to(ancestor)
            prefix = "/".join([".."] * ups) if ups else "."
            return (PurePosixPath(prefix) / down).as_posix()
        except ValueError:
            ancestor = ancestor.parent
            ups += 1
            if ancestor == ancestor.parent:
                # Reached filesystem root with no common ancestor (different drives
                # on Windows). Fall back to absolute POSIX path.
                return draft.as_posix().replace("\\", "/")


def model_dir(store: Path, hf_id: str) -> Path:
    """Expand an HF id (org/repo) to its on-disk directory under store."""
    return store / hf_id


def resolve_model_dir(store: Path, hf_id: str | None, local_dir: str | None) -> Path:
    """Resolve a model directory from either hf (HuggingFace) or dir (local) source.

    Args:
        store: root of the model repository (from local.yaml).
        hf_id: HF coordinate (org/repo), or None if using local_dir.
        local_dir: local directory path, or None if using hf_id.
               If relative, resolved against store. If absolute, used as-is.

    Returns:
        Path to the model directory. For hf sources, returns store / hf_id as-is
        (may contain symlinks or be unresolved). For dir sources, returns a
        normalized absolute path (via .resolve()).

    Note:
        Path(store) / abs_path already yields abs_path on each OS when abs_path
        is absolute, so we leverage that directly for both branches. Relative
        dir paths are joined to store and then resolved. Windows edge case
        (drive-relative paths) are handled by Path's native semantics.
    """
    if hf_id is not None:
        return store / hf_id
    if local_dir is not None:
        result = store / local_dir
        return result.resolve()
    raise ValueError("both hf_id and local_dir are None")
