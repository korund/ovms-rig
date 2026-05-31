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
