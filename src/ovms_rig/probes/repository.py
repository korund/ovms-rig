"""Check that the fetch destination device is reachable, and list which
declared models are already materialized on disk.

We deliberately do NOT verify directory layout inside the store -- that is
fetch's job. We only confirm:
  - if a destination path is configured, that the nearest existing ancestor
    is on a reachable, writable device (so fetch can actually create the
    leaf directory);
  - which declared models, by name, already exist as subdirectories.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ovms_rig.config import Declaration, OvmsConfig
from ovms_rig.report import CheckResult

DESTINATION = "model store destination"
INVENTORY = "declared models on disk"


def check_destination(decl: Declaration) -> CheckResult:
    local = decl.local
    # repository_path is required by schema; no None-branch needed here.
    path = local.models.repository_path
    anchor = _nearest_existing_ancestor(path)
    if anchor is None:
        return CheckResult(
            name=DESTINATION,
            status="error",
            summary=f"no existing ancestor for {path} (device unreachable?)",
            details={"path": str(path)},
        )
    if not _is_writable(anchor):
        return CheckResult(
            name=DESTINATION,
            status="error",
            summary=f"{anchor} is not writable",
            details={"path": str(path), "anchor": str(anchor)},
        )
    return CheckResult(
        name=DESTINATION,
        status="ok",
        summary=str(path),
        details={"anchor": str(anchor), "leaf_exists": path.exists()},
    )


def check_inventory(decl: Declaration) -> CheckResult:
    ovms = decl.ovms
    local = decl.local
    store = local.models.repository_path
    declared = sorted(ovms.repository)
    if not store.exists():
        return CheckResult(
            name=INVENTORY,
            status="ok",
            summary=f"0/{len(declared)} present (store not materialized)",
            details={"declared": declared, "present": [], "missing": declared},
            hint="run `ovms-rig fetch` to populate the store",
        )
    # Inventory keys by short name (ovms.repository key) but probes the disk
    # at the HF path -- which is the layout produced by `ovms --pull`.
    present = [name for name in declared if _weights_dir(store, ovms, name).is_dir()]
    missing = [name for name in declared if name not in present]
    summary = f"{len(present)}/{len(declared)} present"
    hint = "run `ovms-rig fetch` to populate missing models" if missing else None
    return CheckResult(
        name=INVENTORY,
        status="ok",
        summary=summary,
        details={"declared": declared, "present": present, "missing": missing},
        hint=hint,
    )


def _weights_dir(store: Path, ovms: OvmsConfig, name: str) -> Path:
    """Map a declared model name to its on-disk location.

    Resolves either hf (HuggingFace org/repo) or dir (local directory) source.
    """
    model = ovms.repository[name]
    return model.weights_dir(store)


def _nearest_existing_ancestor(path: Path) -> Path | None:
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return candidate
    return None


def _is_writable(directory: Path) -> bool:
    try:
        with tempfile.TemporaryFile(dir=directory):
            return True
    except OSError:
        return False
