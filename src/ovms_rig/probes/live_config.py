"""Check presence of live OVMS config files for declared served entries.

Content comparison vs the declaration is apply's job. Here we only report
which files exist on disk.
"""

from __future__ import annotations

from ovms_rig.config import LocalConfig, OvmsConfig
from ovms_rig.report import CheckResult

NAME = "live ovms config"

CONFIG_JSON = "config.json"
GRAPH_PBTXT = "graph.pbtxt"


def check(ovms: OvmsConfig, local: LocalConfig) -> CheckResult:
    store = local.models.repository_path
    if store is None or not store.exists():
        return CheckResult(
            name=NAME,
            status="ok",
            summary="store not materialized; no live config to inspect",
        )

    present: dict[str, list[str]] = {}
    for entry in ovms.served:
        model_dir = store / entry.model
        files = [
            f for f in (CONFIG_JSON, GRAPH_PBTXT) if (model_dir / f).is_file()
        ]
        present[entry.name] = files

    total_files = sum(len(v) for v in present.values())
    expected = 2 * len(ovms.served)
    hint = (
        "run `ovms-rig apply` to materialize live config"
        if total_files < expected
        else None
    )
    return CheckResult(
        name=NAME,
        status="ok",
        summary=f"{total_files}/{expected} files present",
        details={"per_served": present},
        hint=hint,
    )
