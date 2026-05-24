"""Check declared models (endpoints) and their sources.

Verifies:
- Each model's source exists in repository.
- Which profiles contain each model.
"""

from __future__ import annotations

from ovms_rig.config import OvmsConfig
from ovms_rig.report import CheckResult

NAME = "models (endpoints)"


def check(ovms: OvmsConfig) -> CheckResult:
    if not ovms.models:
        return CheckResult(
            name=NAME,
            status="ok",
            summary="no models declared",
        )

    # Build model details with source and profile membership.
    model_details: dict[str, dict] = {}
    source_errors: list[str] = []

    for model_name, entry in ovms.models.items():
        source_name = entry.source

        # Verify source exists in repository.
        if source_name not in ovms.repository:
            source_errors.append(f"{model_name}: source '{source_name}' not in repository")
            source_status = "missing"
        else:
            source_status = "ok"

        # Find which profiles contain this model.
        profiles_containing = [
            pname for pname, profile in ovms.profiles.items()
            if model_name in profile.models
        ]

        model_details[model_name] = {
            "source": source_name,
            "source_status": source_status,
            "profiles": profiles_containing,
        }

    # Determine overall status and summary.
    if source_errors:
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"{len(ovms.models)} declared, {len(source_errors)} source error(s)",
            details={"models": model_details, "errors": source_errors},
        )

    summary = f"{len(ovms.models)} model(s), all sources valid"
    return CheckResult(
        name=NAME,
        status="ok",
        summary=summary,
        details={"models": model_details},
    )
