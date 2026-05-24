"""Check profile declarations and active status.

Reports:
- List of profiles with their models.
- Which profile is active (if any).
- Model-to-profile membership.
"""

from __future__ import annotations

from ovms_rig.config import OvmsConfig
from ovms_rig.report import CheckResult

NAME = "profiles"


def check(ovms: OvmsConfig) -> CheckResult:
    if not ovms.profiles:
        return CheckResult(
            name=NAME,
            status="ok",
            summary="no profiles declared (all models are unavailable)",
        )

    # Find active profile.
    active_profile_name: str | None = None
    for profile_name, profile in ovms.profiles.items():
        if profile.active:
            active_profile_name = profile_name
            break

    # Build model-to-profiles membership mapping.
    model_membership: dict[str, list[str]] = {}
    for model_name in ovms.models:
        model_membership[model_name] = []

    for profile_name, profile in ovms.profiles.items():
        for model_name in profile.models:
            if model_name in model_membership:
                model_membership[model_name].append(profile_name)

    # Build details.
    profile_details: dict[str, dict] = {}
    for profile_name, profile in ovms.profiles.items():
        profile_details[profile_name] = {
            "active": profile.active,
            "models": profile.models,
        }

    summary = (
        f"active profile: '{active_profile_name}' ({len(ovms.profiles)} total)"
        if active_profile_name
        else f"no active profile ({len(ovms.profiles)} total)"
    )

    return CheckResult(
        name=NAME,
        status="ok",
        summary=summary,
        details={
            "profiles": profile_details,
            "model_membership": model_membership,
        },
    )
