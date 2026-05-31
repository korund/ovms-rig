"""Check live OVMS config (config.json) vs active profile.

Compares the models registered in live config.json -- across both
mediapipe_config_list (LLM) and model_config_list (plain) -- with the set of
models that should be active based on the active profile.
"""

from __future__ import annotations

import json

from ovms_rig.config import Declaration
from ovms_rig.report import CheckResult

NAME = "live ovms config"


def check(decl: Declaration) -> CheckResult:
    ovms = decl.ovms
    local = decl.local
    store = local.models.repository_path
    config_json_path = store / "config.json"

    # Determine which models should be active.
    active_models: set[str] = set()
    active_profile_name: str | None = None
    for profile_name, profile in ovms.profiles.items():
        if profile.active:
            active_profile_name = profile_name
            active_models = set(profile.models)
            break

    # If config.json doesn't exist and no profile is active, that's OK.
    if not config_json_path.exists():
        if active_profile_name is None:
            return CheckResult(
                name=NAME,
                status="ok",
                summary="config.json not present (no active profile, as expected)",
            )
        else:
            return CheckResult(
                name=NAME,
                status="warn",
                summary=f"config.json missing (profile '{active_profile_name}' is active)",
                hint="run `ovms-rig activate {profile_name}` to rebuild live config",
            )

    # Read config.json and extract mediapipe_config_list.
    try:
        data = json.loads(config_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return CheckResult(
            name=NAME,
            status="error",
            summary=f"failed to read config.json: {e}",
        )

    # Live models come from both lists: mediapipe_config_list (LLM, name at top
    # level) and model_config_list (plain, name nested under "config").
    live_models: set[str] = set()
    for entry in data.get("mediapipe_config_list", []):
        name = entry.get("name")
        if name:
            live_models.add(name)
    for entry in data.get("model_config_list", []):
        name = entry.get("config", {}).get("name")
        if name:
            live_models.add(name)

    # Compare expected vs actual.
    if active_models == live_models:
        if active_models:
            summary = f"OK: {len(active_models)} model(s) from profile '{active_profile_name}'"
        else:
            summary = "OK: empty config (no active profile)"
        return CheckResult(
            name=NAME,
            status="ok",
            summary=summary,
            details={
                "active_profile": active_profile_name,
                "expected_models": sorted(active_models),
                "live_models": sorted(live_models),
            },
        )

    # Mismatch.
    extra = live_models - active_models
    missing = active_models - live_models
    summary = f"mismatch vs profile '{active_profile_name}' (expected {len(active_models)}, got {len(live_models)})"
    return CheckResult(
        name=NAME,
        status="warn",
        summary=summary,
        details={
            "active_profile": active_profile_name,
            "expected_models": sorted(active_models),
            "live_models": sorted(live_models),
            "extra_in_live": sorted(extra),
            "missing_from_live": sorted(missing),
        },
        hint="run `ovms-rig activate {profile_name}` to rebuild live config".format(
            profile_name=active_profile_name or "(none)"
        ),
    )
