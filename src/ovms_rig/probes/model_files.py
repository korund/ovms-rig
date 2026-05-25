"""Check presence of required per-model files on disk for the active profile.

Verifies that each model in the active profile has:
  - graph.pbtxt (required) - missing blocks OVMS load
  - generation_config.json (optional) - missing drops sampling overrides

Silently skips models whose directories do not exist on disk (handled by
repository.inventory probe). Returns ok if no active profile is set.
"""

from __future__ import annotations

from pathlib import Path

from ovms_rig.config import Declaration
from ovms_rig.report import CheckResult

NAME = "model files"


def check(decl: Declaration) -> CheckResult:
    ovms = decl.ovms
    local = decl.local
    store = local.models.repository_path

    # Find active profile.
    active_profile_name: str | None = None
    for profile_name, profile in ovms.profiles.items():
        if profile.active:
            active_profile_name = profile_name
            break

    if active_profile_name is None:
        return CheckResult(
            name=NAME,
            status="ok",
            summary="no active profile -- nothing to check",
        )

    active_profile = ovms.profiles[active_profile_name]
    checked = []
    missing_required: dict[str, list[str]] = {}
    missing_optional: dict[str, list[str]] = {}

    for model_name in active_profile.models:
        # Skip if model not in ovms.models (declared but not found).
        if model_name not in ovms.models:
            continue

        entry = ovms.models[model_name]
        dirs_to_check = []

        # Primary model directory.
        primary_dir = _weights_dir(store, ovms, entry.source)
        dirs_to_check.append((model_name, "primary", primary_dir))

        # Draft model directory, if specified.
        if entry.graph.draft_model:
            draft_dir = _weights_dir(store, ovms, entry.graph.draft_model)
            dirs_to_check.append((model_name, "draft", draft_dir))

        for check_model_name, model_type, model_dir in dirs_to_check:
            # Skip if directory does not exist on disk.
            if not model_dir.is_dir():
                continue

            checked.append(check_model_name)

            # Check required files.
            graph_file = model_dir / "graph.pbtxt"
            if not graph_file.exists():
                if check_model_name not in missing_required:
                    missing_required[check_model_name] = []
                missing_required[check_model_name].append("graph.pbtxt")

            # Check optional files.
            gen_config_file = model_dir / "generation_config.json"
            if not gen_config_file.exists():
                if check_model_name not in missing_optional:
                    missing_optional[check_model_name] = []
                missing_optional[check_model_name].append("generation_config.json")

    # Deduplicate checked list and sort.
    checked = sorted(set(checked))

    # Determine status.
    if missing_required:
        status = "error"
        summary = f"{len(missing_required)}/{len(checked)} models missing required files"
    else:
        status = "ok"
        summary = f"{len(checked)}/{len(checked)} models complete" if checked else "no models materialized"

    # Build hint.
    hints = []
    if missing_required:
        hints.append("missing graph.pbtxt blocks OVMS load")
    if missing_optional:
        hints.append("missing generation_config.json drops sampling overrides")
    hint = "; ".join(hints) if hints else None

    return CheckResult(
        name=NAME,
        status=status,
        summary=summary,
        details={
            "checked": checked,
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        },
        hint=hint,
    )


def _weights_dir(store: Path, ovms_config, model_key: str) -> Path:
    """Map a model repository key to its on-disk location (HF-layout)."""
    return store / ovms_config.repository[model_key].hf
