"""Entry point and subcommand dispatcher for the ovms-rig loader."""

from __future__ import annotations

import sys

import click

from ovms_rig.stages import apply, fetch, profile, start, status

LOG_LEVELS = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"]


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config/ovms.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to the OVMS declaration.",
)
@click.option(
    "--local",
    "local_path",
    default="config/local.yaml",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Path to per-host overrides. Missing file is fine.",
)
@click.option(
    "--ovms-path",
    "ovms_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Override path to the ovms binary. Wins over local.yaml and PATH.",
)
@click.option(
    "--log-level",
    "log_level",
    default=None,
    type=click.Choice(LOG_LEVELS, case_sensitive=False),
    help="Override runtime log level from ovms.yaml.",
)
@click.pass_context
def main(
    ctx: click.Context,
    config_path: str,
    local_path: str,
    ovms_path: str | None,
    log_level: str | None,
) -> None:
    """Declarative loader for OpenVINO Model Server."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config_path
    ctx.obj["local_path"] = local_path
    ctx.obj["ovms_path"] = ovms_path
    ctx.obj["log_level"] = log_level


@main.command("status")
@click.pass_context
def cmd_status(ctx: click.Context) -> None:
    """Report current state of the rig vs declaration. Read-only, no side effects."""
    sys.exit(status.run(ctx.obj))


@main.command(
    "fetch",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.argument("repository_name")
@click.pass_context
def cmd_fetch(ctx: click.Context, repository_name: str) -> None:
    """Pull a single repository entry. Extra args forwarded to `ovms --pull` verbatim."""
    ctx.obj["repository_name"] = repository_name
    ctx.obj["extras"] = list(ctx.args)
    sys.exit(fetch.run(ctx.obj))


@main.command("activate")
@click.argument("profile_name")
@click.pass_context
def cmd_activate(ctx: click.Context, profile_name: str) -> None:
    """Activate a profile: update ovms.yaml and rebuild live config."""
    sys.exit(profile.set_active_profile(ctx.obj, profile_name))


@main.command("deactivate")
@click.pass_context
def cmd_deactivate(ctx: click.Context) -> None:
    """Deactivate all profiles: live config becomes empty."""
    sys.exit(profile.set_active_profile(ctx.obj, None))


@main.command(
    "start",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def cmd_start(ctx: click.Context) -> None:
    """Start ovms in the foreground. Extra args forwarded to ovms."""
    ctx.obj["extras"] = list(ctx.args)
    sys.exit(start.run(ctx.obj))


if __name__ == "__main__":
    main()
