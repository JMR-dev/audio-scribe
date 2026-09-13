"""Command line interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, NoReturn

import click

from ccn_transcribe import doctor as doctor_mod
from ccn_transcribe import errors, logsetup, paths
from ccn_transcribe.config import DEFAULT_FORMATS, RunConfig
from ccn_transcribe.formats import FORMATS
from ccn_transcribe.jobs import batch, store
from ccn_transcribe.jobs.plan import Stage
from ccn_transcribe.media import ffmpeg, ytdlp_opts
from ccn_transcribe.stages import download as download_stage

STAGE_NAMES = [s.value for s in Stage]


def parse_formats(raw: str) -> tuple[str, ...]:
    wanted = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [name for name in wanted if name not in FORMATS]
    if unknown:
        raise errors.ConfigError(
            f"unknown output format(s): {', '.join(unknown)}",
            hint=f"Choose from: {', '.join(FORMATS)}",
        )
    return wanted or DEFAULT_FORMATS


def build_config(**kw: Any) -> RunConfig:
    """Turn CLI options into a RunConfig."""
    workdir = Path(str(kw["workdir"]))
    use_aria2c = download_stage.resolve_aria2c(
        requested=not bool(kw["no_aria2c"]),
        available=download_stage.aria2c_available(),
    )
    allocation = ytdlp_opts.allocation_for(ytdlp_opts.filesystem_type(workdir.parent))
    download = ytdlp_opts.DownloadConfig(
        max_height=kw["max_height"],
        use_aria2c=use_aria2c,
        playlist=bool(kw["playlist"]),
        connections=int(kw["aria2c_conns"]),
        concurrent_fragments=int(kw["concurrent_fragments"]),
        retries=int(kw["retries"]),
        cookies_from_browser=kw["cookies_from_browser"],
        js_runtime=kw["js_runtime"],
        file_allocation=allocation,
        lowest_speed=str(kw["aria2c_lowest_speed"]),
        extra_aria2c_args=tuple(kw["aria2c_arg"]),
        quiet=bool(kw["quiet"]),
    )
    no_retain = bool(kw["no_retain"])
    force_stages = frozenset(Stage(s) for s in kw["force_stage"])
    if kw["force"]:
        force_stages = frozenset(Stage)
    return RunConfig(
        workdir=workdir,
        out_dir=Path(str(kw["out_dir"])) if kw["out_dir"] else None,
        formats=parse_formats(str(kw["formats"])),
        language=kw["language"],
        task=str(kw["task"]),
        num_beams=int(kw["num_beams"]),
        initial_prompt=kw["initial_prompt"],
        device=kw["device"],
        model=kw["model"],
        ov_cache_dir=Path(str(kw["ov_cache_dir"])) if kw["ov_cache_dir"] else None,
        audio_profile=str(kw["audio_profile"]),
        no_retain_video=no_retain or bool(kw["no_retain_video"]),
        no_retain_audio=no_retain or bool(kw["no_retain_audio"]),
        assume_yes=bool(kw["yes"]),
        force=force_stages,
        download=download,
    )


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="ccn-transcribe")
def main() -> None:
    """Download videos, extract FLAC, and transcribe with Whisper."""


@main.command("run")
@click.argument("urls", nargs=-1)
@click.option(
    "-i",
    "--input-file",
    "input_files",
    multiple=True,
    type=click.Path(path_type=Path),
    help="File of URLs, one per line ('-' for stdin).",
)
@click.option(
    "-w",
    "--workdir",
    default="transcripts",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Where job directories live.",
)
@click.option(
    "--out-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Also copy finished transcripts here, named by title.",
)
@click.option(
    "--formats",
    default=",".join(DEFAULT_FORMATS),
    show_default=True,
    help="Comma-separated output formats.",
)
@click.option(
    "--device",
    "--backend",
    default=None,
    help="auto (default), openvino, cpu, nvidia or amd. Explicit means no fallback.",
)
@click.option("--model", default=None, help="Override the backend's default model.")
@click.option("--language", default=None, help="Language code, e.g. en. Omit to autodetect.")
@click.option(
    "--task",
    type=click.Choice(["transcribe", "translate"]),
    default="transcribe",
    show_default=True,
)
@click.option("--num-beams", default=1, show_default=True)
@click.option("--initial-prompt", default=None, help="Steers the first window only.")
@click.option(
    "--audio-profile",
    type=click.Choice(["source", "whisper"]),
    default="source",
    show_default=True,
    help="source keeps the original rate and channels.",
)
@click.option("--playlist", is_flag=True, help="Expand a playlist instead of one video.")
@click.option("--max-height", default=1080, show_default=True, help="Cap the video height.")
@click.option("--no-aria2c", is_flag=True, help="Use yt-dlp's native downloader.")
@click.option("--aria2c-conns", default=16, show_default=True)
@click.option(
    "--aria2c-lowest-speed",
    default="50K",
    show_default=True,
    help="Abort a stalled connection below this rate; 0 disables.",
)
@click.option("--aria2c-arg", multiple=True, help="Extra aria2c argument (repeatable).")
@click.option("--concurrent-fragments", default=16, show_default=True)
@click.option("--retries", default=5, show_default=True)
@click.option("--cookies-from-browser", default=None, help="e.g. firefox")
@click.option("--js-runtime", default=None, help="deno (default), node, bun or quickjs.")
@click.option("--ov-cache-dir", default=None, type=click.Path(path_type=Path))
@click.option("--no-retain", is_flag=True, help="Delete both video and audio when done.")
@click.option("--no-retain-video", is_flag=True, help="Delete the video after extraction.")
@click.option("--no-retain-audio", is_flag=True, help="Delete the FLAC after transcription.")
@click.option("-y", "--yes", is_flag=True, help="Confirm destructive flags non-interactively.")
@click.option("--force", is_flag=True, help="Re-run every stage.")
@click.option(
    "--force-stage",
    multiple=True,
    type=click.Choice(STAGE_NAMES),
    help="Re-run from this stage (repeatable).",
)
@click.option("--dry-run", is_flag=True, help="Print the resume plan and exit.")
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet", is_flag=True)
def run_command(urls: tuple[str, ...], input_files: tuple[Path, ...], **kw: Any) -> None:
    """Transcribe one or more URLs."""
    logsetup.configure(int(kw["verbose"]), bool(kw["quiet"]))
    try:
        ffmpeg.require_binaries()
        config = build_config(**kw)
        targets = batch.read_urls(urls, input_files)
        if not targets:
            raise errors.ConfigError("no URLs given", hint="Pass a URL or -i urls.txt.")
        batch.confirm_retention(config)
        result = batch.run_batch(targets, config, dry_run=bool(kw["dry_run"]))
    except errors.CcnError as exc:
        _fail(exc, verbose=bool(kw["verbose"]))
    click.echo(result.summary())
    sys.exit(result.exit_code)


@main.command("doctor")
def doctor_command() -> None:
    """Check the environment and explain anything wrong."""
    checks = doctor_mod.run_checks()
    click.echo(doctor_mod.render(checks))
    sys.exit(doctor_mod.worst(checks))


@main.command("status")
@click.argument("job_ids", nargs=-1)
@click.option("-w", "--workdir", default="transcripts", type=click.Path(path_type=Path))
def status_command(job_ids: tuple[str, ...], workdir: Path) -> None:
    """Show what each job has on disk."""
    workspace = paths.Workspace(workdir)
    wanted = list(job_ids) or sorted(p.name for p in workspace.jobs_dir.glob("*") if p.is_dir())
    if not wanted:
        click.echo("no jobs yet")
        return
    for name in wanted:
        job = workspace.job(name)
        state = store.load(job)
        if state is None:
            click.echo(f"{name}: no state recorded")
            continue
        click.echo(
            f"{name}: video={state.video.status} audio={state.audio.status} "
            f"outputs={state.outputs.status} title={state.title or '-'}"
        )


def _fail(exc: errors.CcnError, *, verbose: bool) -> NoReturn:
    click.echo(f"ERROR: {exc}", err=True)
    if exc.hint:
        click.echo(f"hint: {exc.hint}", err=True)
    if verbose:
        raise exc
    sys.exit(exc.exit_code)


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    main()
