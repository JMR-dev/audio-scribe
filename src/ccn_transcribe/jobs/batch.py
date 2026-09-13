"""Batch orchestration: expand inputs, run each job, summarize."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TextIO

from ccn_transcribe import errors, logsetup, paths
from ccn_transcribe.backends.registry import BackendHolder
from ccn_transcribe.jobs import store
from ccn_transcribe.jobs.runner import JobOutcome, run_job
from ccn_transcribe.stages import download as download_stage

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ccn_transcribe.config import RunConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchResult:
    outcomes: list[JobOutcome] = field(default_factory=list[JobOutcome])

    @property
    def failed(self) -> list[JobOutcome]:
        return [o for o in self.outcomes if not o.ok]

    @property
    def ok(self) -> list[JobOutcome]:
        return [o for o in self.outcomes if o.ok]

    @property
    def exit_code(self) -> int:
        return errors.EXIT_JOB_FAILED if self.failed else errors.EXIT_OK

    def summary(self) -> str:
        lines = [f"{len(self.ok)} ok, {len(self.failed)} failed"]
        lines.extend(f"  FAILED {o.job_id}: {o.error}" for o in self.failed)
        return "\n".join(lines)


def read_urls(urls: Sequence[str], files: Sequence[Path]) -> list[str]:
    """Collect URLs from arguments and batch files, preserving order."""
    collected = list(urls)
    for path in files:
        text = sys.stdin.read() if str(path) == "-" else path.read_text(encoding="utf-8")
        collected.extend(
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    seen: dict[str, None] = {}
    for url in collected:
        seen.setdefault(url, None)
    return list(seen)


def confirm_retention(
    config: RunConfig, *, stream: TextIO | None = None, isatty: bool | None = None
) -> None:
    """Ask once per run, before any job starts.

    A non-TTY refuses rather than auto-accepting: a piped or cron invocation
    would otherwise delete gigabytes with nobody ever seeing the warning.
    """
    if not config.retention_requested or config.assume_yes:
        return
    out = stream or sys.stderr
    interactive = isatty if isatty is not None else (sys.stdin.isatty() and out.isatty())
    if not interactive:
        raise errors.ConfigError(
            "--no-retain-* deletes downloaded media, and this is not a terminal, so "
            "the confirmation cannot be shown",
            hint="Re-run with --yes to confirm non-interactively, or drop the flag.",
        )
    out.write(
        "\nWARNING  source media will be deleted after each job completes:\n"
        f"    {config.retention_summary()}\n\n"
        "  Interrupting a run is safe: media is removed only once the next stage\n"
        "  is committed, so resume always works.\n\n"
        "  What you lose is re-run flexibility. Re-transcribing later with a\n"
        "  different --language / --model / --task will need another download.\n\n"
        "Continue? [y/N] "
    )
    out.flush()
    if sys.stdin.readline().strip().lower() not in ("y", "yes"):
        raise errors.ConfigError("aborted at the retention prompt", hint="Drop --no-retain-*.")


def expand(url: str, config: RunConfig) -> list[tuple[str, dict[str, Any]]]:
    """One (url, metadata) pair per video; playlists become many."""
    info = download_stage.extract_info(url, config.download)
    pairs: list[tuple[str, dict[str, Any]]] = []
    for entry in download_stage.entries_for(info):
        target = entry.get("webpage_url") or entry.get("url") or url
        pairs.append((str(target), entry))
    return pairs


def run_batch(urls: Sequence[str], config: RunConfig, *, dry_run: bool = False) -> BatchResult:
    result = BatchResult()
    workspace = paths.Workspace(config.workdir)
    holder = BackendHolder()
    try:
        for url in urls:
            _run_one(url, config, workspace, holder, result, dry_run=dry_run)
    finally:
        holder.close_all()
    return result


def _run_one(
    url: str,
    config: RunConfig,
    workspace: paths.Workspace,
    holder: BackendHolder,
    result: BatchResult,
    *,
    dry_run: bool,
) -> None:
    try:
        pairs = expand(url, config)
    except errors.FatalError:
        raise
    except errors.CcnError as exc:
        log.error("%s: %s", url, exc)
        result.outcomes.append(JobOutcome(url, url, (), ok=False, error=str(exc)))
        return

    for target, entry in pairs:
        job = workspace.job(download_stage.identify(entry, target))
        try:
            with store.job_lock(job), logsetup.job_log(job):
                result.outcomes.append(
                    run_job(job, target, config, entry=entry, holder=holder, dry_run=dry_run)
                )
        except errors.FatalError:
            raise
        except errors.CcnError as exc:
            # One bad job must never take the batch down with it.
            log.error("%s: %s", job.root.name, exc)
            result.outcomes.append(JobOutcome(job.root.name, target, (), ok=False, error=str(exc)))
