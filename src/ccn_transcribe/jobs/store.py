"""Durable job state: atomic writes, an append-only event log, and a job lock."""

from __future__ import annotations

import fcntl
import json
import logging
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ccn_transcribe import errors
from ccn_transcribe.jobs.state import JobState

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from ccn_transcribe.paths import JobPaths

log = logging.getLogger(__name__)


def _fsync_dir(path: os.PathLike[str]) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_json(path: Path, doc: Any) -> None:
    """Write one JSON document atomically.

    tmp -> fsync -> replace -> fsync(dir) so an interrupted write can never leave a
    truncated record where a good one used to be.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.flush()
        os.fsync(fh.fileno())
    try:
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def save(job: JobPaths, state: JobState) -> None:
    state.touch()
    write_json(job.state_file, state.to_dict())


def load(job: JobPaths) -> JobState | None:
    """Read state.json, or None when there is nothing usable to read."""
    if not job.state_file.exists():
        return None
    try:
        doc = json.loads(job.state_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Artifacts are the source of truth, so an unreadable record is survivable.
        log.warning("state.json for %s is unreadable; rebuilding from artifacts", job.root.name)
        return None
    return JobState.from_dict(doc)


def append_event(job: JobPaths, event: dict[str, Any]) -> None:
    """Append one durable event.

    This is the record of *why* an artifact is absent. Without it a damaged
    state.json would downgrade a policy deletion to "missing" and trigger a
    pointless re-download.
    """
    job.root.mkdir(parents=True, exist_ok=True)
    payload = {"at": datetime.now(UTC).isoformat(timespec="seconds"), **event}
    with job.events_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_events(job: JobPaths) -> list[dict[str, Any]]:
    if not job.events_file.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in job.events_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            # One torn line must not cost us the rest of the history.
            log.warning("skipping an unreadable line in %s", job.events_file.name)
    return events


@contextmanager
def job_lock(job: JobPaths) -> Generator[None]:
    """Hold an exclusive lock for the job's lifetime.

    flock rather than a PID file: the kernel drops it when the process dies, so
    there is no stale-lock case and no heuristic to get wrong.
    """
    job.root.mkdir(parents=True, exist_ok=True)
    handle = job.lock_file.open("w", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise errors.LockedError(
                f"job {job.root.name} is already running in another process",
                job_id=job.root.name,
                hint="Wait for it to finish, or use a different --workdir.",
            ) from exc
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        handle.close()
