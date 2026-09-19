"""The workspace index: which job directories a URL last expanded to.

Turning a URL into a job id normally costs a metadata request. This records what
the last expansion produced, so re-running a single video needs no network at
all, and so a batch whose source is unreachable can still be resumed from the
jobs already on disk. It is only ever a cache: anything unreadable, or written by
a version we do not understand, costs a metadata request rather than a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from audio_scribe import paths
from audio_scribe.jobs import store

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Target:
    """One video to run: which job directory it belongs to and where it lives."""

    job_id: str
    url: str
    # Metadata from a live expansion. Never persisted -- state.json already holds
    # the title and duration, and a stale copy here would quietly outrank it.
    entry: dict[str, Any] | None = field(default=None, compare=False)


def _mapping(value: object) -> Mapping[str, object]:
    """A JSON object as a typed mapping; anything else reads as empty.

    json.loads is untyped by nature, so the shape is checked once here rather
    than assumed at every lookup. Keys are strings by construction.
    """
    return cast("Mapping[str, object]", value) if isinstance(value, dict) else {}


def _urls(workspace: paths.Workspace) -> Mapping[str, object]:
    path = workspace.index_file
    if not path.exists():
        return {}
    try:
        doc = _mapping(store.read_json(path))
    except (OSError, ValueError):
        log.warning("%s is unreadable; it will be rebuilt", path.name)
        return {}
    if doc.get("schema_version") != SCHEMA_VERSION:
        return {}
    return _mapping(doc.get("urls"))


def lookup(workspace: paths.Workspace, url: str, *, playlist: bool) -> list[Target] | None:
    """The jobs this URL expanded to last time, or None if that is not known."""
    entry = _mapping(_urls(workspace).get(paths.normalize_url(url)))
    # The same URL names a different set with and without --playlist, so an entry
    # recorded under one must never answer for the other.
    if not entry or bool(entry.get("playlist")) is not playlist:
        return None
    rows = entry.get("jobs")
    if not isinstance(rows, list):
        return None
    targets: list[Target] = []
    for raw in cast("list[object]", rows):
        row = _mapping(raw)
        job_id, at = row.get("id"), row.get("url")
        if isinstance(job_id, str) and isinstance(at, str) and job_id and at:
            targets.append(Target(job_id=job_id, url=at))
    return targets or None


def record(
    workspace: paths.Workspace, url: str, *, playlist: bool, targets: Sequence[Target]
) -> None:
    urls = dict(_urls(workspace))
    urls[paths.normalize_url(url)] = {
        "playlist": playlist,
        "jobs": [{"id": t.job_id, "url": t.url} for t in targets],
    }
    store.write_json(workspace.index_file, {"schema_version": SCHEMA_VERSION, "urls": urls})
