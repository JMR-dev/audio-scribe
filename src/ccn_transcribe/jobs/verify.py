"""Artifact verification.

A stage counts as complete only if its artifact is actually on disk and readable.
Trusting a recorded marker is how a truncated download or a deleted transcript
gets treated as done.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from ccn_transcribe import errors
from ccn_transcribe.jobs.state import Artifact, ArtifactStatus, Outputs
from ccn_transcribe.media import ffmpeg

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from ccn_transcribe.paths import JobPaths

log = logging.getLogger(__name__)

# A FLAC this far off the video's duration means extraction was cut short.
_DURATION_TOLERANCE = 0.005
_MIN_TOLERANCE_S = 1.0


class Verdict(StrEnum):
    OK = "ok"
    SATISFIED_ABSENT = "satisfied_absent"
    PARTIAL = "partial"
    CORRUPT = "corrupt"
    MISSING = "missing"


def _resolve(job: JobPaths, artifact: Artifact) -> Path | None:
    if artifact.path is None:
        return None
    return job.root / artifact.path


def _part_files(job: JobPaths) -> Iterator[Path]:
    for directory in (job.media_dir, job.tmp_dir):
        # glob yields nothing for a directory that does not exist; no guard needed.
        yield from directory.glob("*.part")


def _resumable_partial_exists(job: JobPaths) -> bool:
    """A .part is only resumable alongside its aria2 control file.

    aria2 writes segments out of order, so a .part without the control file is a
    sparse file with holes rather than a valid prefix; resuming from its length
    would silently produce a corrupt video.
    """
    return any(part.with_suffix(part.suffix + ".aria2").exists() for part in _part_files(job))


def _probe_duration(path: Path) -> float | None:
    try:
        return ffmpeg.probe_duration(path)
    except errors.ProbeError:
        return None


def verify_video(job: JobPaths, artifact: Artifact, declared_size: int | None) -> Verdict:
    if artifact.status is ArtifactStatus.DELETED_BY_POLICY:
        # Short-circuit before touching the filesystem: the obligation is
        # discharged, and probing a deliberately absent file would only error.
        return Verdict.SATISFIED_ABSENT

    path = _resolve(job, artifact)
    if path is None or not path.exists() or path.stat().st_size == 0:
        return Verdict.PARTIAL if _resumable_partial_exists(job) else Verdict.MISSING

    duration = _probe_duration(path)
    if duration is None:
        # Some mkv/webm muxes declare no duration; fall back to the expected size.
        if declared_size and abs(path.stat().st_size - declared_size) < 4096:
            return Verdict.OK
        return Verdict.CORRUPT
    return Verdict.OK if duration > 0 else Verdict.CORRUPT


def verify_audio(job: JobPaths, artifact: Artifact, video_duration: float | None) -> Verdict:
    if artifact.status is ArtifactStatus.DELETED_BY_POLICY:
        return Verdict.SATISFIED_ABSENT

    path = _resolve(job, artifact)
    if path is None or not path.exists() or path.stat().st_size == 0:
        return Verdict.MISSING

    duration = _probe_duration(path)
    if duration is None or duration <= 0:
        return Verdict.CORRUPT
    if video_duration:
        tolerance = max(_MIN_TOLERANCE_S, video_duration * _DURATION_TOLERANCE)
        if abs(duration - video_duration) > tolerance:
            return Verdict.CORRUPT
    return Verdict.OK


def verify_outputs(job: JobPaths, outputs: Outputs, wanted: Sequence[str]) -> Verdict:
    for name in wanted:
        relative = outputs.paths.get(name)
        if relative is None:
            return Verdict.MISSING
        path = job.root / relative
        if not path.exists() or path.stat().st_size == 0:
            return Verdict.MISSING
    return Verdict.OK
