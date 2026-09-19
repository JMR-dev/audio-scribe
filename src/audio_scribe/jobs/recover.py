"""Rebuilding a job record when state.json is gone or unreadable.

Artifacts are the source of truth, so most of the record can be read back off
the filesystem. Two things the filesystem cannot answer are recovered from the
files that do record them: *why* an artifact is absent comes from events.jsonl,
and how the transcript was produced comes from transcript.json. Without this a
damaged state.json costs a full re-download and re-transcription of work that is
sitting on disk.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from audio_scribe import errors, formats
from audio_scribe.jobs import store
from audio_scribe.jobs.state import Artifact, ArtifactStatus, JobState, Outputs
from audio_scribe.jobs.state import TranscriptionRecord as Record
from audio_scribe.media import ffmpeg
from audio_scribe.stages import download as download_stage

if TYPE_CHECKING:
    from pathlib import Path

    from audio_scribe.paths import JobPaths

log = logging.getLogger(__name__)


def rebuild(job: JobPaths, *, job_id: str, source_url: str) -> JobState:
    """Reconstruct as much of the record as the job directory can support."""
    state = JobState.new(job_id=job_id, source_url=source_url)
    _scan_media(job, state)
    _scan_outputs(job, state)
    _restore_transcription(job, state)
    _replay_events(job, state)
    return state


def _safe_duration(path: Path) -> float | None:
    try:
        return ffmpeg.probe_duration(path)
    except errors.ProbeError:
        return None


def _found(job: JobPaths, path: Path) -> Artifact:
    return Artifact(
        status=ArtifactStatus.PRESENT,
        path=str(path.relative_to(job.root)),
        size=path.stat().st_size,
        duration_s=_safe_duration(path),
    )


def _scan_media(job: JobPaths, state: JobState) -> None:
    video = download_stage.find_video(job)
    if video is not None:
        state.video = _found(job, video)
        state.duration_s = state.video.duration_s
    if job.audio_file.exists():
        state.audio = _found(job, job.audio_file)


def _scan_outputs(job: JobPaths, state: JobState) -> None:
    written = {
        path.suffix.lstrip("."): str(path.relative_to(job.root))
        for path in sorted(job.out_dir.glob("transcript.*"))
        if path.suffix.lstrip(".") in formats.FORMATS
    }
    if written:
        state.outputs = Outputs(status=ArtifactStatus.PRESENT, paths=written)


def _restore_transcription(job: JobPaths, state: JobState) -> None:
    path = job.transcript_file("json")
    if not path.exists():
        return
    try:
        doc: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        model: dict[str, Any] = doc["model"]
        state.transcription = Record(
            model_id=model["id"],
            model_revision=model.get("revision"),
            backend=doc["backend"],
            device=doc["device"],
            language=doc.get("language"),
            # Absent in transcripts written before these were recorded. Defaulting
            # can only disagree with what the user asks for next, and a disagreement
            # costs a re-transcription rather than a wrong answer.
            task=doc.get("task", "transcribe"),
            num_beams=doc.get("num_beams", 1),
            segments=len(doc.get("segments", ())),
            audio_s=doc.get("audio_s"),
            wall_s=doc.get("wall_s"),
            rtf=doc.get("rtf"),
        )
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("%s is unreadable; the transcription record cannot be recovered", path.name)
        return

    state.duration_s = state.duration_s or state.transcription.audio_s
    source: dict[str, Any] = doc.get("source") or {}
    state.title = source.get("title") or state.title


def _replay_events(job: JobPaths, state: JobState) -> None:
    for event in store.read_events(job):
        if event.get("event") != "deleted":
            continue
        artifact = {"video": state.video, "audio": state.audio}.get(str(event.get("artifact")))
        # A deletion explains an absence. If the file is back, it was re-fetched
        # since, and the event is history rather than the current truth.
        if artifact is None or artifact.status is ArtifactStatus.PRESENT:
            continue
        artifact.status = ArtifactStatus.DELETED_BY_POLICY
        artifact.reason = event.get("reason")
        artifact.deleted_at = event.get("at")
