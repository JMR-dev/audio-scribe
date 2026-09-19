"""The transcription stage: decode, route to a backend, and reuse saved segments."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from audio_scribe.backends import registry
from audio_scribe.media import ffmpeg
from audio_scribe.transcript import Segment

if TYPE_CHECKING:
    from pathlib import Path

    from audio_scribe.backends.base import Backend, TranscribeRequest
    from audio_scribe.backends.registry import BackendHolder
    from audio_scribe.paths import JobPaths
    from audio_scribe.transcript import TranscriptResult

log = logging.getLogger(__name__)


def run(
    job: JobPaths,
    audio: Path,
    request: TranscribeRequest,
    *,
    holder: BackendHolder | None = None,
    requested: str | None = None,
    model_id: str | None = None,
    cache_dir: Path | None = None,
    chain: tuple[type[Backend], ...] | None = None,
) -> TranscriptResult:
    job.ensure()
    pcm = ffmpeg.decode_16k_mono(audio)
    log.info("transcribing %.1fs of audio", len(pcm) / ffmpeg.SAMPLE_RATE)
    return registry.transcribe_with_fallback(
        pcm,
        request,
        holder=holder,
        requested=requested,
        model_id=model_id,
        cache_dir=cache_dir,
        chain=chain,
    )


def saved_segments(job: JobPaths) -> tuple[Segment, ...] | None:
    """Segments from a previous run, so a deleted subtitle file costs a re-render
    rather than re-transcribing a long recording."""
    path = job.transcript_file("json")
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        raw = doc["segments"]
        return tuple(Segment(s["start"], s["end"], s["text"]) for s in raw)
    except (OSError, ValueError, KeyError, TypeError):
        log.warning("%s is unreadable; a re-transcription will be needed", path.name)
        return None
