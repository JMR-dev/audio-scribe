"""FLAC extraction.

Written to the job's tmp/ first and moved into place only once ffprobe confirms
it, so an interrupted extraction never leaves a plausible-looking stub behind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ccn_transcribe import errors
from ccn_transcribe.media import ffmpeg

if TYPE_CHECKING:
    from pathlib import Path

    from ccn_transcribe.paths import JobPaths

log = logging.getLogger(__name__)


def extract(job: JobPaths, video: Path, *, profile: str = "source", compression: int = 5) -> Path:
    job.ensure()
    staging = job.tmp_dir / "audio.flac"
    staging.unlink(missing_ok=True)
    ffmpeg.extract_flac(video, staging, profile=profile, compression=compression)

    duration = ffmpeg.probe_duration(staging)
    if duration is None or duration <= 0:
        staging.unlink(missing_ok=True)
        raise errors.DecodeError(
            "extracted FLAC has no readable duration",
            hint="The source audio track may be damaged.",
        )
    staging.replace(job.audio_file)
    log.info("extracted %.1fs of audio to %s", duration, job.audio_file.name)
    return job.audio_file
