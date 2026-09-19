"""FLAC extraction.

Written to the job's tmp/ first and moved into place only once ffprobe confirms
it, so an interrupted extraction never leaves a plausible-looking stub behind.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from audio_scribe import errors
from audio_scribe.media import ffmpeg

if TYPE_CHECKING:
    from pathlib import Path

    from audio_scribe.paths import JobPaths

log = logging.getLogger(__name__)


def extract(
    job: JobPaths,
    video: Path,
    *,
    profile: str = "source",
    compression: int = 5,
    expected_s: float | None = None,
) -> Path:
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
    if expected_s and not ffmpeg.durations_match(duration, expected_s):
        # ffmpeg exits 0 on a truncated container, extracting the few frames it
        # found. Without this the pipeline transcribes near-silence and calls it
        # a success, which is worse than failing.
        staging.unlink(missing_ok=True)
        raise errors.DecodeError(
            f"extracted {duration:.1f}s of audio, which does not match the "
            f"{expected_s:.1f}s source",
            hint="The source media is probably truncated; re-run with --force-stage download.",
        )

    staging.replace(job.audio_file)
    log.info("extracted %.1fs of audio to %s", duration, job.audio_file.name)
    return job.audio_file
