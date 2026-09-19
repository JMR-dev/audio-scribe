"""Run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from audio_scribe.jobs.plan import Stage
from audio_scribe.media.ytdlp_opts import DEFAULT_DOWNLOAD, DownloadConfig

if TYPE_CHECKING:
    from audio_scribe.backends.base import TranscribeRequest

DEFAULT_FORMATS = ("txt", "srt", "vtt", "json")


@dataclass(frozen=True, slots=True)
class RunConfig:
    workdir: Path = Path("transcripts")
    out_dir: Path | None = None
    formats: tuple[str, ...] = DEFAULT_FORMATS
    language: str | None = None
    task: str = "transcribe"
    num_beams: int = 1
    initial_prompt: str | None = None
    device: str | None = None
    model: str | None = None
    ov_cache_dir: Path | None = None
    audio_profile: str = "source"
    no_retain_video: bool = False
    no_retain_audio: bool = False
    assume_yes: bool = False
    force: frozenset[Stage] = field(default_factory=frozenset[Stage])
    download: DownloadConfig = DEFAULT_DOWNLOAD
    warmup: bool = True
    attempts: int = 3

    @property
    def retention_requested(self) -> bool:
        return self.no_retain_video or self.no_retain_audio

    def retention_summary(self) -> str:
        parts: list[str] = []
        if self.no_retain_video:
            parts.append("video after the FLAC is extracted and verified")
        if self.no_retain_audio:
            parts.append("audio after all transcript outputs are written")
        return "; ".join(parts)

    def request(self) -> TranscribeRequest:
        from audio_scribe.backends.base import (  # pylint: disable=import-outside-toplevel
            TranscribeRequest,
        )

        return TranscribeRequest(
            language=self.language,
            task=self.task,
            num_beams=self.num_beams,
            initial_prompt=self.initial_prompt,
        )

    def with_download(self, **changes: object) -> RunConfig:
        return replace(self, download=replace(self.download, **changes))  # type: ignore[arg-type]
