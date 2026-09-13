"""The transcript data model, shared by backends and output writers."""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

# A repaired segment gets at least this long a span so subtitle renderers do not
# choke on a zero-length cue.
_MIN_SPAN_S = 0.05

RawChunk = tuple[float | None, float | None, str]


@dataclass(frozen=True, slots=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    segments: tuple[Segment, ...]
    language: str | None
    backend: str
    device: str
    model_id: str
    model_revision: str | None
    audio_s: float
    wall_s: float

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def rtf(self) -> float:
        """Realtime factor: seconds of audio processed per second of wall time."""
        if self.wall_s <= 0:
            return 0.0
        return self.audio_s / self.wall_s


def normalize_segments(raw: Sequence[RawChunk], audio_s: float) -> tuple[Segment, ...]:
    """Turn backend chunks into well-formed, in-bounds segments.

    Whisper emits ``-1`` (or nothing) for the end timestamp when the closing
    timestamp token is never predicted, which usually happens on the trailing
    chunk. Left alone that produces a malformed final subtitle.
    """
    _warn_if_non_monotonic(raw)

    segments: list[Segment] = []
    previous_end = 0.0
    for index, (start, end, text) in enumerate(raw):
        stripped = text.strip()
        if not stripped:
            continue
        begin = previous_end if start is None or start < 0 else start
        finish = _resolve_end(end, begin, index, raw, audio_s)
        segments.append(Segment(begin, finish, stripped))
        previous_end = finish
    return tuple(segments)


def _resolve_end(
    end: float | None, begin: float, index: int, raw: Sequence[RawChunk], audio_s: float
) -> float:
    if end is None or end < 0 or end <= begin:
        end = _next_start_after(begin, index, raw)
        if end is None:
            end = audio_s
    end = min(end, audio_s)
    if end <= begin:
        end = min(begin + _MIN_SPAN_S, audio_s)
    return end


def _next_start_after(begin: float, index: int, raw: Sequence[RawChunk]) -> float | None:
    for start, _, _ in raw[index + 1 :]:
        if start is not None and start > begin:
            return start
    return None


def _warn_if_non_monotonic(raw: Sequence[RawChunk]) -> None:
    starts = [s for s, _, _ in raw if s is not None]
    if any(b < a for a, b in itertools.pairwise(starts)):
        log.warning(
            "chunk start timestamps are not monotonic; they may be window-relative "
            "rather than absolute, which would misplace every cue past 0:30"
        )
