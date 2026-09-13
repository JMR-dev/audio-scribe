"""Transcript output writers: plain text, SRT, VTT and JSON.

SRT and VTT differ only in a separator and a header, so both live here where the
divergence is visible on one screen.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ccn_transcribe.transcript import Segment, TranscriptResult

FORMATS = ("txt", "srt", "vtt", "json")


def _timestamp(seconds: float, separator: str) -> str:
    # Convert to integer milliseconds before splitting: formatting the seconds
    # field directly renders 3599.9996 as "00:59:60.000", which is invalid.
    total_ms = max(0, round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3_600_000)
    minutes, total_ms = divmod(total_ms, 60_000)
    secs, millis = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def srt_timestamp(seconds: float) -> str:
    return _timestamp(seconds, ",")


def vtt_timestamp(seconds: float) -> str:
    return _timestamp(seconds, ".")


def to_txt(result: TranscriptResult) -> str:
    return result.text


def to_srt(segments: Sequence[Segment]) -> str:
    blocks = [
        f"{index}\n{srt_timestamp(seg.start)} --> {srt_timestamp(seg.end)}\n{seg.text}\n"
        for index, seg in enumerate(segments, start=1)
    ]
    return "\n".join(blocks)


def to_vtt(segments: Sequence[Segment]) -> str:
    blocks = [
        f"{vtt_timestamp(seg.start)} --> {vtt_timestamp(seg.end)}\n{seg.text}\n" for seg in segments
    ]
    return "WEBVTT\n" + "".join("\n" + b for b in blocks)


def to_json(result: TranscriptResult, source: Mapping[str, object] | None = None) -> str:
    doc: dict[str, object] = {
        "text": result.text,
        "language": result.language,
        "backend": result.backend,
        "device": result.device,
        "model": {"id": result.model_id, "revision": result.model_revision},
        "task": result.task,
        "num_beams": result.num_beams,
        "audio_s": result.audio_s,
        "wall_s": result.wall_s,
        "rtf": result.rtf,
        "segments": [{"start": s.start, "end": s.end, "text": s.text} for s in result.segments],
    }
    if source is not None:
        doc["source"] = dict(source)
    return json.dumps(doc, indent=2, ensure_ascii=False)


def render(
    result: TranscriptResult,
    wanted: Iterable[str],
    source: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Render the requested formats, keyed by extension."""
    out: dict[str, str] = {}
    for name in wanted:
        if name == "txt":
            out[name] = to_txt(result)
        elif name == "srt":
            out[name] = to_srt(result.segments)
        elif name == "vtt":
            out[name] = to_vtt(result.segments)
        elif name == "json":
            out[name] = to_json(result, source)
        else:
            msg = f"unknown output format {name!r}; expected one of {', '.join(FORMATS)}"
            raise ValueError(msg)
    return out
