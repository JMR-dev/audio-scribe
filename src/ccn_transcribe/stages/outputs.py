"""Transcript output writing, atomically and in the requested formats."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from ccn_transcribe import formats

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ccn_transcribe.paths import JobPaths
    from ccn_transcribe.transcript import TranscriptResult

log = logging.getLogger(__name__)

# transcript.json doubles as the segment cache that makes a re-render possible,
# so it is always written even when the user did not ask for it.
ALWAYS = ("json",)


def wanted_with_cache(wanted: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys([*wanted, *ALWAYS]))


def write_atomic(path: os.PathLike[str], text: str) -> None:
    from pathlib import Path  # pylint: disable=import-outside-toplevel

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)


def write(
    job: JobPaths,
    result: TranscriptResult,
    wanted: Sequence[str],
    source: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Write every requested format; returns job-relative paths by extension."""
    job.ensure()
    rendered = formats.render(result, wanted_with_cache(wanted), source)
    written: dict[str, str] = {}
    for name, text in rendered.items():
        path = job.transcript_file(name)
        write_atomic(path, text if text.endswith("\n") else text + "\n")
        written[name] = str(path.relative_to(job.root))
    log.info("wrote %s", ", ".join(sorted(written)))
    return written
