"""Job identity and on-disk layout.

Job ids come from yt-dlp's extractor key plus video id rather than the URL, since
one video has many URL forms. ``index.json`` maps normalized URLs to job ids so
resuming never needs a network round trip.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")
_NOT_ALNUM = re.compile(r"[^a-z0-9]+")

_MAX_ID = 96
_TRUNCATE_TO = 64
_DIGEST_LEN = 10


def normalize_url(url: str) -> str:
    """Collapse URL spellings that name the same resource."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _digest(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()[:_DIGEST_LEN]


def job_id(extractor_key: str, video_id: str) -> str:
    """A filesystem-safe, collision-resistant directory name for one video."""
    extractor = _NOT_ALNUM.sub("", extractor_key.lower())[:24] or "generic"
    safe = _UNSAFE.sub("_", video_id)
    # Sanitizing is lossy, so two distinct ids can collapse to one name; a digest
    # keeps them in separate job directories.
    if safe != video_id or len(safe) > _MAX_ID:
        safe = f"{safe[:_TRUNCATE_TO]}-{_digest(video_id)}"
    return f"{extractor}-{safe}"


def url_job_id(url: str) -> str:
    """Fallback identity for sources yt-dlp cannot name (direct file URLs)."""
    return f"url-{hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16]}"


def output_stem(title: str | None, identifier: str) -> str:
    """A readable, filesystem-safe basename for a collected transcript."""
    cleaned = _UNSAFE.sub("_", (title or "").strip()).strip("_.")
    cleaned = re.sub(r"_{2,}", "_", cleaned)[:80].strip("_.")
    return f"{cleaned}-{identifier}" if cleaned else identifier


@dataclass(frozen=True, slots=True)
class JobPaths:
    root: Path

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def lock_file(self) -> Path:
        return self.root / ".lock"

    @property
    def events_file(self) -> Path:
        return self.root / "events.jsonl"

    @property
    def info_file(self) -> Path:
        return self.root / "info.json"

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def out_dir(self) -> Path:
        return self.root / "out"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def tmp_dir(self) -> Path:
        # Inside the job dir so os.replace stays within one filesystem.
        return self.root / "tmp"

    @property
    def audio_file(self) -> Path:
        return self.media_dir / "audio.flac"

    def transcript_file(self, extension: str) -> Path:
        return self.out_dir / f"transcript.{extension}"

    def ensure(self) -> None:
        for directory in (self.media_dir, self.out_dir, self.logs_dir, self.tmp_dir):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def index_file(self) -> Path:
        return self.root / "index.json"

    def job(self, identifier: str) -> JobPaths:
        return JobPaths(self.jobs_dir / identifier)
