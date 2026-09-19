"""The persisted job record.

Artifacts on disk are the source of truth; this record holds only what the
filesystem cannot answer -- above all *why* an artifact is absent, which is what
keeps a policy deletion from being mistaken for data loss.

There is deliberately no top-level "stage" field. Persisting one is how
"marked done but the file is gone" bugs happen; the resume point is computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from audio_scribe import errors

SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ArtifactStatus(StrEnum):
    PRESENT = "present"
    MISSING = "missing"
    DELETED_BY_POLICY = "deleted_by_policy"


@dataclass
class Artifact:
    status: ArtifactStatus = ArtifactStatus.MISSING
    path: str | None = None
    size: int | None = None
    duration_s: float | None = None
    reason: str | None = None
    deleted_at: str | None = None


@dataclass
class Outputs:
    status: ArtifactStatus = ArtifactStatus.MISSING
    paths: dict[str, str] = field(default_factory=dict[str, str])


@dataclass
class TranscriptionRecord:
    model_id: str | None
    model_revision: str | None
    backend: str
    device: str
    language: str | None
    task: str
    num_beams: int
    segments: int | None = None
    audio_s: float | None = None
    wall_s: float | None = None
    rtf: float | None = None
    started_at: str | None = None
    finished_at: str | None = None


# Changing any of these invalidates an existing transcript. Backend and device are
# excluded on purpose: falling back from GPU to CPU must not force a re-run.
_RERUN_FIELDS = ("language", "task", "num_beams")


@dataclass
class JobState:
    job_id: str
    source_url: str
    schema_version: int = SCHEMA_VERSION
    title: str | None = None
    duration_s: float | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    video: Artifact = field(default_factory=Artifact)
    audio: Artifact = field(default_factory=Artifact)
    outputs: Outputs = field(default_factory=Outputs)
    transcription: TranscriptionRecord | None = None
    attempts: dict[str, int] = field(default_factory=dict[str, int])
    last_error: dict[str, str] | None = None

    @classmethod
    def new(cls, *, job_id: str, source_url: str) -> JobState:
        return cls(job_id=job_id, source_url=source_url)

    def params_changed(self, wanted: TranscriptionRecord) -> bool:
        if self.transcription is None:
            return True
        # model_id of None means "whatever the backend defaults to", which matches
        # the model actually used; only an explicit pin can disagree.
        if wanted.model_id is not None and wanted.model_id != self.transcription.model_id:
            return True
        return any(
            getattr(self.transcription, name) != getattr(wanted, name) for name in _RERUN_FIELDS
        )

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "source_url": self.source_url,
            "title": self.title,
            "duration_s": self.duration_s,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "artifacts": {
                "video": asdict(self.video),
                "audio": asdict(self.audio),
                "outputs": asdict(self.outputs),
            },
            "transcription": asdict(self.transcription) if self.transcription else None,
            "attempts": dict(self.attempts),
            "last_error": self.last_error,
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> JobState:
        version = int(doc.get("schema_version", SCHEMA_VERSION))
        if version > SCHEMA_VERSION:
            raise errors.SchemaTooNewError(
                f"state.json is schema v{version}, but this build understands v{SCHEMA_VERSION}",
                hint="Upgrade audio-scribe, or point --workdir somewhere else.",
            )
        artifacts = doc.get("artifacts", {})
        record = doc.get("transcription")
        return cls(
            job_id=doc["job_id"],
            source_url=doc["source_url"],
            schema_version=SCHEMA_VERSION,
            title=doc.get("title"),
            duration_s=doc.get("duration_s"),
            created_at=doc.get("created_at", _now()),
            updated_at=doc.get("updated_at", _now()),
            video=_artifact(artifacts.get("video")),
            audio=_artifact(artifacts.get("audio")),
            outputs=_outputs(artifacts.get("outputs")),
            transcription=TranscriptionRecord(**record) if record else None,
            attempts=dict(doc.get("attempts", {})),
            last_error=doc.get("last_error"),
        )


def _artifact(raw: dict[str, Any] | None) -> Artifact:
    if not raw:
        return Artifact()
    return Artifact(
        status=ArtifactStatus(raw.get("status", ArtifactStatus.MISSING)),
        path=raw.get("path"),
        size=raw.get("size"),
        duration_s=raw.get("duration_s"),
        reason=raw.get("reason"),
        deleted_at=raw.get("deleted_at"),
    )


def _outputs(raw: dict[str, Any] | None) -> Outputs:
    if not raw:
        return Outputs()
    return Outputs(
        status=ArtifactStatus(raw.get("status", ArtifactStatus.MISSING)),
        paths=dict(raw.get("paths", {})),
    )
