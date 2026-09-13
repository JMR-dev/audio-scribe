"""Exception hierarchy for the pipeline.

Every error carries an actionable ``hint``; the CLI prints ``hint`` instead of a
traceback. The split that matters is ``FatalError`` (abort the whole run) versus
``JobError`` (record it and move to the next URL).
"""

from __future__ import annotations

from typing import ClassVar

EXIT_OK = 0
EXIT_JOB_FAILED = 1
EXIT_FATAL = 2
EXIT_INTERRUPTED = 130


class CcnError(Exception):
    code: ClassVar[str] = "ccn_error"
    exit_code: ClassVar[int] = EXIT_JOB_FAILED
    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        return self.message


class FatalError(CcnError):
    """Aborts the run. Continuing would only produce more bad artifacts."""

    code: ClassVar[str] = "fatal"
    exit_code: ClassVar[int] = EXIT_FATAL


class ConfigError(FatalError):
    code: ClassVar[str] = "config"


class PreflightError(FatalError):
    code: ClassVar[str] = "preflight"


class DiskFullError(FatalError):
    code: ClassVar[str] = "disk_full"


class SchemaTooNewError(FatalError):
    code: ClassVar[str] = "schema_too_new"


class JobError(CcnError):
    """Scoped to one job; the batch continues."""

    code: ClassVar[str] = "job"

    def __init__(self, message: str, *, job_id: str | None = None, hint: str | None = None) -> None:
        super().__init__(message, hint=hint)
        self.job_id = job_id


class LockedError(JobError):
    code: ClassVar[str] = "locked"


class SourceError(JobError):
    """The video itself cannot be obtained."""

    code: ClassVar[str] = "source"


class GeoBlockedError(SourceError):
    code: ClassVar[str] = "geo_blocked"


class PrivateVideoError(SourceError):
    code: ClassVar[str] = "private_video"


class AgeGatedError(SourceError):
    code: ClassVar[str] = "age_gated"


class MembersOnlyError(SourceError):
    code: ClassVar[str] = "members_only"


class NotFoundError(SourceError):
    code: ClassVar[str] = "not_found"


class BotCheckError(SourceError):
    code: ClassVar[str] = "bot_check"


class NetworkError(SourceError):
    code: ClassVar[str] = "network"
    retryable: ClassVar[bool] = True


class DownloadError(JobError):
    code: ClassVar[str] = "download"


class Aria2Error(DownloadError):
    code: ClassVar[str] = "aria2"

    def __init__(
        self,
        message: str,
        *,
        aria2_exit_code: int | None = None,
        job_id: str | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message, job_id=job_id, hint=hint)
        self.aria2_exit_code = aria2_exit_code


class IncompleteDownloadError(DownloadError):
    code: ClassVar[str] = "incomplete_download"


class MediaError(JobError):
    code: ClassVar[str] = "media"


class NoAudioStreamError(MediaError):
    code: ClassVar[str] = "no_audio_stream"


class DecodeError(MediaError):
    code: ClassVar[str] = "decode"


class ProbeError(MediaError):
    code: ClassVar[str] = "probe"


class TranscribeError(JobError):
    code: ClassVar[str] = "transcribe"


class BackendUnavailableError(TranscribeError):
    code: ClassVar[str] = "backend_unavailable"


class BackendRuntimeError(TranscribeError):
    code: ClassVar[str] = "backend_runtime"


class BackendNotImplementedError(TranscribeError):
    code: ClassVar[str] = "backend_not_implemented"


class ModelUnavailableError(TranscribeError):
    code: ClassVar[str] = "model_unavailable"


class OutOfMemoryError(TranscribeError):
    code: ClassVar[str] = "out_of_memory"
