from __future__ import annotations

import pytest

from ccn_transcribe import errors


def test_base_carries_message_and_hint() -> None:
    err = errors.CcnError("something broke", hint="try this")
    assert str(err) == "something broke"
    assert err.hint == "try this"
    assert err.exit_code == errors.EXIT_JOB_FAILED


def test_hint_defaults_to_none() -> None:
    assert errors.CcnError("bare").hint is None


@pytest.mark.parametrize(
    "cls",
    [errors.ConfigError, errors.PreflightError, errors.DiskFullError, errors.SchemaTooNewError],
)
def test_fatal_errors_abort_the_run(cls: type[errors.CcnError]) -> None:
    err = cls("nope")
    assert err.exit_code == errors.EXIT_FATAL
    assert isinstance(err, errors.FatalError)
    # A fatal error must never be mistaken for a per-job failure.
    assert not isinstance(err, errors.JobError)


def test_job_error_carries_job_id() -> None:
    err = errors.JobError("failed", job_id="youtube-abc123")
    assert err.job_id == "youtube-abc123"
    assert err.exit_code == errors.EXIT_JOB_FAILED


def test_job_error_job_id_optional() -> None:
    assert errors.JobError("failed").job_id is None


@pytest.mark.parametrize(
    "cls",
    [
        errors.GeoBlockedError,
        errors.PrivateVideoError,
        errors.AgeGatedError,
        errors.MembersOnlyError,
        errors.NotFoundError,
        errors.BotCheckError,
        errors.NetworkError,
    ],
)
def test_source_errors_are_job_errors(cls: type[errors.SourceError]) -> None:
    assert issubclass(cls, errors.SourceError)
    assert issubclass(cls, errors.JobError)


def test_only_network_errors_are_retryable() -> None:
    assert errors.NetworkError("flaky").retryable is True
    assert errors.PrivateVideoError("nope").retryable is False
    assert errors.NotFoundError("gone").retryable is False


def test_every_error_class_declares_a_distinct_code() -> None:
    classes = [
        v for v in vars(errors).values() if isinstance(v, type) and issubclass(v, errors.CcnError)
    ]
    codes = [c.code for c in classes]
    assert len(codes) == len(set(codes)), "duplicate .code values"


def test_aria2_error_carries_process_exit_code() -> None:
    err = errors.Aria2Error(
        "aria2c failed", aria2_exit_code=5, job_id="youtube-x", hint="slow link"
    )
    assert err.aria2_exit_code == 5
    assert err.job_id == "youtube-x"
    assert err.hint == "slow link"
    assert isinstance(err, errors.DownloadError)


def test_aria2_error_exit_code_optional() -> None:
    assert errors.Aria2Error("boom").aria2_exit_code is None
