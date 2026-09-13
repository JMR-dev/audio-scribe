from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import pytest

from ccn_transcribe import errors, paths
from ccn_transcribe.jobs import state as job_state
from ccn_transcribe.jobs import store

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def job(tmp_path: Path) -> paths.JobPaths:
    j = paths.Workspace(tmp_path).job("youtube-abc")
    j.ensure()
    return j


def _state() -> job_state.JobState:
    return job_state.JobState.new(job_id="youtube-abc", source_url="https://y.test/x")


class TestSaveLoad:
    def test_round_trip(self, job: paths.JobPaths) -> None:
        want = _state()
        want.title = "Hello"
        store.save(job, want)
        assert store.load(job).title == "Hello"  # type: ignore[union-attr]

    def test_load_returns_none_when_absent(self, job: paths.JobPaths) -> None:
        assert store.load(job) is None

    def test_save_leaves_no_temp_file(self, job: paths.JobPaths) -> None:
        store.save(job, _state())
        assert list(job.root.glob("*.tmp")) == []

    def test_save_writes_readable_json(self, job: paths.JobPaths) -> None:
        store.save(job, _state())
        assert json.loads(job.state_file.read_text())["job_id"] == "youtube-abc"

    def test_save_bumps_updated_at(self, job: paths.JobPaths) -> None:
        st = _state()
        st.updated_at = "1999-01-01T00:00:00+00:00"
        store.save(job, st)
        assert store.load(job).updated_at != "1999-01-01T00:00:00+00:00"  # type: ignore[union-attr]

    def test_a_failed_save_leaves_the_previous_state_intact(
        self, job: paths.JobPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The durability claim: a crash mid-save must never truncate good state.
        first = _state()
        first.title = "original"
        store.save(job, first)

        def boom(*_a: object, **_k: object) -> None:
            raise OSError("disk gave up")

        monkeypatch.setattr(os, "replace", boom)
        second = _state()
        second.title = "replacement"
        with pytest.raises(OSError, match="disk gave up"):
            store.save(job, second)

        assert store.load(job).title == "original"  # type: ignore[union-attr]

    def test_corrupt_state_is_treated_as_absent(
        self, job: paths.JobPaths, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Artifacts are the source of truth, so an unreadable record is recoverable.
        job.state_file.write_text("{not json at all")
        assert store.load(job) is None
        assert "state.json" in caplog.text

    def test_a_newer_schema_still_raises(self, job: paths.JobPaths) -> None:
        doc = _state().to_dict()
        doc["schema_version"] = job_state.SCHEMA_VERSION + 1
        job.state_file.write_text(json.dumps(doc))
        with pytest.raises(errors.SchemaTooNewError):
            store.load(job)


class TestEvents:
    def test_append_and_read(self, job: paths.JobPaths) -> None:
        store.append_event(job, {"event": "deleted", "artifact": "video"})
        [got] = store.read_events(job)
        assert got["event"] == "deleted"
        assert got["artifact"] == "video"

    def test_events_accumulate(self, job: paths.JobPaths) -> None:
        store.append_event(job, {"event": "a"})
        store.append_event(job, {"event": "b"})
        assert [e["event"] for e in store.read_events(job)] == ["a", "b"]

    def test_reading_with_no_file_is_empty(self, job: paths.JobPaths) -> None:
        assert store.read_events(job) == []

    def test_each_event_gets_a_timestamp(self, job: paths.JobPaths) -> None:
        store.append_event(job, {"event": "a"})
        assert "at" in store.read_events(job)[0]

    def test_a_damaged_line_does_not_lose_the_rest(self, job: paths.JobPaths) -> None:
        store.append_event(job, {"event": "good"})
        with job.events_file.open("a") as fh:
            fh.write("{truncated\n")
        store.append_event(job, {"event": "also-good"})
        assert [e["event"] for e in store.read_events(job)] == ["good", "also-good"]


class TestLock:
    def test_lock_is_acquired_and_released(self, job: paths.JobPaths) -> None:
        with store.job_lock(job):
            assert job.lock_file.exists()
        with store.job_lock(job):
            pass

    def test_a_second_holder_is_refused(self, job: paths.JobPaths) -> None:
        with store.job_lock(job), pytest.raises(errors.LockedError), store.job_lock(job):
            pass

    def test_the_error_names_the_job(self, job: paths.JobPaths) -> None:
        with store.job_lock(job):
            try:
                with store.job_lock(job):
                    pass
            except errors.LockedError as exc:
                assert "youtube-abc" in str(exc)

    def test_lock_released_even_if_the_body_raises(self, job: paths.JobPaths) -> None:
        with pytest.raises(ValueError, match="boom"), store.job_lock(job):
            raise ValueError("boom")
        with store.job_lock(job):
            pass


def test_blank_lines_in_the_event_log_are_skipped(job: paths.JobPaths) -> None:
    store.append_event(job, {"event": "a"})
    with job.events_file.open("a") as fh:
        fh.write("\n   \n")
    assert [e["event"] for e in store.read_events(job)] == ["a"]
