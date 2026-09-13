from __future__ import annotations

import pytest

from ccn_transcribe import errors
from ccn_transcribe.jobs import state as job_state


def make() -> job_state.JobState:
    return job_state.JobState.new(job_id="youtube-abc", source_url="https://y.test/watch?v=abc")


class TestNew:
    def test_starts_with_everything_missing(self) -> None:
        job = make()
        assert job.video.status is job_state.ArtifactStatus.MISSING
        assert job.audio.status is job_state.ArtifactStatus.MISSING
        assert job.outputs.status is job_state.ArtifactStatus.MISSING

    def test_records_identity_and_timestamps(self) -> None:
        job = make()
        assert job.job_id == "youtube-abc"
        assert job.source_url == "https://y.test/watch?v=abc"
        assert job.created_at and job.updated_at

    def test_carries_the_current_schema_version(self) -> None:
        assert make().schema_version == job_state.SCHEMA_VERSION


class TestRoundTrip:
    def test_survives_serialization(self) -> None:
        job = make()
        job.video = job_state.Artifact(
            job_state.ArtifactStatus.PRESENT, path="media/v.mkv", duration_s=12.5
        )
        job.title = "A title"
        assert job_state.JobState.from_dict(job.to_dict()) == job

    def test_enums_serialize_as_plain_strings(self) -> None:
        doc = make().to_dict()
        assert doc["artifacts"]["video"]["status"] == "missing"

    def test_outputs_paths_round_trip(self) -> None:
        job = make()
        job.outputs = job_state.Outputs(
            job_state.ArtifactStatus.PRESENT, paths={"srt": "out/transcript.srt"}
        )
        assert job_state.JobState.from_dict(job.to_dict()).outputs.paths == {
            "srt": "out/transcript.srt"
        }

    def test_transcription_record_round_trips(self) -> None:
        job = make()
        job.transcription = job_state.TranscriptionRecord(
            model_id="m",
            model_revision="r",
            backend="openvino",
            device="GPU",
            language="en",
            task="transcribe",
            num_beams=1,
        )
        restored = job_state.JobState.from_dict(job.to_dict())
        assert restored.transcription is not None
        assert restored.transcription.device == "GPU"

    def test_unknown_keys_are_ignored_not_fatal(self) -> None:
        doc = make().to_dict()
        doc["something_from_the_future"] = 1
        assert job_state.JobState.from_dict(doc).job_id == "youtube-abc"


class TestSchemaVersioning:
    def test_a_newer_schema_is_refused(self) -> None:
        doc = make().to_dict()
        doc["schema_version"] = job_state.SCHEMA_VERSION + 1
        # Guessing at a future layout risks destroying the user's state dir.
        with pytest.raises(errors.SchemaTooNewError):
            job_state.JobState.from_dict(doc)

    def test_the_current_schema_is_accepted(self) -> None:
        assert job_state.JobState.from_dict(make().to_dict()) is not None


class TestParamsChanged:
    def _record(self, **kw: object) -> job_state.TranscriptionRecord:
        base: dict[str, object] = {
            "model_id": "m",
            "model_revision": "r",
            "backend": "openvino",
            "device": "GPU",
            "language": "en",
            "task": "transcribe",
            "num_beams": 1,
        }
        base.update(kw)
        return job_state.TranscriptionRecord(**base)  # type: ignore[arg-type]

    def test_no_prior_run_counts_as_changed(self) -> None:
        assert make().params_changed(self._record()) is True

    def test_identical_params_are_unchanged(self) -> None:
        job = make()
        job.transcription = self._record()
        assert job.params_changed(self._record()) is False

    @pytest.mark.parametrize(
        ("field", "value"),
        [("model_id", "other"), ("language", "de"), ("task", "translate"), ("num_beams", 5)],
    )
    def test_a_different_request_forces_a_rerun(self, field: str, value: object) -> None:
        job = make()
        job.transcription = self._record()
        assert job.params_changed(self._record(**{field: value})) is True

    def test_device_and_backend_do_not_force_a_rerun(self) -> None:
        # Falling back from GPU to CPU must not invalidate a good transcript.
        job = make()
        job.transcription = self._record()
        assert job.params_changed(self._record(device="CPU", backend="x")) is False


class TestDefaultsFromSparseDocuments:
    def test_a_document_with_no_artifacts_key_gets_defaults(self) -> None:
        doc = make().to_dict()
        del doc["artifacts"]
        restored = job_state.JobState.from_dict(doc)
        assert restored.video.status is job_state.ArtifactStatus.MISSING
        assert not restored.outputs.paths

    def test_an_empty_artifact_entry_gets_defaults(self) -> None:
        doc = make().to_dict()
        doc["artifacts"] = {"video": None, "audio": {}, "outputs": None}
        restored = job_state.JobState.from_dict(doc)
        assert restored.video.status is job_state.ArtifactStatus.MISSING
        assert restored.audio.status is job_state.ArtifactStatus.MISSING
        assert restored.outputs.status is job_state.ArtifactStatus.MISSING
