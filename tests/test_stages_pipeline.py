from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import pytest

from ccn_transcribe import errors, paths
from ccn_transcribe.backends.base import ToolchainStatus, TranscribeRequest
from ccn_transcribe.media import ffmpeg
from ccn_transcribe.stages import audio as audio_stage
from ccn_transcribe.stages import outputs as outputs_stage
from ccn_transcribe.stages import transcribe as transcribe_stage
from ccn_transcribe.transcript import Segment, TranscriptResult

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

RESULT = TranscriptResult(
    segments=(Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world")),
    language="en",
    backend="fake",
    device="FAKE",
    model_id="m",
    model_revision=None,
    audio_s=2.0,
    wall_s=0.5,
)


@pytest.fixture
def job(tmp_path: Path) -> paths.JobPaths:
    j = paths.Workspace(tmp_path).job("youtube-abc")
    j.ensure()
    return j


class TestAudioStage:
    def test_produces_a_verified_flac(self, job: paths.JobPaths, sine_wav: Path) -> None:
        out = audio_stage.extract(job, sine_wav)
        assert out == job.audio_file
        assert ffmpeg.probe_duration(out) == pytest.approx(1.0, abs=0.05)

    def test_keeps_source_rate_by_default(self, job: paths.JobPaths, sine_wav: Path) -> None:
        audio_stage.extract(job, sine_wav)
        assert ffmpeg.probe_field(job.audio_file, "stream=sample_rate", "a:0") == "44100"

    def test_whisper_profile_downmixes(self, job: paths.JobPaths, sine_wav: Path) -> None:
        audio_stage.extract(job, sine_wav, profile="whisper")
        assert ffmpeg.probe_field(job.audio_file, "stream=sample_rate", "a:0") == "16000"

    def test_leaves_no_staging_file_behind(self, job: paths.JobPaths, sine_wav: Path) -> None:
        audio_stage.extract(job, sine_wav)
        assert list(job.tmp_dir.glob("*.flac")) == []

    def test_a_source_without_audio_is_rejected(
        self, job: paths.JobPaths, silent_video: Path
    ) -> None:
        with pytest.raises(errors.NoAudioStreamError):
            audio_stage.extract(job, silent_video)
        assert not job.audio_file.exists()

    def test_an_unverifiable_result_is_not_promoted(
        self, job: paths.JobPaths, sine_wav: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_duration(_p: Path) -> None:
            return None

        monkeypatch.setattr(ffmpeg, "probe_duration", no_duration)
        with pytest.raises(errors.DecodeError):
            audio_stage.extract(job, sine_wav)
        assert not job.audio_file.exists()

    def test_a_stale_staging_file_is_replaced(self, job: paths.JobPaths, sine_wav: Path) -> None:
        (job.tmp_dir / "audio.flac").write_bytes(b"junk from a killed run")
        assert audio_stage.extract(job, sine_wav).exists()


class FakeBackend:
    name: ClassVar[str] = "fake"
    device: ClassVar[str] = "FAKE"
    default_model: ClassVar[str] = "fake-model"

    def __init__(self, model_id: str | None = None, cache_dir: Path | None = None) -> None:
        self.model_id = model_id or self.default_model
        self.cache_dir = cache_dir

    @classmethod
    def probe_hardware(cls) -> bool:
        return True

    @classmethod
    def probe_toolchain(cls) -> ToolchainStatus:
        return ToolchainStatus(ok=True)

    def load(self) -> None:
        return

    def transcribe(
        self, pcm: npt.NDArray[np.float32], request: TranscribeRequest
    ) -> TranscriptResult:
        return TranscriptResult(
            segments=(Segment(0.0, 1.0, "fake"),),
            language=request.language,
            backend=self.name,
            device=self.device,
            model_id=self.model_id,
            model_revision=None,
            audio_s=len(pcm) / 16_000,
            wall_s=0.01,
        )

    def close(self) -> None:
        return


class TestTranscribeStage:
    def test_decodes_and_routes_to_a_backend(self, job: paths.JobPaths, sine_wav: Path) -> None:
        got = transcribe_stage.run(job, sine_wav, TranscribeRequest(), chain=(FakeBackend,))
        assert got.backend == "fake"
        assert got.audio_s == pytest.approx(1.0, abs=0.05)

    def test_a_bad_audio_file_surfaces_as_a_decode_error(
        self, job: paths.JobPaths, not_media: Path
    ) -> None:
        with pytest.raises(errors.DecodeError):
            transcribe_stage.run(job, not_media, TranscribeRequest(), chain=(FakeBackend,))


class TestSavedSegments:
    def test_none_when_there_is_no_transcript(self, job: paths.JobPaths) -> None:
        assert transcribe_stage.saved_segments(job) is None

    def test_reads_back_what_outputs_wrote(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("json",))
        assert transcribe_stage.saved_segments(job) == RESULT.segments

    def test_none_when_the_json_is_damaged(self, job: paths.JobPaths) -> None:
        job.transcript_file("json").write_text("{truncated")
        assert transcribe_stage.saved_segments(job) is None

    def test_none_when_segments_are_missing(self, job: paths.JobPaths) -> None:
        job.transcript_file("json").write_text('{"text": "none here"}')
        assert transcribe_stage.saved_segments(job) is None


class TestOutputsStage:
    def test_writes_every_requested_format(self, job: paths.JobPaths) -> None:
        written = outputs_stage.write(job, RESULT, ("txt", "srt", "vtt"))
        for name in ("txt", "srt", "vtt"):
            assert job.transcript_file(name).exists()
            assert written[name] == f"out/transcript.{name}"

    def test_always_writes_json_as_the_segment_cache(self, job: paths.JobPaths) -> None:
        # Without it, deleting one subtitle file would cost a full re-transcription.
        outputs_stage.write(job, RESULT, ("txt",))
        assert job.transcript_file("json").exists()

    def test_does_not_duplicate_json_when_requested(self) -> None:
        assert outputs_stage.wanted_with_cache(("json", "txt")).count("json") == 1

    def test_paths_are_job_relative(self, job: paths.JobPaths) -> None:
        assert outputs_stage.write(job, RESULT, ("txt",))["txt"].startswith("out/")

    def test_content_is_correct(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("txt", "srt"))
        assert job.transcript_file("txt").read_text().strip() == "hello world"
        assert "00:00:00,000 --> 00:00:01,000" in job.transcript_file("srt").read_text()

    def test_json_carries_run_metadata(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("json",), source={"id": "abc"})
        doc = json.loads(job.transcript_file("json").read_text())
        assert doc["device"] == "FAKE"
        assert doc["source"]["id"] == "abc"

    def test_every_file_ends_with_a_newline(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("txt", "srt", "vtt"))
        for name in ("txt", "srt", "vtt"):
            assert job.transcript_file(name).read_text().endswith("\n")

    def test_leaves_no_temp_files(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("txt",))
        assert list(job.out_dir.glob("*.tmp")) == []

    def test_rewriting_replaces_cleanly(self, job: paths.JobPaths) -> None:
        outputs_stage.write(job, RESULT, ("txt",))
        shorter = TranscriptResult(
            segments=(Segment(0.0, 1.0, "hi"),),
            language="en",
            backend="fake",
            device="FAKE",
            model_id="m",
            model_revision=None,
            audio_s=1.0,
            wall_s=0.1,
        )
        outputs_stage.write(job, shorter, ("txt",))
        assert job.transcript_file("txt").read_text().strip() == "hi"
