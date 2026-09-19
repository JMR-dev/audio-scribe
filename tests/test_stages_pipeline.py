from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

import pytest

from audio_scribe import errors, paths
from audio_scribe.backends.base import ToolchainStatus, TranscribeRequest
from audio_scribe.media import ffmpeg
from audio_scribe.stages import audio as audio_stage
from audio_scribe.stages import outputs as outputs_stage
from audio_scribe.stages import transcribe as transcribe_stage
from audio_scribe.transcript import Segment, TranscriptResult

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

    def test_rejects_an_extraction_shorter_than_its_source(
        self, job: paths.JobPaths, sine_wav: Path
    ) -> None:
        # A truncated video yields a fraction of a second of real audio while the
        # container still claims the full length; without this the pipeline
        # transcribes near-silence and reports success.
        with pytest.raises(errors.DecodeError, match="does not match"):
            audio_stage.extract(job, sine_wav, expected_s=212.861)
        assert not job.audio_file.exists()

    def test_accepts_an_extraction_matching_its_source(
        self, job: paths.JobPaths, sine_wav: Path
    ) -> None:
        assert audio_stage.extract(job, sine_wav, expected_s=1.0).exists()

    def test_no_expectation_means_no_comparison(self, job: paths.JobPaths, sine_wav: Path) -> None:
        assert audio_stage.extract(job, sine_wav, expected_s=None).exists()


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


class TestPublish:
    def test_does_nothing_without_an_out_dir(self, job: paths.JobPaths) -> None:
        written = outputs_stage.write(job, RESULT, ("txt",))
        assert outputs_stage.publish(job, written, ("txt",), out_dir=None, title="T") == []

    def test_copies_requested_formats_named_by_title(
        self, job: paths.JobPaths, tmp_path: Path
    ) -> None:
        written = outputs_stage.write(job, RESULT, ("txt", "srt"))
        collected = tmp_path / "collected"
        got = outputs_stage.publish(
            job, written, ("txt", "srt"), out_dir=collected, title="My Talk"
        )
        assert {p.name for p in got} == {
            "My_Talk-youtube-abc.txt",
            "My_Talk-youtube-abc.srt",
        }
        assert all(p.read_text() for p in got)

    def test_the_job_directory_is_left_canonical(self, job: paths.JobPaths, tmp_path: Path) -> None:
        # Resume depends on the job dir, so publishing copies rather than moves.
        written = outputs_stage.write(job, RESULT, ("txt",))
        outputs_stage.publish(job, written, ("txt",), out_dir=tmp_path / "c", title="T")
        assert job.transcript_file("txt").exists()

    def test_only_requested_formats_are_collected(
        self, job: paths.JobPaths, tmp_path: Path
    ) -> None:
        # json is always written as the segment cache, but need not be collected.
        written = outputs_stage.write(job, RESULT, ("txt",))
        collected = tmp_path / "collected"
        got = outputs_stage.publish(job, written, ("txt",), out_dir=collected, title="T")
        assert [p.suffix for p in got] == [".txt"]

    def test_a_missing_format_is_skipped(self, job: paths.JobPaths, tmp_path: Path) -> None:
        written = outputs_stage.write(job, RESULT, ("txt",))
        got = outputs_stage.publish(job, written, ("txt", "srt"), out_dir=tmp_path / "c", title="T")
        assert len(got) == 1

    def test_creates_the_directory(self, job: paths.JobPaths, tmp_path: Path) -> None:
        written = outputs_stage.write(job, RESULT, ("txt",))
        target = tmp_path / "deep" / "nested"
        outputs_stage.publish(job, written, ("txt",), out_dir=target, title="T")
        assert target.is_dir()


class TestOutputStem:
    def test_combines_title_and_id(self) -> None:
        assert paths.output_stem("My Talk", "youtube-abc") == "My_Talk-youtube-abc"

    def test_strips_path_separators(self) -> None:
        assert "/" not in paths.output_stem("a/b", "j")

    def test_collapses_runs_of_underscores(self) -> None:
        assert paths.output_stem("a   b", "j") == "a_b-j"

    def test_falls_back_to_the_id_when_there_is_no_title(self) -> None:
        assert paths.output_stem(None, "youtube-abc") == "youtube-abc"

    def test_falls_back_when_the_title_is_all_punctuation(self) -> None:
        assert paths.output_stem("///", "youtube-abc") == "youtube-abc"

    def test_truncates_a_very_long_title(self) -> None:
        assert len(paths.output_stem("z" * 300, "j")) < 120


def test_publish_with_nothing_to_copy_returns_empty(job: paths.JobPaths, tmp_path: Path) -> None:
    written = outputs_stage.write(job, RESULT, ("txt",))
    assert outputs_stage.publish(job, written, ("srt",), out_dir=tmp_path / "c", title="T") == []
