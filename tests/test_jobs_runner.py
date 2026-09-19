from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

import pytest

from audio_scribe import errors, paths
from audio_scribe.backends import registry
from audio_scribe.backends.base import ToolchainStatus, TranscribeRequest
from audio_scribe.config import RunConfig
from audio_scribe.jobs import runner, store
from audio_scribe.jobs.plan import Stage, Verdicts
from audio_scribe.jobs.state import Artifact, ArtifactStatus, JobState
from audio_scribe.jobs.verify import Verdict
from audio_scribe.stages import download as download_stage
from audio_scribe.transcript import Segment, TranscriptResult

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


class FakeBackend:
    name: ClassVar[str] = "fake"
    device: ClassVar[str] = "FAKE"
    default_model: ClassVar[str] = "fake-model"

    def __init__(  # pylint: disable=unused-argument
        self, model_id: str | None = None, cache_dir: Path | None = None
    ) -> None:
        self.model_id = model_id or self.default_model

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
            segments=(Segment(0.0, 1.0, "hello"),),
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


@pytest.fixture
def workspace(tmp_path: Path) -> paths.Workspace:
    return paths.Workspace(tmp_path / "transcripts")


@pytest.fixture
def config(tmp_path: Path) -> RunConfig:
    return RunConfig(workdir=tmp_path / "transcripts", formats=("txt", "srt", "json"))


@pytest.fixture(autouse=True)
def _fake_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    registry.reset_demotions()
    monkeypatch.setattr(registry, "DEFAULT_CHAIN", (FakeBackend,))


@pytest.fixture
def seeded(
    workspace: paths.Workspace, sine_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> paths.JobPaths:
    """A job whose download stage produces a real media file."""
    job = workspace.job("youtube-abc")
    job.ensure()

    def fake_download(j: paths.JobPaths, _url: str, *_a: object, **_k: object) -> Path:
        target = j.media_dir / "video.mkv"
        target.write_bytes(sine_wav.read_bytes())
        return target

    monkeypatch.setattr(download_stage, "download", fake_download)
    return job


class TestHappyPath:
    def test_runs_every_stage_and_writes_outputs(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        outcome = runner.run_job(seeded, "https://y.test/x", config)
        assert outcome.ok is True
        assert outcome.stages == (Stage.DOWNLOAD, Stage.AUDIO, Stage.TRANSCRIBE, Stage.OUTPUTS)
        for name in ("txt", "srt", "json"):
            assert seeded.transcript_file(name).exists()

    def test_records_run_metadata(self, seeded: paths.JobPaths, config: RunConfig) -> None:
        runner.run_job(seeded, "https://y.test/x", config)
        state = store.load(seeded)
        assert state is not None
        assert state.transcription is not None
        assert state.transcription.device == "FAKE"
        assert state.video.status is ArtifactStatus.PRESENT
        assert state.audio.status is ArtifactStatus.PRESENT

    def test_a_second_run_does_nothing(self, seeded: paths.JobPaths, config: RunConfig) -> None:
        runner.run_job(seeded, "https://y.test/x", config)
        assert runner.run_job(seeded, "https://y.test/x", config).stages == ()

    def test_title_from_the_entry_reaches_the_json(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", config, entry={"title": "T", "duration": 1})
        doc = json.loads(seeded.transcript_file("json").read_text())
        assert doc["source"]["title"] == "T"

    def test_dry_run_changes_nothing(self, seeded: paths.JobPaths, config: RunConfig) -> None:
        outcome = runner.run_job(seeded, "https://y.test/x", config, dry_run=True)
        assert outcome.stages
        assert not seeded.transcript_file("txt").exists()


class TestResume:
    def test_deleting_a_subtitle_re_renders_without_re_transcribing(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", config)
        seeded.transcript_file("srt").unlink()
        outcome = runner.run_job(seeded, "https://y.test/x", config)
        assert outcome.stages == (Stage.OUTPUTS,)
        assert seeded.transcript_file("srt").exists()

    def test_deleting_the_audio_re_extracts_only(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", config)
        seeded.audio_file.unlink()
        seeded.transcript_file("json").unlink()
        outcome = runner.run_job(seeded, "https://y.test/x", config)
        assert Stage.AUDIO in outcome.stages
        assert Stage.DOWNLOAD not in outcome.stages

    def test_changing_the_language_forces_a_re_transcription(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", config)
        outcome = runner.run_job(seeded, "https://y.test/x", replace(config, language="de"))
        assert Stage.TRANSCRIBE in outcome.stages


class TestRetention:
    def test_video_is_deleted_only_after_the_flac_is_committed(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", replace(config, no_retain_video=True))
        assert download_stage.find_video(seeded) is None
        assert seeded.audio_file.exists()

    def test_the_deletion_is_recorded_as_policy_not_loss(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", replace(config, no_retain_video=True))
        state = store.load(seeded)
        assert state is not None
        assert state.video.status is ArtifactStatus.DELETED_BY_POLICY
        assert state.video.reason == "--no-retain-video"

    def test_a_policy_deletion_is_in_the_event_log(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", replace(config, no_retain_video=True))
        events = store.read_events(seeded)
        assert any(e["artifact"] == "video" for e in events)

    def test_re_running_after_a_policy_deletion_does_not_re_download(
        self, seeded: paths.JobPaths, config: RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", replace(config, no_retain_video=True))

        def forbidden(*_a: object, **_k: object) -> None:
            raise AssertionError("must not re-download an intentionally removed video")

        monkeypatch.setattr(download_stage, "download", forbidden)
        assert runner.run_job(seeded, "https://y.test/x", config).stages == ()

    def test_no_retain_audio_removes_the_flac_after_outputs(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(seeded, "https://y.test/x", replace(config, no_retain_audio=True))
        assert not seeded.audio_file.exists()
        assert seeded.transcript_file("txt").exists()

    def test_forcing_transcribe_walks_back_past_both_deletions(
        self, seeded: paths.JobPaths, config: RunConfig
    ) -> None:
        runner.run_job(
            seeded,
            "https://y.test/x",
            replace(config, no_retain_video=True, no_retain_audio=True),
        )
        outcome = runner.run_job(
            seeded,
            "https://y.test/x",
            replace(config, force=frozenset({Stage.TRANSCRIBE})),
        )
        assert outcome.stages == (
            Stage.DOWNLOAD,
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )


class TestFailures:
    def test_a_job_error_is_reported_not_raised(
        self, workspace: paths.Workspace, config: RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = workspace.job("youtube-bad")
        job.ensure()

        def boom(*_a: object, **_k: object) -> None:
            raise errors.PrivateVideoError("Private video", job_id="youtube-bad")

        monkeypatch.setattr(download_stage, "download", boom)
        outcome = runner.run_job(job, "https://y.test/x", config)
        assert outcome.ok is False
        assert "Private" in (outcome.error or "")

    def test_the_failure_is_recorded_in_state(
        self, workspace: paths.Workspace, config: RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = workspace.job("youtube-bad")
        job.ensure()

        def boom(*_a: object, **_k: object) -> None:
            raise errors.NotFoundError("gone")

        monkeypatch.setattr(download_stage, "download", boom)
        runner.run_job(job, "https://y.test/x", config)
        state = store.load(job)
        assert state is not None
        assert state.last_error is not None

    def test_a_fatal_error_propagates(
        self, workspace: paths.Workspace, config: RunConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Continuing a batch after ENOSPC only makes more corrupt artifacts.
        job = workspace.job("youtube-bad")
        job.ensure()

        def boom(*_a: object, **_k: object) -> None:
            raise errors.DiskFullError("no space")

        monkeypatch.setattr(download_stage, "download", boom)
        with pytest.raises(errors.DiskFullError):
            runner.run_job(job, "https://y.test/x", config)

    def test_forcing_audio_without_a_video_fails_clearly(
        self, workspace: paths.Workspace, config: RunConfig
    ) -> None:
        job = workspace.job("youtube-none")
        job.ensure()
        outcome = runner.run_job(
            job, "https://y.test/x", replace(config, force=frozenset({Stage.AUDIO}))
        )
        assert outcome.ok is False


class TestPlanDescription:
    def test_says_when_nothing_is_needed(self) -> None:
        verdicts = Verdicts(Verdict.OK, Verdict.OK, Verdict.OK)
        assert "already complete" in runner.describe_plan("j", (), verdicts)

    def test_explains_a_retention_driven_re_download(self) -> None:
        verdicts = Verdicts(Verdict.SATISFIED_ABSENT, Verdict.SATISFIED_ABSENT, Verdict.MISSING)
        text = runner.describe_plan("j", (Stage.DOWNLOAD, Stage.AUDIO), verdicts)
        assert "retention flag" in text

    def test_mentions_resuming_a_partial(self) -> None:
        verdicts = Verdicts(Verdict.PARTIAL, Verdict.MISSING, Verdict.MISSING)
        assert "partial" in runner.describe_plan("j", (Stage.DOWNLOAD,), verdicts)


class TestConfig:
    def test_retention_summary_names_both(self) -> None:
        cfg = RunConfig(no_retain_video=True, no_retain_audio=True)
        assert "video" in cfg.retention_summary()
        assert "audio" in cfg.retention_summary()

    def test_retention_requested_is_false_by_default(self) -> None:
        assert RunConfig().retention_requested is False

    def test_request_carries_transcription_parameters(self) -> None:
        request = RunConfig(language="en", num_beams=3).request()
        assert request.language == "en"
        assert request.num_beams == 3

    def test_with_download_replaces_nested_config(self) -> None:
        assert RunConfig().with_download(use_aria2c=True).download.use_aria2c is True


class TestRetentionSummary:
    def test_video_only(self) -> None:
        summary = RunConfig(no_retain_video=True).retention_summary()
        assert "video" in summary
        assert "audio" not in summary

    def test_audio_only(self) -> None:
        summary = RunConfig(no_retain_audio=True).retention_summary()
        assert "audio" in summary
        assert "video" not in summary

    def test_neither(self) -> None:
        assert RunConfig().retention_summary() == ""


def test_audio_stage_fails_clearly_when_the_media_file_is_unrecognized(
    workspace: paths.Workspace, config: RunConfig, sine_wav: Path
) -> None:
    # State says the video is present and it verifies, but it is not named
    # video.* so the stage cannot find it. Fail with a message, not a crash.
    job = workspace.job("youtube-odd")
    job.ensure()
    odd = job.media_dir / "movie.mkv"
    odd.write_bytes(sine_wav.read_bytes())
    state = JobState.new(job_id=job.root.name, source_url="u")
    state.video = Artifact(ArtifactStatus.PRESENT, path="media/movie.mkv", size=odd.stat().st_size)
    store.save(job, state)

    outcome = runner.run_job(job, "u", replace(config, force=frozenset({Stage.AUDIO})))
    assert outcome.ok is False
    assert "no video available" in (outcome.error or "")
