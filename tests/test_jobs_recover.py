from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ccn_transcribe import paths
from ccn_transcribe.jobs import recover, store
from ccn_transcribe.jobs.state import ArtifactStatus

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def job(tmp_path: Path) -> paths.JobPaths:
    j = paths.Workspace(tmp_path).job("youtube-abc")
    j.ensure()
    return j


def _transcript(job: paths.JobPaths, **over: object) -> None:
    doc: dict[str, object] = {
        "text": "hello",
        "language": "en",
        "backend": "openvino",
        "device": "GPU",
        "model": {"id": "OpenVINO/whisper", "revision": "abc123"},
        "task": "translate",
        "num_beams": 3,
        "audio_s": 212.8,
        "wall_s": 52.9,
        "rtf": 4.02,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
        "source": {"id": "youtube-abc", "title": "A Title", "url": "https://a.test/v"},
    }
    doc.update(over)
    job.transcript_file("json").write_text(json.dumps(doc), encoding="utf-8")


class TestRebuildFromArtifacts:
    def test_an_empty_job_recovers_an_empty_record(self, job: paths.JobPaths) -> None:
        state = recover.rebuild(job, job_id="youtube-abc", source_url="https://a.test/v")
        assert state.video.status is ArtifactStatus.MISSING
        assert state.audio.status is ArtifactStatus.MISSING
        assert state.transcription is None

    def test_a_downloaded_video_is_found(self, job: paths.JobPaths, sine_video: Path) -> None:
        # Without this a corrupt state.json re-downloads a file already on disk.
        (job.media_dir / "video.mkv").write_bytes(sine_video.read_bytes())
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.PRESENT
        assert state.video.path == "media/video.mkv"
        assert state.video.size == sine_video.stat().st_size
        assert state.video.duration_s == pytest.approx(1.0, abs=0.2)

    def test_an_unmerged_stream_is_not_mistaken_for_the_video(
        self, job: paths.JobPaths, sine_video: Path
    ) -> None:
        (job.media_dir / "video.f251.webm").write_bytes(sine_video.read_bytes())
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.MISSING

    def test_extracted_audio_is_found(self, job: paths.JobPaths, sine_wav: Path) -> None:
        from ccn_transcribe.media import ffmpeg

        ffmpeg.extract_flac(sine_wav, job.audio_file)
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.audio.status is ArtifactStatus.PRESENT
        assert state.audio.path == "media/audio.flac"
        assert state.audio.duration_s == pytest.approx(1.0, abs=0.2)

    def test_unreadable_media_recovers_without_a_duration(self, job: paths.JobPaths) -> None:
        (job.media_dir / "video.mkv").write_bytes(b"not a video")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.PRESENT
        assert state.video.duration_s is None

    def test_written_outputs_are_found(self, job: paths.JobPaths) -> None:
        for name in ("txt", "srt", "json"):
            job.transcript_file(name).write_text("x", encoding="utf-8")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.outputs.status is ArtifactStatus.PRESENT
        assert state.outputs.paths == {
            "json": "out/transcript.json",
            "srt": "out/transcript.srt",
            "txt": "out/transcript.txt",
        }

    def test_a_stray_temp_file_is_not_an_output(self, job: paths.JobPaths) -> None:
        job.transcript_file("txt").with_suffix(".txt.tmp").write_text("x", encoding="utf-8")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.outputs.paths == {}


class TestRestoreTranscription:
    def test_the_record_comes_back_from_the_transcript(self, job: paths.JobPaths) -> None:
        # Without it params_changed sees no record, and a completed job is
        # re-transcribed from scratch despite every segment being on disk.
        _transcript(job)
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.transcription is not None
        assert state.transcription.model_id == "OpenVINO/whisper"
        assert state.transcription.model_revision == "abc123"
        assert state.transcription.language == "en"
        assert state.transcription.task == "translate"
        assert state.transcription.num_beams == 3
        assert state.transcription.device == "GPU"
        assert state.transcription.segments == 1
        assert state.transcription.rtf == pytest.approx(4.02)

    def test_the_title_comes_back_for_out_dir_naming(self, job: paths.JobPaths) -> None:
        _transcript(job)
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.title == "A Title"
        assert state.duration_s == pytest.approx(212.8)

    def test_a_transcript_from_before_task_was_recorded(self, job: paths.JobPaths) -> None:
        # Older transcripts omit task/num_beams; defaulting them is the safe
        # direction, since a mismatch only costs a re-transcription.
        _transcript(job)
        doc = json.loads(job.transcript_file("json").read_text(encoding="utf-8"))
        del doc["task"], doc["num_beams"]
        job.transcript_file("json").write_text(json.dumps(doc), encoding="utf-8")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.transcription is not None
        assert state.transcription.task == "transcribe"
        assert state.transcription.num_beams == 1

    def test_an_unreadable_transcript_leaves_no_record(self, job: paths.JobPaths) -> None:
        job.transcript_file("json").write_text("{ truncated", encoding="utf-8")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.transcription is None

    def test_a_transcript_missing_its_model_block(self, job: paths.JobPaths) -> None:
        _transcript(job, model="not a mapping")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.transcription is None


def _deleted(job: paths.JobPaths, which: str, reason: str) -> None:
    store.append_event(job, {"event": "deleted", "artifact": which, "reason": reason})


class TestReplayEvents:
    def test_a_policy_deletion_is_recovered(self, job: paths.JobPaths) -> None:
        # The whole point of events.jsonl: without it this reads as data loss and
        # triggers a re-download of media the user asked us to delete.
        _deleted(job, "video", "--no-retain-video")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.DELETED_BY_POLICY
        assert state.video.reason == "--no-retain-video"
        assert state.video.deleted_at is not None

    def test_audio_deletions_too(self, job: paths.JobPaths) -> None:
        _deleted(job, "audio", "--no-retain-audio")
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.audio.status is ArtifactStatus.DELETED_BY_POLICY

    def test_a_superseded_deletion_does_not_hide_a_present_file(
        self, job: paths.JobPaths, sine_video: Path
    ) -> None:
        # Deleted once, downloaded again later. The event is history, not truth:
        # the file on disk wins.
        _deleted(job, "video", "--no-retain-video")
        (job.media_dir / "video.mkv").write_bytes(sine_video.read_bytes())
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.PRESENT

    def test_unrelated_events_are_ignored(self, job: paths.JobPaths) -> None:
        store.append_event(job, {"event": "started"})
        store.append_event(job, {"event": "deleted", "artifact": "outputs"})
        state = recover.rebuild(job, job_id="youtube-abc", source_url="u")
        assert state.video.status is ArtifactStatus.MISSING
        assert state.audio.status is ArtifactStatus.MISSING
