from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from audio_scribe import paths
from audio_scribe.jobs import state as job_state
from audio_scribe.jobs import verify
from audio_scribe.media import ffmpeg

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def job(tmp_path: Path) -> paths.JobPaths:
    j = paths.Workspace(tmp_path).job("youtube-abc")
    j.ensure()
    return j


def _present(path: Path, root: Path) -> job_state.Artifact:
    return job_state.Artifact(job_state.ArtifactStatus.PRESENT, path=str(path.relative_to(root)))


class TestVerifyVideo:
    def test_ok_for_a_real_file(self, job: paths.JobPaths, sine_wav: Path) -> None:
        dst = job.media_dir / "video.mkv"
        dst.write_bytes(sine_wav.read_bytes())
        assert verify.verify_video(job, _present(dst, job.root), None) is verify.Verdict.OK

    def test_deleted_by_policy_is_satisfied(self, job: paths.JobPaths) -> None:
        art = job_state.Artifact(job_state.ArtifactStatus.DELETED_BY_POLICY, path="media/video.mkv")
        assert verify.verify_video(job, art, None) is verify.Verdict.SATISFIED_ABSENT

    def test_deleted_by_policy_never_touches_the_filesystem(
        self, job: paths.JobPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # probe_duration raises on a missing path, so a policy-deleted video must
        # short-circuit before any probe or the whole feature degrades to
        # "re-download everything".
        def boom(_p: Path) -> float:
            raise AssertionError("must not probe a policy-deleted artifact")

        monkeypatch.setattr(ffmpeg, "probe_duration", boom)
        art = job_state.Artifact(job_state.ArtifactStatus.DELETED_BY_POLICY, path="media/video.mkv")
        assert verify.verify_video(job, art, None) is verify.Verdict.SATISFIED_ABSENT

    def test_missing_file_is_missing_not_an_error(self, job: paths.JobPaths) -> None:
        art = job_state.Artifact(job_state.ArtifactStatus.PRESENT, path="media/video.mkv")
        assert verify.verify_video(job, art, None) is verify.Verdict.MISSING

    def test_no_recorded_path_is_missing(self, job: paths.JobPaths) -> None:
        assert (
            verify.verify_video(job, job_state.Artifact(job_state.ArtifactStatus.MISSING), None)
            is verify.Verdict.MISSING
        )

    def test_zero_byte_file_is_missing(self, job: paths.JobPaths) -> None:
        dst = job.media_dir / "video.mkv"
        dst.touch()
        assert verify.verify_video(job, _present(dst, job.root), None) is verify.Verdict.MISSING

    def test_unreadable_file_is_corrupt(self, job: paths.JobPaths) -> None:
        dst = job.media_dir / "video.mkv"
        dst.write_bytes(b"not a video at all")
        assert verify.verify_video(job, _present(dst, job.root), None) is verify.Verdict.CORRUPT

    def test_part_with_control_file_is_partial(self, job: paths.JobPaths) -> None:
        (job.media_dir / "video.mkv.part").write_bytes(b"x" * 10)
        (job.media_dir / "video.mkv.part.aria2").write_bytes(b"ctl")
        art = job_state.Artifact(job_state.ArtifactStatus.MISSING)
        assert verify.verify_video(job, art, None) is verify.Verdict.PARTIAL

    def test_part_without_control_file_is_missing(self, job: paths.JobPaths) -> None:
        # aria2 writes segments out of order, so a .part with no control file is a
        # sparse file with holes, not a valid prefix -- resuming it corrupts output.
        (job.media_dir / "video.mkv.part").write_bytes(b"x" * 10)
        art = job_state.Artifact(job_state.ArtifactStatus.MISSING)
        assert verify.verify_video(job, art, None) is verify.Verdict.MISSING

    def test_part_in_tmp_is_also_found(self, job: paths.JobPaths) -> None:
        (job.tmp_dir / "video.mkv.part").write_bytes(b"x" * 10)
        (job.tmp_dir / "video.mkv.part.aria2").write_bytes(b"ctl")
        assert (
            verify.verify_video(job, job_state.Artifact(job_state.ArtifactStatus.MISSING), None)
            is verify.Verdict.PARTIAL
        )

    def test_falls_back_to_declared_size_when_duration_is_unknown(
        self, job: paths.JobPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dst = job.media_dir / "video.mkv"
        dst.write_bytes(b"x" * 5000)

        def no_duration(_p: Path) -> None:
            return None

        monkeypatch.setattr(ffmpeg, "probe_duration", no_duration)
        assert verify.verify_video(job, _present(dst, job.root), 5000) is verify.Verdict.OK

    def test_size_mismatch_without_duration_is_corrupt(
        self, job: paths.JobPaths, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dst = job.media_dir / "video.mkv"
        dst.write_bytes(b"x" * 10)

        def no_duration(_p: Path) -> None:
            return None

        monkeypatch.setattr(ffmpeg, "probe_duration", no_duration)
        assert verify.verify_video(job, _present(dst, job.root), 999_999) is verify.Verdict.CORRUPT

    def test_truncation_is_caught_by_size_even_though_it_probes(
        self, job: paths.JobPaths, sine_video: Path
    ) -> None:
        # Matroska writes the duration in its header, so ffprobe reports the full
        # length of a file that was cut off after a kilobyte. The recorded size is
        # the only evidence that the bytes are gone.
        dst = job.media_dir / "video.mkv"
        whole = sine_video.read_bytes()
        dst.write_bytes(whole[:1024])
        assert ffmpeg.probe_duration(dst) == pytest.approx(1.0, abs=0.2)
        assert (
            verify.verify_video(job, _present(dst, job.root), len(whole)) is verify.Verdict.CORRUPT
        )

    def test_a_whole_file_matching_its_recorded_size_is_ok(
        self, job: paths.JobPaths, sine_video: Path
    ) -> None:
        dst = job.media_dir / "video.mkv"
        dst.write_bytes(sine_video.read_bytes())
        assert (
            verify.verify_video(job, _present(dst, job.root), dst.stat().st_size)
            is verify.Verdict.OK
        )


class TestVerifyAudio:
    def _flac(self, job: paths.JobPaths, src: Path) -> job_state.Artifact:
        ffmpeg.extract_flac(src, job.audio_file)
        return _present(job.audio_file, job.root)

    def test_ok_for_a_real_flac(self, job: paths.JobPaths, sine_wav: Path) -> None:
        assert verify.verify_audio(job, self._flac(job, sine_wav), None) is verify.Verdict.OK

    def test_deleted_by_policy_is_satisfied(self, job: paths.JobPaths) -> None:
        art = job_state.Artifact(
            job_state.ArtifactStatus.DELETED_BY_POLICY, path="media/audio.flac"
        )
        assert verify.verify_audio(job, art, None) is verify.Verdict.SATISFIED_ABSENT

    def test_missing_is_missing(self, job: paths.JobPaths) -> None:
        art = job_state.Artifact(job_state.ArtifactStatus.PRESENT, path="media/audio.flac")
        assert verify.verify_audio(job, art, None) is verify.Verdict.MISSING

    def test_zero_byte_is_missing(self, job: paths.JobPaths) -> None:
        job.audio_file.touch()
        art = _present(job.audio_file, job.root)
        assert verify.verify_audio(job, art, None) is verify.Verdict.MISSING

    def test_matches_the_video_duration(self, job: paths.JobPaths, sine_wav: Path) -> None:
        assert verify.verify_audio(job, self._flac(job, sine_wav), 1.0) is verify.Verdict.OK

    def test_truncated_extraction_is_corrupt(self, job: paths.JobPaths, sine_wav: Path) -> None:
        # A one-second FLAC against a one-hour video means extraction was cut short.
        assert verify.verify_audio(job, self._flac(job, sine_wav), 3600.0) is verify.Verdict.CORRUPT

    def test_garbage_is_corrupt(self, job: paths.JobPaths) -> None:
        job.audio_file.write_bytes(b"nonsense")
        assert (
            verify.verify_audio(job, _present(job.audio_file, job.root), None)
            is verify.Verdict.CORRUPT
        )


class TestVerifyOutputs:
    def _write(self, job: paths.JobPaths, *names: str) -> job_state.Outputs:
        out: dict[str, str] = {}
        for n in names:
            p = job.transcript_file(n)
            p.write_text("content")
            out[n] = str(p.relative_to(job.root))
        return job_state.Outputs(job_state.ArtifactStatus.PRESENT, paths=out)

    def test_ok_when_every_wanted_format_exists(self, job: paths.JobPaths) -> None:
        got = self._write(job, "txt", "srt")
        assert verify.verify_outputs(job, got, ("txt", "srt")) is verify.Verdict.OK

    def test_missing_when_a_wanted_format_is_absent(self, job: paths.JobPaths) -> None:
        got = self._write(job, "txt")
        assert verify.verify_outputs(job, got, ("txt", "srt")) is verify.Verdict.MISSING

    def test_missing_when_a_recorded_file_was_deleted(self, job: paths.JobPaths) -> None:
        got = self._write(job, "txt")
        job.transcript_file("txt").unlink()
        assert verify.verify_outputs(job, got, ("txt",)) is verify.Verdict.MISSING

    def test_empty_file_is_missing(self, job: paths.JobPaths) -> None:
        got = self._write(job, "txt")
        job.transcript_file("txt").write_text("")
        assert verify.verify_outputs(job, got, ("txt",)) is verify.Verdict.MISSING

    def test_nothing_recorded_is_missing(self, job: paths.JobPaths) -> None:
        assert (
            verify.verify_outputs(
                job, job_state.Outputs(job_state.ArtifactStatus.MISSING), ("txt",)
            )
            is verify.Verdict.MISSING
        )

    def test_extra_formats_on_disk_do_not_matter(self, job: paths.JobPaths) -> None:
        got = self._write(job, "txt", "srt", "vtt")
        assert verify.verify_outputs(job, got, ("txt",)) is verify.Verdict.OK


def test_partial_scan_tolerates_missing_directories(tmp_path: Path) -> None:
    # A job dir that was never fully created must not raise during verification.
    bare = paths.Workspace(tmp_path).job("youtube-bare")
    bare.root.mkdir(parents=True)
    art = job_state.Artifact(job_state.ArtifactStatus.MISSING)
    assert verify.verify_video(bare, art, None) is verify.Verdict.MISSING
