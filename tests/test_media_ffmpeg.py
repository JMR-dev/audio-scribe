from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest

from audio_scribe import errors
from audio_scribe.media import ffmpeg

if TYPE_CHECKING:
    from pathlib import Path


class TestProbeDuration:
    def test_reads_duration_of_a_real_file(self, sine_wav: Path) -> None:
        assert ffmpeg.probe_duration(sine_wav) == pytest.approx(1.0, abs=0.05)

    def test_raises_on_a_non_media_file(self, not_media: Path) -> None:
        with pytest.raises(errors.ProbeError):
            ffmpeg.probe_duration(not_media)

    def test_raises_on_a_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(errors.ProbeError):
            ffmpeg.probe_duration(tmp_path / "nope.wav")

    def test_returns_none_when_the_container_declares_no_duration(
        self, sine_wav: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Some webm/mkv muxes report format.duration=N/A with no stream duration.
        def na(*_a: object, **_k: object) -> str:
            return "N/A"

        monkeypatch.setattr(ffmpeg, "probe_field", na)
        assert ffmpeg.probe_duration(sine_wav) is None


class TestHasAudioStream:
    def test_true_for_audio(self, sine_wav: Path) -> None:
        assert ffmpeg.has_audio_stream(sine_wav) is True

    def test_false_for_a_video_without_audio(self, silent_video: Path) -> None:
        assert ffmpeg.has_audio_stream(silent_video) is False

    def test_false_for_a_non_media_file(self, not_media: Path) -> None:
        assert ffmpeg.has_audio_stream(not_media) is False


class TestCommandBuilders:
    def test_flac_command_always_forces_s16(self, tmp_path: Path) -> None:
        # The FLAC encoder accepts only s16/s32; AAC/Opus sources decode to fltp,
        # so leaving the sample format to filter-graph negotiation can fail.
        cmd = ffmpeg.flac_command(tmp_path / "in.mkv", tmp_path / "out.flac")
        assert cmd[cmd.index("-sample_fmt") + 1] == "s16"

    def test_flac_command_maps_the_first_audio_track_only(self, tmp_path: Path) -> None:
        cmd = ffmpeg.flac_command(tmp_path / "in.mkv", tmp_path / "out.flac")
        assert cmd[cmd.index("-map") + 1] == "0:a:0"
        assert "-vn" in cmd

    def test_source_profile_does_not_resample(self, tmp_path: Path) -> None:
        cmd = ffmpeg.flac_command(tmp_path / "in.mkv", tmp_path / "out.flac")
        assert "-ar" not in cmd
        assert "-ac" not in cmd

    def test_whisper_profile_resamples_to_16k_mono(self, tmp_path: Path) -> None:
        cmd = ffmpeg.flac_command(tmp_path / "i.mkv", tmp_path / "o.flac", profile="whisper")
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-ac") + 1] == "1"

    def test_rejects_an_unknown_profile(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="profile"):
            ffmpeg.flac_command(tmp_path / "i.mkv", tmp_path / "o.flac", profile="nope")

    def test_decode_command_requests_16k_mono_float32(self, tmp_path: Path) -> None:
        cmd = ffmpeg.decode_command(tmp_path / "a.flac")
        assert cmd[cmd.index("-f") + 1] == "f32le"
        assert cmd[cmd.index("-ar") + 1] == "16000"
        assert cmd[cmd.index("-ac") + 1] == "1"

    def test_probe_command_omits_stream_selection_when_not_given(self, tmp_path: Path) -> None:
        assert "-select_streams" not in ffmpeg.probe_command(tmp_path / "a", "format=duration")

    def test_probe_command_includes_stream_selection_when_given(self, tmp_path: Path) -> None:
        cmd = ffmpeg.probe_command(tmp_path / "a", "stream=index", "a:0")
        assert cmd[cmd.index("-select_streams") + 1] == "a:0"


class TestExtractFlac:
    def test_produces_a_playable_flac(self, sine_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "audio.flac"
        ffmpeg.extract_flac(sine_wav, out)
        assert ffmpeg.probe_duration(out) == pytest.approx(1.0, abs=0.05)

    def test_preserves_source_rate_and_channels_by_default(
        self, sine_wav: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "audio.flac"
        ffmpeg.extract_flac(sine_wav, out)
        assert ffmpeg.probe_field(out, "stream=sample_rate", "a:0") == "44100"
        assert ffmpeg.probe_field(out, "stream=channels", "a:0") == "2"

    def test_whisper_profile_downmixes(self, sine_wav: Path, tmp_path: Path) -> None:
        out = tmp_path / "audio.flac"
        ffmpeg.extract_flac(sine_wav, out, profile="whisper")
        assert ffmpeg.probe_field(out, "stream=sample_rate", "a:0") == "16000"
        assert ffmpeg.probe_field(out, "stream=channels", "a:0") == "1"

    def test_rejects_a_source_with_no_audio(self, silent_video: Path, tmp_path: Path) -> None:
        with pytest.raises(errors.NoAudioStreamError):
            ffmpeg.extract_flac(silent_video, tmp_path / "a.flac")

    def test_raises_decode_error_when_ffmpeg_fails(
        self, not_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Bypass the audio-stream gate so the ffmpeg failure itself is exercised.
        def yes(_p: Path) -> bool:
            return True

        monkeypatch.setattr(ffmpeg, "has_audio_stream", yes)
        with pytest.raises(errors.DecodeError):
            ffmpeg.extract_flac(not_media, tmp_path / "a.flac")


class TestDecode16kMono:
    def test_returns_float32_at_16k(self, sine_wav: Path) -> None:
        pcm = ffmpeg.decode_16k_mono(sine_wav)
        assert pcm.dtype == np.float32
        assert len(pcm) == pytest.approx(16000, abs=800)

    def test_audio_is_not_silent(self, sine_wav: Path) -> None:
        assert float(np.abs(ffmpeg.decode_16k_mono(sine_wav)).max()) > 0.1

    def test_raises_decode_error_on_garbage(self, not_media: Path) -> None:
        with pytest.raises(errors.DecodeError):
            ffmpeg.decode_16k_mono(not_media)


class TestPreflight:
    def test_passes_when_binaries_exist(self) -> None:
        ffmpeg.require_binaries()

    def test_reports_the_real_flac_encoder_state(self) -> None:
        assert ffmpeg.has_flac_encoder() is True

    def test_raises_when_a_binary_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(_name: str, _mode: int = 0, _path: str | None = None) -> str | None:
            return None

        monkeypatch.setattr(shutil, "which", missing)
        with pytest.raises(errors.PreflightError, match="ffmpeg"):
            ffmpeg.require_binaries()

    def test_raises_when_the_flac_encoder_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_flac() -> bool:
            return False

        monkeypatch.setattr(ffmpeg, "has_flac_encoder", no_flac)
        with pytest.raises(errors.PreflightError, match="flac"):
            ffmpeg.require_binaries()

    def test_a_missing_binary_surfaces_as_preflight_not_oserror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise FileNotFoundError("ffprobe")

        monkeypatch.setattr(subprocess, "run", boom)
        with pytest.raises(errors.PreflightError):
            ffmpeg.has_flac_encoder()


class TestDurationsMatch:
    def test_an_exact_match(self) -> None:
        assert ffmpeg.durations_match(212.861, 212.861) is True

    def test_within_the_relative_tolerance(self) -> None:
        assert ffmpeg.durations_match(3600.0, 3610.0) is True

    def test_beyond_the_relative_tolerance(self) -> None:
        assert ffmpeg.durations_match(3600.0, 3700.0) is False

    def test_short_clips_get_an_absolute_floor(self) -> None:
        # 0.5% of two seconds is 10ms; rounding in container metadata exceeds that.
        assert ffmpeg.durations_match(2.0, 2.9) is True

    def test_a_near_empty_extraction_never_matches(self) -> None:
        assert ffmpeg.durations_match(0.02, 212.861) is False
