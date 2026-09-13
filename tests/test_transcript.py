from __future__ import annotations

import logging

import pytest

from ccn_transcribe.transcript import Segment, TranscriptResult, normalize_segments


def _result(**kw: object) -> TranscriptResult:
    base: dict[str, object] = {
        "segments": (Segment(0.0, 1.0, "hi"),),
        "language": "en",
        "backend": "openvino",
        "device": "GPU",
        "model_id": "m",
        "model_revision": "rev",
        "audio_s": 10.0,
        "wall_s": 2.0,
    }
    base.update(kw)
    return TranscriptResult(**base)  # type: ignore[arg-type]


class TestNormalize:
    def test_passes_through_clean_segments(self) -> None:
        got = normalize_segments([(0.0, 1.5, "a"), (1.5, 3.0, "b")], audio_s=3.0)
        assert got == (Segment(0.0, 1.5, "a"), Segment(1.5, 3.0, "b"))

    def test_negative_end_sentinel_clamps_to_audio_duration(self) -> None:
        # OpenVINO GenAI mirrors HF Whisper and emits -1 when the closing
        # timestamp token is never predicted, typically on the final chunk.
        got = normalize_segments([(0.0, 2.0, "a"), (2.0, -1.0, "b")], audio_s=5.0)
        assert got[-1].end == 5.0

    def test_negative_end_uses_next_start_when_one_follows(self) -> None:
        got = normalize_segments([(0.0, -1.0, "a"), (2.0, 3.0, "b")], audio_s=9.0)
        assert got[0].end == 2.0

    def test_none_end_is_handled_like_the_sentinel(self) -> None:
        got = normalize_segments([(0.0, None, "a")], audio_s=4.0)
        assert got[0].end == 4.0

    def test_none_start_continues_from_previous_end(self) -> None:
        got = normalize_segments([(0.0, 2.0, "a"), (None, 4.0, "b")], audio_s=6.0)
        assert got[1].start == 2.0

    def test_negative_start_is_clamped_to_zero(self) -> None:
        assert normalize_segments([(-3.0, 1.0, "a")], audio_s=2.0)[0].start == 0.0

    def test_end_beyond_audio_duration_is_clamped(self) -> None:
        assert normalize_segments([(0.0, 99.0, "a")], audio_s=5.0)[0].end == 5.0

    def test_end_before_start_is_repaired_to_a_positive_span(self) -> None:
        got = normalize_segments([(3.0, 1.0, "a")], audio_s=10.0)
        assert got[0].end > got[0].start

    def test_repaired_span_never_exceeds_audio_duration(self) -> None:
        got = normalize_segments([(10.0, 1.0, "a")], audio_s=10.0)
        assert got[0].end <= 10.0

    def test_blank_segments_are_dropped(self) -> None:
        assert normalize_segments([(0.0, 1.0, "   "), (1.0, 2.0, "real")], audio_s=2.0) == (
            Segment(1.0, 2.0, "real"),
        )

    def test_text_is_stripped(self) -> None:
        assert normalize_segments([(0.0, 1.0, "  hi  ")], audio_s=2.0)[0].text == "hi"

    def test_empty_input_gives_empty_output(self) -> None:
        got = normalize_segments([], audio_s=5.0)
        assert isinstance(got, tuple)
        assert not got

    def test_non_monotonic_starts_are_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        # Guards the unverified assumption that start_ts is absolute across
        # WhisperPipeline's internal 30s windows rather than window-relative.
        with caplog.at_level(logging.WARNING):
            normalize_segments([(5.0, 6.0, "a"), (1.0, 2.0, "b")], audio_s=30.0)
        assert "monotonic" in caplog.text.lower()

    def test_monotonic_starts_are_not_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            normalize_segments([(0.0, 1.0, "a"), (1.0, 2.0, "b")], audio_s=30.0)
        assert caplog.text == ""


class TestTranscriptResult:
    def test_text_joins_segments(self) -> None:
        r = _result(segments=(Segment(0, 1, "hello"), Segment(1, 2, "world")))
        assert r.text == "hello world"

    def test_rtf_is_audio_over_wall(self) -> None:
        assert _result(audio_s=100.0, wall_s=20.0).rtf == pytest.approx(5.0)

    def test_rtf_is_zero_when_wall_time_is_zero(self) -> None:
        assert _result(wall_s=0.0).rtf == 0.0


def test_negative_end_falls_back_to_audio_duration_when_no_later_start_qualifies() -> None:
    # The following chunk starts *before* this one, so it cannot serve as an end.
    got = normalize_segments([(2.0, -1.0, "a"), (1.0, 3.0, "b")], audio_s=8.0)
    assert got[0].end == 8.0
