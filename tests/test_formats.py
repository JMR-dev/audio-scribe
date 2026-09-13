from __future__ import annotations

import json

import pytest

from ccn_transcribe import formats
from ccn_transcribe.transcript import Segment, TranscriptResult

SEGMENTS = (Segment(0.0, 1.5, "Hello there."), Segment(1.5, 3.25, "General Kenobi."))

RESULT = TranscriptResult(
    segments=SEGMENTS,
    language="en",
    backend="openvino",
    device="GPU",
    model_id="OpenVINO/whisper-large-v3-turbo-int8-ov",
    model_revision="b568445d",
    audio_s=3.25,
    wall_s=1.0,
)


class TestTimestamps:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0.0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3600.0, "01:00:00,000"),
            (7322.5, "02:02:02,500"),
        ],
    )
    def test_srt_timestamp(self, seconds: float, expected: str) -> None:
        assert formats.srt_timestamp(seconds) == expected

    def test_vtt_uses_a_dot_separator(self) -> None:
        assert formats.vtt_timestamp(61.25) == "00:01:01.250"

    def test_rounding_never_produces_sixty_seconds(self) -> None:
        # Naive f"{s:06.3f}" renders this as 00:59:60.000, which is invalid and
        # which some players silently mis-seek on.
        assert formats.srt_timestamp(3599.9996) == "01:00:00,000"
        assert formats.vtt_timestamp(3599.9996) == "01:00:00.000"

    def test_negative_times_clamp_to_zero(self) -> None:
        assert formats.srt_timestamp(-5.0) == "00:00:00,000"

    def test_hours_beyond_two_digits_still_render(self) -> None:
        assert formats.srt_timestamp(360000.0).startswith("100:00:00")


class TestTxt:
    def test_joins_segment_text(self) -> None:
        assert formats.to_txt(RESULT) == "Hello there. General Kenobi."

    def test_ends_with_a_newline(self) -> None:
        assert formats.to_txt(RESULT).endswith("\n") is False  # body only; writer adds newline


class TestSrt:
    def test_structure(self) -> None:
        assert formats.to_srt(SEGMENTS) == (
            "1\n"
            "00:00:00,000 --> 00:00:01,500\n"
            "Hello there.\n"
            "\n"
            "2\n"
            "00:00:01,500 --> 00:00:03,250\n"
            "General Kenobi.\n"
        )

    def test_empty_segments_give_empty_output(self) -> None:
        assert formats.to_srt(()) == ""


class TestVtt:
    def test_has_the_required_header(self) -> None:
        assert formats.to_vtt(SEGMENTS).startswith("WEBVTT\n\n")

    def test_uses_dot_separated_timestamps_and_no_cue_numbers(self) -> None:
        body = formats.to_vtt(SEGMENTS)
        assert "00:00:00.000 --> 00:00:01.500" in body
        assert "," not in body.split("Hello", maxsplit=1)[0]

    def test_empty_segments_still_emit_the_header(self) -> None:
        assert formats.to_vtt(()) == "WEBVTT\n"


class TestJson:
    def test_records_run_metadata(self) -> None:
        doc = json.loads(formats.to_json(RESULT))
        assert doc["backend"] == "openvino"
        assert doc["device"] == "GPU"
        assert doc["model"]["id"] == "OpenVINO/whisper-large-v3-turbo-int8-ov"
        assert doc["model"]["revision"] == "b568445d"
        assert doc["language"] == "en"
        assert doc["audio_s"] == 3.25
        assert doc["wall_s"] == 1.0
        assert doc["rtf"] == pytest.approx(3.25)

    def test_records_segments_with_times(self) -> None:
        doc = json.loads(formats.to_json(RESULT))
        assert doc["segments"] == [
            {"start": 0.0, "end": 1.5, "text": "Hello there."},
            {"start": 1.5, "end": 3.25, "text": "General Kenobi."},
        ]

    def test_includes_full_text(self) -> None:
        assert json.loads(formats.to_json(RESULT))["text"] == "Hello there. General Kenobi."

    def test_merges_source_metadata_when_given(self) -> None:
        doc = json.loads(formats.to_json(RESULT, source={"id": "abc", "title": "T"}))
        assert doc["source"] == {"id": "abc", "title": "T"}

    def test_source_key_absent_when_not_given(self) -> None:
        assert "source" not in json.loads(formats.to_json(RESULT))

    def test_is_utf8_readable_not_escaped(self) -> None:
        result = TranscriptResult(
            segments=(Segment(0.0, 1.0, "naïve café"),),
            language="fr",
            backend="openvino",
            device="CPU",
            model_id="m",
            model_revision=None,
            audio_s=1.0,
            wall_s=1.0,
        )
        assert "naïve café" in formats.to_json(result)


def test_render_returns_every_requested_format() -> None:
    out = formats.render(RESULT, ("txt", "srt", "vtt", "json"))
    assert set(out) == {"txt", "srt", "vtt", "json"}
    assert out["srt"].startswith("1\n")


def test_render_rejects_an_unknown_format() -> None:
    with pytest.raises(ValueError, match="unknown"):
        formats.render(RESULT, ("txt", "docx"))
