from __future__ import annotations

import pytest

from audio_scribe.jobs.plan import Stage, Verdicts, plan_stages
from audio_scribe.jobs.verify import Verdict

OK = Verdict.OK
GONE = Verdict.SATISFIED_ABSENT
MISS = Verdict.MISSING
PART = Verdict.PARTIAL
BAD = Verdict.CORRUPT


def plan(video: Verdict, audio: Verdict, outputs: Verdict, **kw: object) -> tuple[Stage, ...]:
    return plan_stages(Verdicts(video=video, audio=audio, outputs=outputs), **kw)  # type: ignore[arg-type]


class TestNothingToDo:
    def test_everything_present_is_a_no_op(self) -> None:
        assert plan(OK, OK, OK) == ()

    def test_media_deleted_by_policy_is_still_a_no_op(self) -> None:
        # The whole point of --no-retain: done is done, do not re-fetch.
        assert plan(GONE, GONE, OK) == ()


class TestEntryPoint:
    def test_missing_outputs_with_good_audio_skips_download(self) -> None:
        assert plan(OK, OK, MISS) == (Stage.TRANSCRIBE, Stage.OUTPUTS)

    def test_policy_deleted_video_does_not_trigger_a_download(self) -> None:
        assert plan(GONE, OK, MISS) == (Stage.TRANSCRIBE, Stage.OUTPUTS)

    def test_missing_audio_with_good_video_extracts_only(self) -> None:
        assert plan(OK, MISS, MISS) == (Stage.AUDIO, Stage.TRANSCRIBE, Stage.OUTPUTS)

    def test_nothing_present_runs_everything(self) -> None:
        assert plan(MISS, MISS, MISS) == (
            Stage.DOWNLOAD,
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_a_partial_download_still_enters_at_download(self) -> None:
        assert plan(PART, MISS, MISS)[0] is Stage.DOWNLOAD

    def test_a_corrupt_video_is_re_downloaded(self) -> None:
        assert plan(BAD, MISS, MISS)[0] is Stage.DOWNLOAD

    def test_a_corrupt_audio_is_re_extracted(self) -> None:
        assert plan(OK, BAD, MISS) == (Stage.AUDIO, Stage.TRANSCRIBE, Stage.OUTPUTS)


class TestSegmentReuse:
    def test_outputs_alone_are_re_rendered_when_segments_survive(self) -> None:
        # Deleting transcript.srt should not cost a re-transcription of a
        # two-hour lecture when transcript.json still holds the segments.
        assert plan(OK, OK, MISS, segments_available=True) == (Stage.OUTPUTS,)

    def test_policy_deleted_audio_is_fine_when_segments_survive(self) -> None:
        assert plan(GONE, GONE, MISS, segments_available=True) == (Stage.OUTPUTS,)

    def test_changed_params_ignore_saved_segments(self) -> None:
        assert Stage.TRANSCRIBE in plan(OK, OK, MISS, segments_available=True, params_changed=True)


class TestParamsChanged:
    def test_a_new_language_forces_a_re_transcription(self) -> None:
        assert plan(OK, OK, OK, params_changed=True) == (Stage.TRANSCRIBE, Stage.OUTPUTS)

    def test_it_re_extracts_audio_that_was_policy_deleted(self) -> None:
        assert plan(OK, GONE, OK, params_changed=True) == (
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_it_re_downloads_when_both_were_policy_deleted(self) -> None:
        # A stage that is re-running is NOT satisfied by a policy deletion.
        assert plan(GONE, GONE, OK, params_changed=True) == (
            Stage.DOWNLOAD,
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )


class TestForce:
    def test_forcing_transcribe_reruns_it(self) -> None:
        assert plan(OK, OK, OK, force=frozenset({Stage.TRANSCRIBE})) == (
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_forcing_transcribe_walks_back_past_policy_deletions(self) -> None:
        assert plan(GONE, GONE, OK, force=frozenset({Stage.TRANSCRIBE})) == (
            Stage.DOWNLOAD,
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_forcing_download_reruns_the_whole_pipeline(self) -> None:
        assert plan(OK, OK, OK, force=frozenset({Stage.DOWNLOAD})) == (
            Stage.DOWNLOAD,
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_forcing_audio_keeps_the_existing_download(self) -> None:
        assert plan(OK, OK, OK, force=frozenset({Stage.AUDIO})) == (
            Stage.AUDIO,
            Stage.TRANSCRIBE,
            Stage.OUTPUTS,
        )

    def test_forcing_outputs_only_re_renders(self) -> None:
        assert plan(OK, OK, OK, force=frozenset({Stage.OUTPUTS}), segments_available=True) == (
            Stage.OUTPUTS,
        )


class TestStageOrder:
    @pytest.mark.parametrize("segments", [True, False])
    def test_stages_always_come_out_in_pipeline_order(self, segments: bool) -> None:
        got = plan(MISS, MISS, MISS, segments_available=segments)
        assert list(got) == sorted(got, key=list(Stage).index)
