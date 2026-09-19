"""Resume planning.

Walk the stages backwards and enter at the first one whose output does not verify.
A stage that is *not* re-running is satisfied by an artifact deleted on purpose;
a stage that *is* re-running is not, and we walk further back to re-materialize it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from audio_scribe.jobs.verify import Verdict


class Stage(StrEnum):
    DOWNLOAD = "download"
    AUDIO = "audio"
    TRANSCRIBE = "transcribe"
    OUTPUTS = "outputs"


@dataclass(frozen=True, slots=True)
class Verdicts:
    video: Verdict
    audio: Verdict
    outputs: Verdict


def plan_stages(
    verdicts: Verdicts,
    *,
    params_changed: bool = False,
    segments_available: bool = False,
    force: frozenset[Stage] = frozenset(),
) -> tuple[Stage, ...]:
    """The stages to run, in pipeline order."""
    need_outputs = (
        verdicts.outputs is not Verdict.OK
        or params_changed
        or bool(force & {Stage.OUTPUTS, Stage.TRANSCRIBE, Stage.AUDIO, Stage.DOWNLOAD})
    )
    # Re-rendering from saved segments avoids re-transcribing a long recording
    # just because one subtitle file was deleted.
    need_transcribe = need_outputs and (
        params_changed
        or not segments_available
        or bool(force & {Stage.TRANSCRIBE, Stage.AUDIO, Stage.DOWNLOAD})
    )
    need_audio = (
        Stage.AUDIO in force
        or Stage.DOWNLOAD in force
        or (need_transcribe and verdicts.audio is not Verdict.OK)
    )
    need_download = Stage.DOWNLOAD in force or (need_audio and verdicts.video is not Verdict.OK)

    wanted = {
        Stage.DOWNLOAD: need_download,
        Stage.AUDIO: need_audio,
        Stage.TRANSCRIBE: need_transcribe,
        Stage.OUTPUTS: need_outputs,
    }
    return tuple(stage for stage in Stage if wanted[stage])
