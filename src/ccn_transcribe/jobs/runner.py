"""Per-job execution: verify, plan, run the needed stages, apply retention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ccn_transcribe import errors
from ccn_transcribe.jobs import store, verify
from ccn_transcribe.jobs.plan import Stage, Verdicts, plan_stages
from ccn_transcribe.jobs.state import Artifact, ArtifactStatus, JobState, Outputs
from ccn_transcribe.jobs.state import TranscriptionRecord as Record
from ccn_transcribe.media import ffmpeg
from ccn_transcribe.stages import audio as audio_stage
from ccn_transcribe.stages import download as download_stage
from ccn_transcribe.stages import outputs as outputs_stage
from ccn_transcribe.stages import transcribe as transcribe_stage

if TYPE_CHECKING:
    from pathlib import Path

    from ccn_transcribe.backends.registry import BackendHolder
    from ccn_transcribe.config import RunConfig
    from ccn_transcribe.paths import JobPaths

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    job_id: str
    url: str
    stages: tuple[Stage, ...]
    ok: bool
    error: str | None = None
    rtf: float | None = None
    device: str | None = None


def wanted_record(config: RunConfig) -> Record:
    return Record(
        model_id=config.model,
        model_revision=None,
        backend=config.device or "auto",
        device=config.device or "auto",
        language=config.language,
        task=config.task,
        num_beams=config.num_beams,
    )


def current_verdicts(job: JobPaths, state: JobState, config: RunConfig) -> Verdicts:
    return Verdicts(
        video=verify.verify_video(job, state.video, state.video.size),
        audio=verify.verify_audio(job, state.audio, state.duration_s),
        outputs=verify.verify_outputs(job, state.outputs, config.formats),
    )


def describe_plan(job_id: str, stages: tuple[Stage, ...], verdicts: Verdicts) -> str:
    if not stages:
        return f"{job_id}: already complete"
    notes: list[str] = []
    if Stage.DOWNLOAD in stages and verdicts.video is verify.Verdict.SATISFIED_ABSENT:
        notes.append("re-downloading because the video was removed by a retention flag")
    if Stage.AUDIO in stages and verdicts.audio is verify.Verdict.SATISFIED_ABSENT:
        notes.append("re-extracting audio removed by a retention flag")
    if verdicts.video is verify.Verdict.PARTIAL:
        notes.append("resuming a partial download")
    plan_text = " -> ".join(s.value for s in stages)
    return f"{job_id}: {plan_text}" + (f" ({'; '.join(notes)})" if notes else "")


def _record_deletion(job: JobPaths, state: JobState, which: str, reason: str, path: Path) -> None:
    """Remove an artifact on purpose, and record that it was on purpose."""
    path.unlink(missing_ok=True)
    artifact = state.video if which == "video" else state.audio
    artifact.status = ArtifactStatus.DELETED_BY_POLICY
    artifact.reason = reason
    artifact.deleted_at = datetime.now(UTC).isoformat(timespec="seconds")
    store.append_event(job, {"event": "deleted", "artifact": which, "reason": reason})
    store.save(job, state)
    log.info("%s removed by %s", which, reason)


def run_job(
    job: JobPaths,
    url: str,
    config: RunConfig,
    *,
    entry: dict[str, Any] | None = None,
    holder: BackendHolder | None = None,
    dry_run: bool = False,
) -> JobOutcome:
    """Run whatever stages this job still needs."""
    job.ensure()
    state = store.load(job) or JobState.new(job_id=job.root.name, source_url=url)
    if entry:
        state.title = entry.get("title") or state.title
        duration = entry.get("duration")
        state.duration_s = float(duration) if duration else state.duration_s

    verdicts = current_verdicts(job, state, config)
    segments_available = transcribe_stage.saved_segments(job) is not None
    stages = plan_stages(
        verdicts,
        params_changed=state.params_changed(wanted_record(config)),
        segments_available=segments_available,
        force=config.force,
    )
    log.info("%s", describe_plan(job.root.name, stages, verdicts))
    if dry_run or not stages:
        return JobOutcome(job.root.name, url, stages, ok=True)

    try:
        _execute(job, url, config, state, stages=stages, holder=holder)
    except errors.FatalError:
        raise
    except errors.CcnError as exc:
        state.last_error = {"stage": "unknown", "message": str(exc)}
        store.save(job, state)
        return JobOutcome(job.root.name, url, stages, ok=False, error=str(exc))

    rtf = state.transcription.rtf if state.transcription else None
    device = state.transcription.device if state.transcription else None
    return JobOutcome(job.root.name, url, stages, ok=True, rtf=rtf, device=device)


def _execute(
    job: JobPaths,
    url: str,
    config: RunConfig,
    state: JobState,
    *,
    stages: tuple[Stage, ...],
    holder: BackendHolder | None,
) -> None:
    video: Path | None = download_stage.find_video(job)

    if Stage.DOWNLOAD in stages:
        video = _do_download(job, url, config, state)
    if Stage.AUDIO in stages:
        _do_audio(job, config, state, video)
    # need_outputs is the root condition in plan_stages, so every non-empty plan
    # ends in OUTPUTS and this always has work to do.
    _do_transcript(job, config, state, stages=stages, holder=holder)


def _do_download(job: JobPaths, url: str, config: RunConfig, state: JobState) -> Path:
    state.attempts["download"] = state.attempts.get("download", 0) + 1
    video = download_stage.download(job, url, config.download, attempts=config.attempts)
    state.video = Artifact(
        status=ArtifactStatus.PRESENT,
        path=str(video.relative_to(job.root)),
        size=video.stat().st_size,
        duration_s=ffmpeg.probe_duration(video),
    )
    state.duration_s = state.video.duration_s or state.duration_s
    store.save(job, state)
    return video


def _do_audio(job: JobPaths, config: RunConfig, state: JobState, video: Path | None) -> None:
    if video is None:
        raise errors.MediaError(
            "no video available to extract audio from",
            job_id=job.root.name,
            hint="Re-run without --force-stage audio, or let the download stage run.",
        )
    state.attempts["audio"] = state.attempts.get("audio", 0) + 1
    flac = audio_stage.extract(job, video, profile=config.audio_profile)
    state.audio = Artifact(
        status=ArtifactStatus.PRESENT,
        path=str(flac.relative_to(job.root)),
        size=flac.stat().st_size,
        duration_s=ffmpeg.probe_duration(flac),
    )
    store.save(job, state)
    # Only now, with the successor verified and committed, is deletion safe.
    if config.no_retain_video:
        _record_deletion(job, state, "video", "--no-retain-video", video)


def _do_transcript(
    job: JobPaths,
    config: RunConfig,
    state: JobState,
    *,
    stages: tuple[Stage, ...],
    holder: BackendHolder | None,
) -> None:
    if Stage.TRANSCRIBE in stages:
        record = wanted_record(config)
        record.started_at = state.updated_at
        state.transcription = record
        store.save(job, state)

        state.attempts["transcribe"] = state.attempts.get("transcribe", 0) + 1
        result = transcribe_stage.run(
            job,
            job.audio_file,
            config.request(),
            holder=holder,
            requested=config.device,
            model_id=config.model,
            cache_dir=config.ov_cache_dir,
        )
        state.transcription = Record(
            model_id=result.model_id,
            model_revision=result.model_revision,
            backend=result.backend,
            device=result.device,
            language=config.language,
            task=config.task,
            num_beams=config.num_beams,
            segments=len(result.segments),
            audio_s=result.audio_s,
            wall_s=result.wall_s,
            rtf=result.rtf,
            started_at=record.started_at,
            finished_at=state.updated_at,
        )
        written = outputs_stage.write(job, result, config.formats, source=_source(state))
    else:
        written = _rerender(job, config, state)

    state.outputs = Outputs(status=ArtifactStatus.PRESENT, paths=written)
    store.save(job, state)
    outputs_stage.publish(job, written, config.formats, out_dir=config.out_dir, title=state.title)
    if config.no_retain_audio:
        _record_deletion(job, state, "audio", "--no-retain-audio", job.audio_file)


def _rerender(job: JobPaths, config: RunConfig, state: JobState) -> dict[str, str]:
    """Re-render other formats from the saved segments, no model needed."""
    from ccn_transcribe.transcript import (  # pylint: disable=import-outside-toplevel
        TranscriptResult,
    )

    segments = transcribe_stage.saved_segments(job)
    record = state.transcription
    if segments is None or record is None:  # pragma: no cover - planner guarantees this
        raise errors.TranscribeError("no saved segments to re-render from")
    result = TranscriptResult(
        segments=segments,
        language=record.language,
        backend=record.backend,
        device=record.device,
        model_id=record.model_id or "<backend default>",
        model_revision=record.model_revision,
        audio_s=record.audio_s or 0.0,
        wall_s=record.wall_s or 0.0,
    )
    return outputs_stage.write(job, result, config.formats, source=_source(state))


def _source(state: JobState) -> dict[str, object]:
    return {"id": state.job_id, "title": state.title, "url": state.source_url}
