"""The download stage.

yt-dlp classification lives in a regex table so it can be unit-tested without the
network and amended when YouTube changes its wording.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from audio_scribe import errors, paths
from audio_scribe.media import ytdlp_opts

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from audio_scribe.paths import JobPaths

log = logging.getLogger(__name__)

YdlFactory = Callable[[dict[str, Any]], Any]

# (pattern, error class, hint). Order matters: first match wins.
SOURCE_PATTERNS: tuple[tuple[re.Pattern[str], type[errors.SourceError], str], ...] = (
    (
        re.compile(r"private video|sign in if you.{0,20}been granted access", re.I),
        errors.PrivateVideoError,
        "This video is private. Supply cookies: --cookies-from-browser firefox",
    ),
    (
        re.compile(r"sign in to confirm your age|age.?restricted|inappropriate for some", re.I),
        errors.AgeGatedError,
        "Age-restricted. Use --cookies-from-browser with a signed-in account.",
    ),
    (
        re.compile(r"members-only|available to (this channel's )?members", re.I),
        errors.MembersOnlyError,
        "Members-only content. Needs cookies from a subscribed account.",
    ),
    (
        re.compile(r"confirm you.{0,5}re not a bot|not a bot", re.I),
        errors.BotCheckError,
        "YouTube bot check. Try --cookies-from-browser, or check that a JS runtime "
        "is available: audio-scribe doctor",
    ),
    (
        re.compile(
            r"geo.?restricted|geo.?block|available in your country|"
            r"not available from your location|blocked it in your",
            re.I,
        ),
        errors.GeoBlockedError,
        "Geo-blocked from this location.",
    ),
    (
        re.compile(
            r"http error 4(0[34]|10)|video unavailable|does not exist|been removed|"
            r"is not available",
            re.I,
        ),
        errors.NotFoundError,
        "Video not available (removed, or the URL is wrong).",
    ),
    (
        re.compile(
            r"unable to download|connection reset|timed out|timeout|"
            r"temporary failure|http error 5\d\d|network is unreachable",
            re.I,
        ),
        errors.NetworkError,
        "Network problem; this is retryable.",
    ),
)

# aria2c process exit codes worth explaining. 9 is promoted to a fatal DiskFull.
ARIA2_EXITS: dict[int, tuple[str, str]] = {
    1: ("unknown error", "See logs/aria2.log in the job directory."),
    2: ("timed out", "The server stopped responding; retrying usually clears it."),
    3: ("resource not found", "Format URLs are signed and expire (~6h); re-run to re-extract."),
    5: ("aborted: too slow", "Hit --lowest-speed-limit. Lower it for a genuinely slow link."),
    6: ("network problem", "Retryable."),
    7: ("unfinished downloads remained", "The .aria2 control file allows a resume."),
    8: ("server does not support resume", "Restarting the download from scratch."),
    22: ("HTTP 4xx response header", "Often a 403 from too many connections: --aria2c-conns 4."),
    24: ("HTTP authorization failed", "Needs cookies: --cookies-from-browser."),
}


def classify(message: str, job_id: str | None = None) -> errors.AudioScribeError:
    """Map a yt-dlp message onto an actionable error."""
    if "no space left" in message.lower():
        return errors.DiskFullError(message, hint="Free some space and re-run.")
    for pattern, cls, hint in SOURCE_PATTERNS:
        if pattern.search(message):
            return cls(message, job_id=job_id, hint=hint)

    found = re.search(r"exited with code (\d+)", message)
    if found:
        code = int(found.group(1))
        if code == 9:
            return errors.DiskFullError(message, hint="aria2c ran out of disk space.")
        reason, hint = ARIA2_EXITS.get(code, ("failed", "See logs/aria2.log."))
        return errors.Aria2Error(
            f"aria2c {reason} (exit {code})", aria2_exit_code=code, job_id=job_id, hint=hint
        )
    return errors.DownloadError(message, job_id=job_id, hint="Re-run with -v for detail.")


def default_factory(options: dict[str, Any]) -> Any:
    from yt_dlp import YoutubeDL  # pylint: disable=import-outside-toplevel

    return YoutubeDL(options)  # pyright: ignore[reportArgumentType]


def extract_info(
    url: str,
    config: ytdlp_opts.DownloadConfig = ytdlp_opts.DEFAULT_DOWNLOAD,
    *,
    factory: YdlFactory | None = None,
) -> dict[str, Any]:
    """Metadata only, no download."""
    options = ytdlp_opts.ydl_opts("-", config)
    options.update({"skip_download": True, "writeinfojson": False, "extract_flat": "in_playlist"})
    make = factory or default_factory
    try:
        with make(options) as ydl:
            info: dict[str, Any] | None = ydl.extract_info(url, download=False)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise classify(str(exc)) from exc
    if info is None:
        # yt-dlp reported the reason on its own output and handed back nothing, so
        # the cause is whatever it just printed -- a bad URL, or an unreachable one.
        raise errors.NotFoundError(
            f"no metadata for {url}",
            hint="Check the URL, and the yt-dlp error above for why it could not be read.",
        )
    return info


def entries_for(info: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a playlist into one record per video."""
    raw = info.get("entries")
    if raw is None:
        return [info]
    return [entry for entry in raw if entry]


def identify(entry: dict[str, Any], url: str) -> str:
    extractor = entry.get("extractor_key") or entry.get("ie_key") or ""
    video_id = entry.get("id")
    if extractor and video_id:
        return paths.job_id(str(extractor), str(video_id))
    return paths.url_job_id(url)


def find_video(job: JobPaths) -> Path | None:
    """The merged media file, whatever extension yt-dlp settled on.

    Only the file yt-dlp's output template names directly. A failed merge leaves
    per-format streams like video.f251.webm behind, which sort first and hold one
    stream each; returning one would silently transcribe half the source.
    """
    for candidate in sorted(job.media_dir.glob("video.*")):
        if candidate.stem == "video" and candidate.suffix not in (".part", ".aria2", ".ytdl"):
            return candidate
    return None


def _part_files(job: JobPaths) -> Iterator[Path]:
    for directory in (job.media_dir, job.tmp_dir):
        # glob yields nothing for a directory that does not exist; no guard needed.
        yield from directory.glob("*.part")


def clear_stale_parts(job: JobPaths) -> list[Path]:
    """Delete .part files that cannot be resumed, and say which.

    aria2 writes its 16 segments out of order, so without the .aria2 control file
    a .part is a sparse file full of holes rather than a valid prefix. yt-dlp's
    continuedl would resume from its length and produce a video that plays for a
    while and then isn't the source any more -- a corrupt transcript, not an error.
    Re-downloading is the cheaper mistake, so an orphan goes even when the native
    downloader (which does write sequentially) could have resumed it.
    """
    removed: list[Path] = []
    for part in sorted(_part_files(job)):
        if part.with_suffix(part.suffix + ".aria2").exists():
            continue
        part.unlink()
        removed.append(part)
        log.warning("discarded unresumable partial download %s", part.name)
    return removed


def discard_video(job: JobPaths) -> Path | None:
    """Delete the merged file so yt-dlp will actually fetch it again.

    yt-dlp's overwrites default is off, so it reports an existing target as
    "already downloaded" and skips it; a truncated video.mkv would otherwise never
    be repaired. Per-format streams are left alone -- yt-dlp re-merges complete
    ones without re-downloading, so clearing them is a pessimization.
    """
    video = find_video(job)
    if video is not None:
        video.unlink()
        log.info("discarded %s so it can be downloaded again", video.name)
    return video


def download_options(job: JobPaths, config: ytdlp_opts.DownloadConfig) -> dict[str, Any]:
    return ytdlp_opts.ydl_opts(
        str(job.media_dir / "video.%(ext)s"),
        config,
        aria2_log=job.logs_dir / "aria2.log",
    )


def download(
    job: JobPaths,
    url: str,
    config: ytdlp_opts.DownloadConfig = ytdlp_opts.DEFAULT_DOWNLOAD,
    *,
    factory: YdlFactory | None = None,
    attempts: int = 3,
    backoff: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Download to the job directory, retrying transient failures ourselves.

    yt-dlp's ExternalFD calls the downloader exactly once and turns a nonzero exit
    into report_error(); its `retries` option does not cover external downloaders.
    Since --lowest-speed-limit deliberately makes aria2c exit nonzero on a stall,
    without this loop that stall would become a job failure instead of a resume.
    """
    job.ensure()
    # Before the loop, not inside it: a failed attempt leaves a .part with a live
    # control file, which is exactly the progress the next attempt resumes from.
    clear_stale_parts(job)
    discard_video(job)
    options = download_options(job, config)
    make = factory or default_factory

    last: errors.AudioScribeError | None = None
    for attempt in range(1, attempts + 1):
        try:
            with make(options) as ydl:
                ydl.download([url])
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failure = classify(str(exc), job_id=job.root.name)
            if isinstance(failure, errors.FatalError) or not is_retryable(failure):
                raise failure from exc
            last = failure
            if attempt < attempts:
                delay = backoff ** (attempt - 1)
                log.warning(
                    "download attempt %d/%d failed (%s); retrying in %.0fs",
                    attempt,
                    attempts,
                    failure,
                    delay,
                )
                sleep(delay)
            continue

        found = find_video(job)
        if found is not None:
            return found
        last = errors.IncompleteDownloadError(
            "yt-dlp reported success but produced no media file",
            job_id=job.root.name,
            hint="Check the job's logs, or re-run with -v.",
        )
        break

    raise last or errors.DownloadError("download failed", job_id=job.root.name)


def is_retryable(failure: errors.AudioScribeError) -> bool:
    # A private video will still be private on the third try, and continuing past
    # ENOSPC only makes more corrupt artifacts.
    if isinstance(failure, errors.FatalError):
        return False
    return failure.retryable or isinstance(failure, (errors.Aria2Error, errors.DownloadError))


def aria2c_available(which: Callable[[str], str | None] | None = None) -> bool:
    lookup = which or shutil.which
    return lookup("aria2c") is not None


def resolve_aria2c(requested: bool, *, available: bool) -> bool:
    """Never fail because aria2c is missing; fall back and say so."""
    if requested and not available:
        log.warning("aria2c not found; using yt-dlp's native downloader")
        return False
    return requested


def describe_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "url": entry.get("webpage_url") or entry.get("url"),
        "duration": entry.get("duration"),
        "uploader": entry.get("uploader"),
        "upload_date": entry.get("upload_date"),
    }


def summarize(entries: Sequence[dict[str, Any]]) -> str:
    return f"{len(entries)} video{'s' if len(entries) != 1 else ''}"
