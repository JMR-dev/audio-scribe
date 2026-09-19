from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from audio_scribe import errors, paths
from audio_scribe.media import ytdlp_opts
from audio_scribe.stages import download

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def job(tmp_path: Path) -> paths.JobPaths:
    j = paths.Workspace(tmp_path).job("youtube-abc")
    j.ensure()
    return j


class FakeYdl:
    """Stands in for YoutubeDL; records calls and can be told to fail."""

    def __init__(
        self,
        options: dict[str, Any],
        *,
        raises: Exception | None = None,
        info: dict[str, Any] | None = None,
        on_download: Any = None,
    ) -> None:
        self.options = options
        self.raises = raises
        self.info = info
        self.on_download = on_download
        self.downloads: list[list[str]] = []

    def __enter__(self) -> FakeYdl:
        return self

    def __exit__(self, *_exc: object) -> None:
        return

    def extract_info(  # pylint: disable=unused-argument
        self, url: str, download: bool = True
    ) -> dict[str, Any] | None:
        if self.raises:
            raise self.raises
        return self.info

    def download(self, urls: list[str]) -> None:
        self.downloads.append(urls)
        if self.on_download is not None:
            self.on_download(len(self.downloads))
        if self.raises:
            raise self.raises


def factory_for(**kw: Any) -> Any:
    made: list[FakeYdl] = []

    def make(options: dict[str, Any]) -> FakeYdl:
        ydl = FakeYdl(options, **kw)
        made.append(ydl)
        return ydl

    make.made = made  # type: ignore[attr-defined]
    return make


class TestClassify:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                "ERROR: Private video. Sign in if you've been granted access",
                errors.PrivateVideoError,
            ),
            ("Sign in to confirm your age", errors.AgeGatedError),
            ("This video is available to members-only", errors.MembersOnlyError),
            ("Sign in to confirm you're not a bot", errors.BotCheckError),
            (
                "The uploader has not made this video available in your country",
                errors.GeoBlockedError,
            ),
            ("ERROR: Video unavailable", errors.NotFoundError),
            ("HTTP Error 404: Not Found", errors.NotFoundError),
            ("HTTP Error 503: Service Unavailable", errors.NetworkError),
            ("Unable to download webpage: timed out", errors.NetworkError),
        ],
    )
    def test_maps_messages_to_actionable_errors(
        self, message: str, expected: type[errors.JobError]
    ) -> None:
        assert isinstance(download.classify(message), expected)

    def test_every_source_error_carries_a_hint(self) -> None:
        assert download.classify("Private video").hint

    def test_the_bot_check_hint_points_at_doctor(self) -> None:
        # A missing JS runtime presents as a bot check, so this is the likely cause.
        hint = download.classify("Sign in to confirm you're not a bot").hint or ""
        assert "doctor" in hint

    def test_only_network_errors_are_retryable(self) -> None:
        assert download.classify("timed out").retryable is True
        assert download.classify("Private video").retryable is False

    def test_no_space_becomes_a_fatal_disk_full(self) -> None:
        # Continuing after ENOSPC just produces more corrupt artifacts.
        assert isinstance(download.classify("No space left on device"), errors.DiskFullError)

    def test_an_unrecognized_message_is_still_a_job_error(self) -> None:
        failure = download.classify("something entirely new")
        assert isinstance(failure, errors.DownloadError)
        assert failure.hint

    def test_the_job_id_is_attached(self) -> None:
        failure = download.classify("Private video", job_id="youtube-x")
        assert isinstance(failure, errors.JobError)
        assert failure.job_id == "youtube-x"


class TestAria2ExitCodes:
    def test_a_stall_abort_is_explained(self) -> None:
        failure = download.classify("aria2c exited with code 5")
        assert isinstance(failure, errors.Aria2Error)
        assert failure.aria2_exit_code == 5
        assert "lowest-speed-limit" in (failure.hint or "")

    def test_expired_urls_are_explained(self) -> None:
        assert "expire" in (download.classify("aria2c exited with code 3").hint or "")

    def test_a_403_suggests_fewer_connections(self) -> None:
        assert "aria2c-conns" in (download.classify("aria2c exited with code 22").hint or "")

    def test_out_of_disk_is_fatal_not_per_job(self) -> None:
        assert isinstance(download.classify("aria2c exited with code 9"), errors.DiskFullError)

    def test_an_unknown_exit_code_still_classifies(self) -> None:
        failure = download.classify("aria2c exited with code 77")
        assert isinstance(failure, errors.Aria2Error)
        assert failure.aria2_exit_code == 77


class TestExtractInfo:
    def test_returns_metadata(self) -> None:
        make = factory_for(info={"id": "abc", "title": "T"})
        assert download.extract_info("u", factory=make)["id"] == "abc"

    def test_does_not_download(self) -> None:
        make = factory_for(info={"id": "abc"})
        download.extract_info("u", factory=make)
        assert make.made[0].options["skip_download"] is True

    def test_flattens_playlists_during_enumeration(self) -> None:
        make = factory_for(info={"id": "abc"})
        download.extract_info("u", factory=make)
        assert make.made[0].options["extract_flat"] == "in_playlist"

    def test_no_metadata_raises_not_found(self) -> None:
        with pytest.raises(errors.NotFoundError):
            download.extract_info("u", factory=factory_for(info=None))

    def test_errors_are_classified(self) -> None:
        make = factory_for(raises=RuntimeError("Private video"))
        with pytest.raises(errors.PrivateVideoError):
            download.extract_info("u", factory=make)


class TestEntries:
    def test_a_single_video_is_one_entry(self) -> None:
        assert download.entries_for({"id": "a"}) == [{"id": "a"}]

    def test_a_playlist_expands(self) -> None:
        info = {"entries": [{"id": "a"}, {"id": "b"}]}
        assert [e["id"] for e in download.entries_for(info)] == ["a", "b"]

    def test_dead_entries_are_dropped(self) -> None:
        # ignoreerrors leaves a None where an unavailable video was.
        info = {"entries": [{"id": "a"}, None, {"id": "b"}]}
        assert len(download.entries_for(info)) == 2

    def test_an_empty_playlist_is_empty(self) -> None:
        assert download.entries_for({"entries": []}) == []


class TestIdentify:
    def test_uses_extractor_and_id(self) -> None:
        assert download.identify({"extractor_key": "Youtube", "id": "abc"}, "u") == "youtube-abc"

    def test_accepts_ie_key(self) -> None:
        assert download.identify({"ie_key": "Youtube", "id": "abc"}, "u") == "youtube-abc"

    def test_falls_back_to_the_url_when_unidentifiable(self) -> None:
        assert download.identify({}, "https://a.test/x").startswith("url-")


class TestFindVideo:
    def test_finds_the_media_file(self, job: paths.JobPaths) -> None:
        (job.media_dir / "video.mkv").write_bytes(b"x")
        assert download.find_video(job) == job.media_dir / "video.mkv"

    def test_ignores_part_and_control_files(self, job: paths.JobPaths) -> None:
        (job.media_dir / "video.mkv.part").write_bytes(b"x")
        (job.media_dir / "video.mkv.part.aria2").write_bytes(b"x")
        assert download.find_video(job) is None

    def test_ignores_the_info_json(self, job: paths.JobPaths) -> None:
        (job.media_dir / "video.info.json").write_text("{}")
        assert download.find_video(job) is None

    def test_none_when_nothing_downloaded(self, job: paths.JobPaths) -> None:
        assert download.find_video(job) is None

    def test_ignores_unmerged_per_format_files(self, job: paths.JobPaths) -> None:
        # A failed merge leaves video.f251.webm behind; it sorts before video.mkv
        # and is a single stream, so returning it would transcribe half the source.
        (job.media_dir / "video.f251.webm").write_bytes(b"audio only")
        assert download.find_video(job) is None


class TestClearStaleParts:
    def test_removes_a_part_with_no_control_file(self, job: paths.JobPaths) -> None:
        stale = job.media_dir / "video.mkv.part"
        stale.write_bytes(b"sparse")
        assert download.clear_stale_parts(job) == [stale]
        assert not stale.exists()

    def test_keeps_a_part_that_aria2_can_resume(self, job: paths.JobPaths) -> None:
        part = job.media_dir / "video.mkv.part"
        part.write_bytes(b"sparse")
        (job.media_dir / "video.mkv.part.aria2").write_bytes(b"ctrl")
        assert download.clear_stale_parts(job) == []
        assert part.exists()

    def test_sweeps_the_tmp_directory_too(self, job: paths.JobPaths) -> None:
        stale = job.tmp_dir / "video.mkv.part"
        stale.write_bytes(b"sparse")
        assert download.clear_stale_parts(job) == [stale]

    def test_nothing_to_do_is_not_an_error(self, job: paths.JobPaths) -> None:
        assert download.clear_stale_parts(job) == []


class TestDiscardVideo:
    def test_removes_the_merged_file(self, job: paths.JobPaths) -> None:
        video = job.media_dir / "video.mkv"
        video.write_bytes(b"truncated")
        assert download.discard_video(job) == video
        assert not video.exists()

    def test_leaves_per_format_files_alone(self, job: paths.JobPaths) -> None:
        # yt-dlp re-merges complete per-format files without re-downloading them;
        # deleting them turns a cheap remux into a full re-download.
        intermediate = job.media_dir / "video.f137.mp4"
        intermediate.write_bytes(b"video stream")
        assert download.discard_video(job) is None
        assert intermediate.exists()

    def test_nothing_to_discard(self, job: paths.JobPaths) -> None:
        assert download.discard_video(job) is None


class TestDownloadClearsStaleState:
    def test_a_corrupt_video_is_unlinked_before_yt_dlp_runs(self, job: paths.JobPaths) -> None:
        # yt-dlp's overwrites default is off: it reports an existing target as
        # "already downloaded" and skips, so a truncated file is never repaired.
        video = job.media_dir / "video.mkv"
        video.write_bytes(b"truncated")
        seen: list[bool] = []

        def on_download(_n: int) -> None:
            seen.append(video.exists())
            video.write_bytes(b"complete")

        download.download(job, "u", factory=factory_for(on_download=on_download))
        assert seen == [False]
        assert video.read_bytes() == b"complete"

    def test_an_orphan_part_is_unlinked_before_yt_dlp_runs(self, job: paths.JobPaths) -> None:
        stale = job.media_dir / "video.mkv.part"
        stale.write_bytes(b"sparse")
        seen: list[bool] = []

        def on_download(_n: int) -> None:
            seen.append(stale.exists())
            (job.media_dir / "video.mkv").write_bytes(b"data")

        download.download(job, "u", factory=factory_for(on_download=on_download))
        assert seen == [False]

    def test_a_resumable_part_survives(self, job: paths.JobPaths) -> None:
        part = job.media_dir / "video.mkv.part"
        part.write_bytes(b"sparse")
        (job.media_dir / "video.mkv.part.aria2").write_bytes(b"ctrl")
        seen: list[bool] = []

        def on_download(_n: int) -> None:
            seen.append(part.exists())
            (job.media_dir / "video.mkv").write_bytes(b"data")

        download.download(job, "u", factory=factory_for(on_download=on_download))
        assert seen == [True]

    def test_the_sweep_happens_once_not_per_attempt(self, job: paths.JobPaths) -> None:
        # aria2 writes its control file during the failed attempt; sweeping again
        # between retries would delete the very progress the retry exists to keep.
        part = job.media_dir / "video.mkv.part"
        calls: list[int] = []

        def on_download(_n: int) -> None:
            calls.append(len(calls) + 1)
            if len(calls) < 2:
                part.write_bytes(b"sparse")
                raise RuntimeError("aria2c exited with code 5")
            assert part.exists()
            (job.media_dir / "video.mkv").write_bytes(b"data")

        download.download(
            job, "u", factory=factory_for(on_download=on_download), sleep=lambda _s: None
        )
        assert len(calls) == 2


class TestDownload:
    def _succeed(self, job: paths.JobPaths) -> Any:
        def on_download(_n: int) -> None:
            (job.media_dir / "video.mkv").write_bytes(b"data")

        return factory_for(on_download=on_download)

    def test_returns_the_downloaded_file(self, job: paths.JobPaths) -> None:
        assert download.download(job, "u", factory=self._succeed(job)).name == "video.mkv"

    def test_writes_the_aria2_log_into_the_job(self, job: paths.JobPaths) -> None:
        make = self._succeed(job)
        download.download(job, "u", ytdlp_opts.DownloadConfig(use_aria2c=True), factory=make)
        args = make.made[0].options["external_downloader_args"]["aria2c"]
        assert any(str(job.logs_dir) in a for a in args)

    def test_retries_a_transient_failure(self, job: paths.JobPaths) -> None:
        # yt-dlp does not retry external downloader failures, so this loop is the
        # only thing standing between a stall-abort and a failed job.
        attempts: list[int] = []

        def on_download(_n: int) -> None:
            attempts.append(len(attempts) + 1)
            if len(attempts) < 2:
                raise RuntimeError("aria2c exited with code 5")
            (job.media_dir / "video.mkv").write_bytes(b"data")

        download.download(
            job, "u", factory=factory_for(on_download=on_download), sleep=lambda _s: None
        )
        assert len(attempts) == 2

    def test_gives_up_after_the_attempt_cap(self, job: paths.JobPaths) -> None:
        make = factory_for(raises=RuntimeError("aria2c exited with code 5"))
        with pytest.raises(errors.Aria2Error):
            download.download(job, "u", factory=make, attempts=3, sleep=lambda _s: None)
        assert len(make.made) == 3

    def test_does_not_retry_a_permanent_failure(self, job: paths.JobPaths) -> None:
        # A private video will still be private on the third try.
        make = factory_for(raises=RuntimeError("Private video"))
        with pytest.raises(errors.PrivateVideoError):
            download.download(job, "u", factory=make, sleep=lambda _s: None)
        assert len(make.made) == 1

    def test_does_not_retry_a_fatal_error(self, job: paths.JobPaths) -> None:
        make = factory_for(raises=RuntimeError("No space left on device"))
        with pytest.raises(errors.DiskFullError):
            download.download(job, "u", factory=make, sleep=lambda _s: None)
        assert len(make.made) == 1

    def test_success_without_a_file_is_an_error(self, job: paths.JobPaths) -> None:
        with pytest.raises(errors.IncompleteDownloadError):
            download.download(job, "u", factory=factory_for(), sleep=lambda _s: None)

    def test_backoff_grows(self, job: paths.JobPaths) -> None:
        delays: list[float] = []
        make = factory_for(raises=RuntimeError("timed out"))
        with pytest.raises(errors.JobError):
            download.download(job, "u", factory=make, attempts=3, backoff=2.0, sleep=delays.append)
        assert delays == [1.0, 2.0]


class TestAria2Availability:
    def test_detects_the_real_binary(self) -> None:
        assert download.aria2c_available() is True

    def test_reports_absence(self) -> None:
        assert download.aria2c_available(which=lambda _n: None) is False

    def test_missing_aria2c_downgrades_rather_than_failing(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        assert download.resolve_aria2c(requested=True, available=False) is False
        assert "native downloader" in caplog.text

    def test_available_aria2c_is_used(self) -> None:
        assert download.resolve_aria2c(requested=True, available=True) is True

    def test_opting_out_is_respected(self) -> None:
        assert download.resolve_aria2c(requested=False, available=True) is False


class TestDescribe:
    def test_extracts_the_fields_the_json_output_records(self) -> None:
        got = download.describe_entry(
            {
                "id": "a",
                "title": "T",
                "webpage_url": "u",
                "duration": 12.0,
                "uploader": "U",
                "upload_date": "20260913",
            }
        )
        assert got == {
            "id": "a",
            "title": "T",
            "url": "u",
            "duration": 12.0,
            "uploader": "U",
            "upload_date": "20260913",
        }

    def test_falls_back_to_the_flat_url(self) -> None:
        assert download.describe_entry({"url": "u"})["url"] == "u"

    def test_summarize_pluralizes(self) -> None:
        assert download.summarize([{}]) == "1 video"
        assert download.summarize([{}, {}]) == "2 videos"


def test_the_default_factory_builds_a_real_youtube_dl() -> None:
    ydl = download.default_factory({"quiet": True, "skip_download": True})
    assert hasattr(ydl, "extract_info")


class TestRetryPolicy:
    def test_network_failures_are_retried(self) -> None:
        assert download.is_retryable(download.classify("timed out")) is True

    def test_aria2_stalls_are_retried(self) -> None:
        assert download.is_retryable(download.classify("aria2c exited with code 5")) is True

    def test_permanent_source_failures_are_not(self) -> None:
        assert download.is_retryable(download.classify("Private video")) is False

    def test_fatal_errors_are_never_retried(self) -> None:
        # Continuing past ENOSPC only produces more corrupt artifacts.
        assert download.is_retryable(errors.DiskFullError("full")) is False
        assert download.is_retryable(errors.SchemaTooNewError("newer")) is False
