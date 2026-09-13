from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import ccn_transcribe.__main__ as entry
from ccn_transcribe import cli, doctor, errors, logsetup, paths
from ccn_transcribe.config import RunConfig
from ccn_transcribe.jobs import batch, store
from ccn_transcribe.jobs.plan import Stage
from ccn_transcribe.jobs.runner import JobOutcome
from ccn_transcribe.jobs.state import JobState
from ccn_transcribe.media import ytdlp_opts
from ccn_transcribe.stages import download as download_stage


class TestReadUrls:
    def test_takes_arguments(self) -> None:
        assert batch.read_urls(("a", "b"), ()) == ["a", "b"]

    def test_reads_a_batch_file(self, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text("https://a.test/1\nhttps://a.test/2\n")
        assert batch.read_urls((), (listing,)) == ["https://a.test/1", "https://a.test/2"]

    def test_skips_blank_lines_and_comments(self, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text("# a comment\n\nhttps://a.test/1\n   \n")
        assert batch.read_urls((), (listing,)) == ["https://a.test/1"]

    def test_deduplicates_while_preserving_order(self, tmp_path: Path) -> None:
        listing = tmp_path / "urls.txt"
        listing.write_text("b\na\n")
        assert batch.read_urls(("a",), (listing,)) == ["a", "b"]

    def test_reads_stdin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("https://a.test/1\n"))
        assert batch.read_urls((), (Path("-"),)) == ["https://a.test/1"]


class TestRetentionPrompt:
    def test_silent_when_nothing_is_deleted(self) -> None:
        batch.confirm_retention(RunConfig())

    def test_yes_bypasses_the_prompt(self) -> None:
        batch.confirm_retention(RunConfig(no_retain_video=True, assume_yes=True))

    def test_a_non_tty_refuses_rather_than_auto_accepting(self) -> None:
        # A piped or cron invocation must never delete gigabytes unseen.
        with pytest.raises(errors.ConfigError, match="not a terminal"):
            batch.confirm_retention(
                RunConfig(no_retain_video=True), stream=io.StringIO(), isatty=False
            )

    def test_the_refusal_names_the_escape_hatch(self) -> None:
        try:
            batch.confirm_retention(
                RunConfig(no_retain_video=True), stream=io.StringIO(), isatty=False
            )
        except errors.ConfigError as exc:
            assert "--yes" in (exc.hint or "")

    def test_accepting_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
        batch.confirm_retention(RunConfig(no_retain_video=True), stream=io.StringIO(), isatty=True)

    def test_declining_aborts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO("n\n"))
        with pytest.raises(errors.ConfigError, match="aborted"):
            batch.confirm_retention(
                RunConfig(no_retain_video=True), stream=io.StringIO(), isatty=True
            )

    def test_the_warning_is_accurate_about_resume(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Resume genuinely still works; the real cost is re-run flexibility.
        monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))
        out = io.StringIO()
        batch.confirm_retention(RunConfig(no_retain_audio=True), stream=out, isatty=True)
        text = out.getvalue()
        assert "resume always works" in text
        assert "re-run flexibility" in text


class TestBatchResult:
    def test_counts_and_exit_code(self) -> None:
        result = batch.BatchResult(
            [
                JobOutcome("a", "u", (), ok=True),
                JobOutcome("b", "u", (), ok=False, error="nope"),
            ]
        )
        assert len(result.ok) == 1
        assert len(result.failed) == 1
        assert result.exit_code == errors.EXIT_JOB_FAILED
        assert "1 ok, 1 failed" in result.summary()

    def test_all_ok_exits_zero(self) -> None:
        assert batch.BatchResult([JobOutcome("a", "u", (), ok=True)]).exit_code == 0

    def test_failures_are_listed(self) -> None:
        result = batch.BatchResult([JobOutcome("b", "u", (), ok=False, error="boom")])
        assert "FAILED b: boom" in result.summary()


class TestExpand:
    def _workspace(self, tmp_path: Path) -> paths.Workspace:
        return paths.Workspace(tmp_path / "transcripts")

    def _extractor(
        self, monkeypatch: pytest.MonkeyPatch, doc: dict[str, Any], calls: list[int] | None = None
    ) -> None:
        def info(*_a: object, **_k: object) -> dict[str, Any]:
            if calls is not None:
                calls.append(1)
            return doc

        monkeypatch.setattr(download_stage, "extract_info", info)

    def test_a_single_video_is_one_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._extractor(monkeypatch, {"id": "a", "webpage_url": "https://y.test/a"})
        targets = batch.expand("u", RunConfig(), self._workspace(tmp_path))
        assert [t.url for t in targets] == ["https://y.test/a"]
        assert targets[0].entry == {"id": "a", "webpage_url": "https://y.test/a"}

    def test_a_playlist_becomes_many(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._extractor(
            monkeypatch, {"entries": [{"id": "a", "url": "u1"}, {"id": "b", "url": "u2"}]}
        )
        targets = batch.expand("u", RunConfig(), self._workspace(tmp_path))
        assert [t.url for t in targets] == ["u1", "u2"]

    def test_falls_back_to_the_original_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._extractor(monkeypatch, {"id": "a"})
        assert batch.expand("orig", RunConfig(), self._workspace(tmp_path))[0].url == "orig"

    def test_a_known_single_video_needs_no_metadata_request(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The whole point of the index: resuming one video is offline and instant.
        workspace = self._workspace(tmp_path)
        calls: list[int] = []
        self._extractor(monkeypatch, {"id": "a", "webpage_url": "https://y.test/a"}, calls)
        first = batch.expand("u", RunConfig(), workspace)
        second = batch.expand("u", RunConfig(), workspace)
        assert len(calls) == 1
        assert [t.job_id for t in second] == [t.job_id for t in first]

    def test_a_known_playlist_is_still_re_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Skipping the request would silently ignore videos added since last time.
        workspace = self._workspace(tmp_path)
        calls: list[int] = []
        self._extractor(
            monkeypatch, {"entries": [{"id": "a", "url": "u1"}, {"id": "b", "url": "u2"}]}, calls
        )
        batch.expand("u", RunConfig(), workspace)
        batch.expand("u", RunConfig(), workspace)
        assert len(calls) == 2

    def test_a_dry_run_records_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = self._workspace(tmp_path)
        self._extractor(monkeypatch, {"id": "a", "webpage_url": "https://y.test/a"})
        batch.expand("u", RunConfig(), workspace, remember=False)
        assert not workspace.index_file.exists()

    def test_an_unreachable_source_resumes_what_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workspace = self._workspace(tmp_path)
        self._extractor(
            monkeypatch, {"entries": [{"id": "a", "url": "u1"}, {"id": "b", "url": "u2"}]}
        )
        batch.expand("u", RunConfig(), workspace)

        def offline(*_a: object, **_k: object) -> dict[str, Any]:
            raise errors.NetworkError("no route to host")

        monkeypatch.setattr(download_stage, "extract_info", offline)
        assert [t.url for t in batch.expand("u", RunConfig(), workspace)] == ["u1", "u2"]

    def test_an_unreachable_unknown_source_still_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def offline(*_a: object, **_k: object) -> dict[str, Any]:
            raise errors.NetworkError("no route to host")

        monkeypatch.setattr(download_stage, "extract_info", offline)
        with pytest.raises(errors.NetworkError):
            batch.expand("u", RunConfig(), self._workspace(tmp_path))

    def test_an_unreadable_source_of_any_kind_uses_the_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A dead proxy surfaces as "no metadata", not as a network error, so the
        # fallback keys on the source being unreadable rather than on which way.
        workspace = self._workspace(tmp_path)
        self._extractor(
            monkeypatch, {"entries": [{"id": "a", "url": "u1"}, {"id": "b", "url": "u2"}]}
        )
        batch.expand("u", RunConfig(), workspace)

        def gone(*_a: object, **_k: object) -> dict[str, Any]:
            raise errors.NotFoundError("no metadata for u")

        monkeypatch.setattr(download_stage, "extract_info", gone)
        assert [t.url for t in batch.expand("u", RunConfig(), workspace)] == ["u1", "u2"]

    def test_an_unknown_url_is_never_invented(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Nothing recorded means nothing to fall back to, so a typo still fails.
        def gone(*_a: object, **_k: object) -> dict[str, Any]:
            raise errors.NotFoundError("no metadata for typo")

        monkeypatch.setattr(download_stage, "extract_info", gone)
        with pytest.raises(errors.NotFoundError):
            batch.expand("typo", RunConfig(), self._workspace(tmp_path))


class TestRunBatch:
    def _config(self, tmp_path: Path) -> RunConfig:
        return RunConfig(workdir=tmp_path / "transcripts")

    def test_one_bad_url_does_not_stop_the_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def info(url: str, *_a: object, **_k: object) -> dict[str, Any]:
            if "bad" in url:
                raise errors.NotFoundError("Video unavailable")
            return {"id": "good", "extractor_key": "Youtube", "webpage_url": url}

        monkeypatch.setattr(download_stage, "extract_info", info)

        def fake_run(job: Any, url: str, *_a: object, **_k: object) -> JobOutcome:
            return JobOutcome(job.root.name, url, (), ok=True)

        monkeypatch.setattr("ccn_transcribe.jobs.batch.run_job", fake_run)
        result = batch.run_batch(
            ["https://y.test/bad", "https://y.test/good"], self._config(tmp_path)
        )
        assert len(result.ok) == 1
        assert len(result.failed) == 1
        assert result.exit_code == 1

    def test_a_fatal_error_stops_everything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def info(*_a: object, **_k: object) -> dict[str, Any]:
            raise errors.DiskFullError("no space")

        monkeypatch.setattr(download_stage, "extract_info", info)
        with pytest.raises(errors.DiskFullError):
            batch.run_batch(["u"], self._config(tmp_path))

    def test_a_job_error_is_captured_per_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def info(url: str, *_a: object, **_k: object) -> dict[str, Any]:
            return {"id": "x", "extractor_key": "Youtube", "webpage_url": url}

        def boom(*_a: object, **_k: object) -> None:
            raise errors.MediaError("no audio")

        monkeypatch.setattr(download_stage, "extract_info", info)
        monkeypatch.setattr("ccn_transcribe.jobs.batch.run_job", boom)
        result = batch.run_batch(["u"], self._config(tmp_path))
        assert result.failed[0].error == "no audio"


class TestLogSetup:
    def test_configure_is_idempotent(self) -> None:
        logsetup.configure()
        logsetup.configure(verbose=1)
        assert len(logging.getLogger(logsetup.ROOT).handlers) == 1

    def test_quiet_raises_the_level(self) -> None:
        logsetup.configure(quiet=True)
        assert logging.getLogger(logsetup.ROOT).level == logging.WARNING

    def test_job_log_writes_and_detaches(self, tmp_path: Path) -> None:
        logsetup.configure()
        job = paths.Workspace(tmp_path).job("youtube-x")
        job.ensure()
        before = len(logging.getLogger(logsetup.ROOT).handlers)
        with logsetup.job_log(job):
            logging.getLogger("ccn_transcribe.test").info("hello")
        assert "hello" in (job.logs_dir / "job.log").read_text()
        # A 200-URL batch must not leak 200 open handlers.
        assert len(logging.getLogger(logsetup.ROOT).handlers) == before


class TestDoctor:
    def test_every_check_has_a_name_and_status(self) -> None:
        for check in doctor.run_checks():
            assert check.name
            assert check.status in doctor.Status

    def test_this_machine_passes(self) -> None:
        assert doctor.worst(doctor.run_checks()) == 0

    def test_renders_hints_only_for_problems(self) -> None:
        checks = [
            doctor.Check("good", doctor.Status.PASS, "fine", "unused hint"),
            doctor.Check("bad", doctor.Status.FAIL, "broken", "fix it"),
        ]
        text = doctor.render(checks)
        assert "unused hint" not in text
        assert "fix it" in text

    def test_failures_are_counted(self) -> None:
        checks = [doctor.Check("bad", doctor.Status.FAIL, "broken")]
        assert "1 check(s) FAILED" in doctor.render(checks)
        assert doctor.worst(checks) == 2

    def test_detects_the_intel_gpu_here(self) -> None:
        names = {c.name: c for c in doctor.run_checks()}
        assert "openvino devices" in names
        assert "GPU" in names["openvino devices"].detail

    def test_the_js_runtime_check_passes_with_the_venv_deno(self) -> None:
        assert doctor.check_js_runtime().status is doctor.Status.PASS

    def test_a_world_writable_render_node_warns(self) -> None:
        # It works today only because the node is 0666; say so.
        statuses = {c.name: c.status for c in doctor.check_render_nodes()}
        assert any(n.startswith("permission") for n in statuses)


class TestCli:
    def test_help(self) -> None:
        result = CliRunner().invoke(cli.main, ["--help"])
        assert result.exit_code == 0
        assert "doctor" in result.output

    def test_doctor_runs(self) -> None:
        result = CliRunner().invoke(cli.main, ["doctor"])
        assert result.exit_code == 0

    def test_run_without_urls_is_an_error(self) -> None:
        result = CliRunner().invoke(cli.main, ["run"])
        assert result.exit_code == errors.EXIT_FATAL
        assert "no URLs" in result.output

    def test_an_unknown_format_is_rejected(self) -> None:
        result = CliRunner().invoke(cli.main, ["run", "u", "--formats", "docx"])
        assert result.exit_code == errors.EXIT_FATAL
        assert "unknown output format" in result.output

    def test_no_retain_on_a_non_tty_refuses(self) -> None:
        result = CliRunner().invoke(cli.main, ["run", "u", "--no-retain"])
        assert result.exit_code == errors.EXIT_FATAL
        assert "--yes" in result.output

    def test_status_with_no_jobs(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(cli.main, ["status", "-w", str(tmp_path)])
        assert "no jobs yet" in result.output

    def test_status_reports_a_job(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-abc")
        job.ensure()
        state = JobState.new(job_id="youtube-abc", source_url="u")
        state.title = "A Title"
        store.save(job, state)
        result = CliRunner().invoke(cli.main, ["status", "-w", str(tmp_path)])
        assert "youtube-abc" in result.output
        assert "A Title" in result.output

    def test_status_for_a_job_without_state(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-bare")
        job.ensure()
        result = CliRunner().invoke(cli.main, ["status", "-w", str(tmp_path)])
        assert "no state recorded" in result.output

    def test_parse_formats_defaults_when_empty(self) -> None:
        assert cli.parse_formats("") == ("txt", "srt", "vtt", "json")

    def test_parse_formats_trims(self) -> None:
        assert cli.parse_formats(" txt , srt ") == ("txt", "srt")


class TestCliWiring:
    def _kw(self, tmp_path: Path, **over: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "workdir": tmp_path / "transcripts",
            "out_dir": None,
            "formats": "txt",
            "device": None,
            "model": None,
            "language": None,
            "task": "transcribe",
            "num_beams": 1,
            "initial_prompt": None,
            "audio_profile": "source",
            "playlist": False,
            "max_height": 1080,
            "no_aria2c": False,
            "aria2c_conns": 16,
            "aria2c_lowest_speed": "50K",
            "aria2c_arg": (),
            "concurrent_fragments": 16,
            "retries": 5,
            "cookies_from_browser": None,
            "js_runtime": None,
            "ov_cache_dir": None,
            "no_retain": False,
            "no_retain_video": False,
            "no_retain_audio": False,
            "yes": False,
            "force": False,
            "force_stage": (),
            "dry_run": False,
            "verbose": 0,
            "quiet": False,
        }
        base.update(over)
        return base

    def test_force_selects_every_stage(self, tmp_path: Path) -> None:
        config = cli.build_config(**self._kw(tmp_path, force=True))
        assert config.force == frozenset(Stage)

    def test_force_stage_selects_one(self, tmp_path: Path) -> None:
        config = cli.build_config(**self._kw(tmp_path, force_stage=("audio",)))
        assert config.force == frozenset({Stage.AUDIO})

    def test_no_retain_implies_both(self, tmp_path: Path) -> None:
        config = cli.build_config(**self._kw(tmp_path, no_retain=True))
        assert config.no_retain_video is True
        assert config.no_retain_audio is True

    def test_aria2c_is_enabled_when_available(self, tmp_path: Path) -> None:
        assert cli.build_config(**self._kw(tmp_path)).download.use_aria2c is True

    def test_no_aria2c_opts_out(self, tmp_path: Path) -> None:
        config = cli.build_config(**self._kw(tmp_path, no_aria2c=True))
        assert config.download.use_aria2c is False

    def test_file_allocation_matches_the_filesystem(self, tmp_path: Path) -> None:
        expected = ytdlp_opts.allocation_for(ytdlp_opts.filesystem_type(tmp_path))
        assert cli.build_config(**self._kw(tmp_path)).download.file_allocation == expected

    def test_the_repo_filesystem_gets_falloc(self) -> None:
        # /home here is btrfs, where fallocate is instant.
        assert ytdlp_opts.allocation_for(ytdlp_opts.filesystem_type(Path.cwd())) == "falloc"


class TestCliRun:
    def test_a_successful_run_reports_and_exits_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_batch(*_a: object, **_k: object) -> batch.BatchResult:
            return batch.BatchResult([JobOutcome("youtube-a", "u", (), ok=True)])

        monkeypatch.setattr(batch, "run_batch", fake_batch)
        result = CliRunner().invoke(cli.main, ["run", "https://y.test/a", "-w", str(tmp_path)])
        assert result.exit_code == 0
        assert "1 ok, 0 failed" in result.output

    def test_a_failed_job_exits_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_batch(*_a: object, **_k: object) -> batch.BatchResult:
            return batch.BatchResult([JobOutcome("youtube-a", "u", (), ok=False, error="x")])

        monkeypatch.setattr(batch, "run_batch", fake_batch)
        result = CliRunner().invoke(cli.main, ["run", "https://y.test/a", "-w", str(tmp_path)])
        assert result.exit_code == errors.EXIT_JOB_FAILED

    def test_verbose_re_raises_for_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise errors.ConfigError("bad config", hint="fix it")

        monkeypatch.setattr(batch, "run_batch", boom)
        result = CliRunner().invoke(cli.main, ["run", "u", "-w", str(tmp_path), "-v"])
        assert isinstance(result.exception, errors.ConfigError)

    def test_the_hint_is_printed(self) -> None:
        result = CliRunner().invoke(cli.main, ["run", "u", "--formats", "docx"])
        assert "hint:" in result.output


def test_a_fatal_error_inside_a_job_stops_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Continuing after ENOSPC would only produce more corrupt artifacts.
    def info(url: str, *_a: object, **_k: object) -> dict[str, Any]:
        return {"id": "x", "extractor_key": "Youtube", "webpage_url": url}

    def boom(*_a: object, **_k: object) -> None:
        raise errors.DiskFullError("no space")

    monkeypatch.setattr(download_stage, "extract_info", info)
    monkeypatch.setattr("ccn_transcribe.jobs.batch.run_job", boom)
    with pytest.raises(errors.DiskFullError):
        batch.run_batch(["u"], RunConfig(workdir=tmp_path / "t"))


def test_an_error_without_a_hint_prints_only_the_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise errors.ConfigError("bare failure")

    monkeypatch.setattr(batch, "run_batch", boom)
    result = CliRunner().invoke(cli.main, ["run", "u", "-w", str(tmp_path)])
    assert "ERROR: bare failure" in result.output
    assert "hint:" not in result.output


def test_module_entrypoint_is_the_cli() -> None:
    # python -m ccn_transcribe must reach the same command group, and the
    # __main__ guard must be present: Python 3.14's forkserver re-imports
    # __main__, and without the guard the program runs a second copy of itself.
    assert entry.main is cli.main
    source = Path(entry.__file__).read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in source
