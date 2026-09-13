from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ccn_transcribe import paths

if TYPE_CHECKING:
    from pathlib import Path


class TestJobId:
    def test_combines_extractor_and_video_id(self) -> None:
        assert paths.job_id("Youtube", "dQw4w9WgXcQ") == "youtube-dQw4w9WgXcQ"

    def test_extractor_key_is_lowercased(self) -> None:
        # Extractor-key casing has changed across yt-dlp releases; folding it
        # keeps a version bump from orphaning existing job directories.
        assert paths.job_id("YouTube", "x") == paths.job_id("youtube", "x")

    def test_extractor_punctuation_is_stripped(self) -> None:
        assert paths.job_id("Some:Site!", "x").startswith("somesite-")

    def test_empty_extractor_falls_back_to_generic(self) -> None:
        assert paths.job_id("", "x").startswith("generic-")

    def test_unsafe_characters_are_replaced(self) -> None:
        assert "/" not in paths.job_id("youtube", "a/b")

    def test_ids_that_sanitize_alike_stay_distinct(self) -> None:
        # "a/b" and "a:b" both sanitize to "a_b"; without a digest suffix they
        # would share one job directory and corrupt each other's state.
        assert paths.job_id("youtube", "a/b") != paths.job_id("youtube", "a:b")

    def test_clean_ids_get_no_digest_suffix(self) -> None:
        assert paths.job_id("youtube", "abc-123_x.y") == "youtube-abc-123_x.y"

    def test_very_long_ids_are_truncated_but_stay_distinct(self) -> None:
        long_a = "z" * 200 + "a"
        long_b = "z" * 200 + "b"
        assert len(paths.job_id("youtube", long_a)) < 120
        assert paths.job_id("youtube", long_a) != paths.job_id("youtube", long_b)

    def test_is_deterministic(self) -> None:
        assert paths.job_id("youtube", "a/b") == paths.job_id("youtube", "a/b")

    @pytest.mark.parametrize("hostile", ["..", "../..", "../../etc/passwd", "/etc/passwd"])
    def test_cannot_escape_the_jobs_directory(self, hostile: str, tmp_path: Path) -> None:
        jid = paths.job_id("youtube", hostile)
        resolved = (tmp_path / "jobs" / jid).resolve()
        assert resolved.parent == (tmp_path / "jobs").resolve()


class TestUrlJobId:
    def test_is_stable(self) -> None:
        url = "https://example.com/a.mp4"
        assert paths.url_job_id(url) == paths.url_job_id(url)

    def test_differs_between_urls(self) -> None:
        assert paths.url_job_id("https://a.test/x") != paths.url_job_id("https://a.test/y")

    def test_is_filesystem_safe(self) -> None:
        jid = paths.url_job_id("https://a.test/x?q=1&r=2#frag")
        assert jid.startswith("url-")
        assert all(c.isalnum() or c in "-_." for c in jid)


class TestNormalizeUrl:
    def test_strips_surrounding_whitespace(self) -> None:
        assert paths.normalize_url("  https://a.test/x  ") == "https://a.test/x"

    def test_drops_the_fragment(self) -> None:
        assert paths.normalize_url("https://a.test/x#t=30") == "https://a.test/x"

    def test_keeps_the_query(self) -> None:
        # YouTube identifies the video in the query string.
        assert paths.normalize_url("https://y.test/watch?v=abc") == "https://y.test/watch?v=abc"

    def test_lowercases_scheme_and_host_only(self) -> None:
        assert paths.normalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_equivalent_urls_map_to_one_job(self) -> None:
        a = paths.normalize_url("https://a.test/x#one")
        b = paths.normalize_url(" https://a.test/x#two ")
        assert paths.url_job_id(a) == paths.url_job_id(b)


class TestWorkspace:
    def test_job_directory_lives_under_jobs(self, tmp_path: Path) -> None:
        ws = paths.Workspace(tmp_path)
        assert ws.job("youtube-x").root == tmp_path / "jobs" / "youtube-x"

    def test_index_file_location(self, tmp_path: Path) -> None:
        assert paths.Workspace(tmp_path).index_file == tmp_path / "index.json"

    def test_ensure_creates_the_tree(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-x")
        job.ensure()
        for d in (job.media_dir, job.out_dir, job.logs_dir, job.tmp_dir):
            assert d.is_dir()

    def test_ensure_is_idempotent(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-x")
        job.ensure()
        job.ensure()
        assert job.media_dir.is_dir()

    def test_artifact_locations(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-x")
        assert job.state_file.name == "state.json"
        assert job.lock_file.name == ".lock"
        assert job.events_file.name == "events.jsonl"
        assert job.info_file.name == "info.json"
        assert job.audio_file == job.media_dir / "audio.flac"

    def test_tmp_is_inside_the_job_dir(self, tmp_path: Path) -> None:
        # os.replace is only atomic within one filesystem.
        job = paths.Workspace(tmp_path).job("youtube-x")
        assert job.tmp_dir.parent == job.root

    def test_transcript_paths_are_named_per_format(self, tmp_path: Path) -> None:
        job = paths.Workspace(tmp_path).job("youtube-x")
        assert job.transcript_file("srt") == job.out_dir / "transcript.srt"
