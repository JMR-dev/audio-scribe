from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from ccn_transcribe import paths
from ccn_transcribe.jobs import index

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def workspace(tmp_path: Path) -> paths.Workspace:
    return paths.Workspace(tmp_path)


def _one(job_id: str = "youtube-abc", url: str = "https://a.test/v") -> list[index.Target]:
    return [index.Target(job_id=job_id, url=url)]


class TestRoundTrip:
    def test_nothing_recorded_is_a_miss(self, workspace: paths.Workspace) -> None:
        assert index.lookup(workspace, "https://a.test/v", playlist=False) is None

    def test_a_recorded_url_comes_back(self, workspace: paths.Workspace) -> None:
        index.record(workspace, "https://a.test/v", playlist=False, targets=_one())
        assert index.lookup(workspace, "https://a.test/v", playlist=False) == _one()

    def test_url_spellings_that_name_one_resource_share_an_entry(
        self, workspace: paths.Workspace
    ) -> None:
        index.record(workspace, "HTTPS://A.test/v", playlist=False, targets=_one())
        assert index.lookup(workspace, "https://a.test/v#frag", playlist=False) == _one()

    def test_a_playlist_entry_does_not_answer_a_single_video_lookup(
        self, workspace: paths.Workspace
    ) -> None:
        # The same URL means different things with and without --playlist, and
        # answering one from the other would silently transcribe the wrong set.
        index.record(workspace, "https://a.test/v", playlist=True, targets=_one())
        assert index.lookup(workspace, "https://a.test/v", playlist=False) is None

    def test_re_recording_replaces_the_entry(self, workspace: paths.Workspace) -> None:
        index.record(workspace, "https://a.test/v", playlist=False, targets=_one("old", "u1"))
        index.record(workspace, "https://a.test/v", playlist=False, targets=_one("new", "u2"))
        assert index.lookup(workspace, "https://a.test/v", playlist=False) == _one("new", "u2")

    def test_entries_for_other_urls_survive(self, workspace: paths.Workspace) -> None:
        index.record(workspace, "https://a.test/1", playlist=False, targets=_one("j1", "u1"))
        index.record(workspace, "https://a.test/2", playlist=False, targets=_one("j2", "u2"))
        assert index.lookup(workspace, "https://a.test/1", playlist=False) == _one("j1", "u1")

    def test_each_job_keeps_its_own_url(self, workspace: paths.Workspace) -> None:
        # A playlist entry must not hand every job the playlist URL: downloading
        # from it would fetch the wrong video.
        targets = [
            index.Target(job_id="j1", url="https://a.test/1"),
            index.Target(job_id="j2", url="https://a.test/2"),
        ]
        index.record(workspace, "https://a.test/list", playlist=True, targets=targets)
        assert index.lookup(workspace, "https://a.test/list", playlist=True) == targets

    def test_the_entry_metadata_is_not_persisted(self, workspace: paths.Workspace) -> None:
        rich = [index.Target(job_id="j1", url="u1", entry={"title": "T"})]
        index.record(workspace, "https://a.test/v", playlist=False, targets=rich)
        found = index.lookup(workspace, "https://a.test/v", playlist=False) or []
        assert [t.entry for t in found] == [None]


class TestDamagedIndex:
    def test_unreadable_json_is_a_miss_not_a_crash(self, workspace: paths.Workspace) -> None:
        workspace.index_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.index_file.write_text("{ truncated", encoding="utf-8")
        assert index.lookup(workspace, "https://a.test/v", playlist=False) is None

    def test_a_damaged_index_is_replaced_on_the_next_write(
        self, workspace: paths.Workspace
    ) -> None:
        workspace.index_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.index_file.write_text("[]", encoding="utf-8")
        index.record(workspace, "https://a.test/v", playlist=False, targets=_one())
        assert index.lookup(workspace, "https://a.test/v", playlist=False) == _one()

    def test_an_entry_missing_its_jobs_is_a_miss(self, workspace: paths.Workspace) -> None:
        workspace.index_file.parent.mkdir(parents=True, exist_ok=True)
        doc = {"schema_version": 1, "urls": {"https://a.test/v": {"playlist": False}}}
        workspace.index_file.write_text(json.dumps(doc), encoding="utf-8")
        assert index.lookup(workspace, "https://a.test/v", playlist=False) is None

    def test_a_malformed_job_row_is_skipped(self, workspace: paths.Workspace) -> None:
        workspace.index_file.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "schema_version": 1,
            "urls": {
                "https://a.test/v": {
                    "playlist": False,
                    "jobs": [{"id": "good", "url": "u"}, {"id": "no url"}, "junk"],
                }
            },
        }
        workspace.index_file.write_text(json.dumps(doc), encoding="utf-8")
        assert index.lookup(workspace, "https://a.test/v", playlist=False) == _one("good", "u")

    def test_a_future_schema_is_ignored_rather_than_guessed_at(
        self, workspace: paths.Workspace
    ) -> None:
        # The index is a cache, so a version we cannot read costs a metadata
        # request rather than a wrong answer.
        workspace.index_file.parent.mkdir(parents=True, exist_ok=True)
        rows = [{"id": "j", "url": "u"}]
        doc = {"schema_version": 99, "urls": {"https://a.test/v": {"jobs": rows}}}
        workspace.index_file.write_text(json.dumps(doc), encoding="utf-8")
        assert index.lookup(workspace, "https://a.test/v", playlist=False) is None
