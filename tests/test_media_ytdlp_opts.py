from __future__ import annotations

import shutil
import sys
from typing import TYPE_CHECKING

import pytest

from ccn_transcribe.media import ytdlp_opts as opts

if TYPE_CHECKING:
    from pathlib import Path


class TestAria2cArgs:
    def test_sets_a_retry_wait(self) -> None:
        # aria2's default is 0, so it burns all five tries in under a second
        # against a struggling server.
        assert "--retry-wait=3" in opts.aria2c_args()

    def test_enables_the_stall_abort(self) -> None:
        # --lowest-speed-limit defaults to 0, meaning a wedged connection hangs
        # forever. This is the single biggest real-world hang fix.
        assert any(a.startswith("--lowest-speed-limit=") for a in opts.aria2c_args())

    def test_stall_abort_can_be_disabled_for_genuinely_slow_links(self) -> None:
        assert "--lowest-speed-limit=0" in opts.aria2c_args(lowest_speed="0")

    def test_checkpoints_progress_often_enough_to_resume(self) -> None:
        # aria2 saves the control file every 60s by default, so a hard kill in the
        # first minute leaves a control file recording zero completed pieces and
        # the "resume" silently re-downloads the lot. Measured: 0/13 pieces.
        assert "--auto-save-interval=20" in opts.aria2c_args()

    def test_the_checkpoint_interval_is_tunable(self) -> None:
        assert "--auto-save-interval=5" in opts.aria2c_args(auto_save=5)

    def test_raises_the_disk_cache(self) -> None:
        assert "--disk-cache=64M" in opts.aria2c_args()

    def test_shortens_the_timeouts(self) -> None:
        args = opts.aria2c_args()
        assert "--timeout=30" in args
        assert "--connect-timeout=15" in args

    def test_writes_a_log_when_asked(self, tmp_path: Path) -> None:
        args = opts.aria2c_args(log_file=tmp_path / "aria2.log")
        assert f"--log={tmp_path / 'aria2.log'}" in args
        assert "--log-level=notice" in args

    def test_omits_the_log_when_not_asked(self) -> None:
        assert not any(a.startswith("--log=") for a in opts.aria2c_args())

    def test_never_passes_continue(self) -> None:
        # aria2 writes segments out of order, so --continue would resume from the
        # file length over a sparse file and silently corrupt the download.
        args = opts.aria2c_args()
        assert not any(a.startswith(("--continue", "-c")) for a in args)

    def test_does_not_repeat_flags_yt_dlp_already_locks(self) -> None:
        # These are appended after user args and cannot be overridden; passing
        # them is a silent no-op that only confuses the next reader.
        args = " ".join(opts.aria2c_args())
        for locked in (
            "--allow-overwrite",
            "--always-resume",
            "--auto-file-renaming",
            "--force-save",
        ):
            assert locked not in args

    def test_does_not_restate_aria2_defaults(self) -> None:
        # --max-tries=5 is already aria2's default; yt-dlp does not set it.
        assert not any(a.startswith("--max-tries") for a in opts.aria2c_args())

    def test_connection_count_is_only_sent_when_it_differs(self) -> None:
        # -x16 is already yt-dlp's default and aria2's per-server cap.
        assert not any(a.startswith("-x") for a in opts.aria2c_args())
        assert "-x4" in opts.aria2c_args(connections=4)
        assert "-s4" in opts.aria2c_args(connections=4)

    def test_user_arguments_sort_last_so_they_win(self) -> None:
        # aria2 takes the last occurrence of a repeated flag.
        args = opts.aria2c_args(extra=["--disk-cache=8M"])
        assert args[-1] == "--disk-cache=8M"

    def test_file_allocation_is_falloc_on_supporting_filesystems(self) -> None:
        assert "--file-allocation=falloc" in opts.aria2c_args(file_allocation="falloc")

    def test_file_allocation_falls_back_to_none(self) -> None:
        assert "--file-allocation=none" in opts.aria2c_args(file_allocation="none")


class TestAllocationDetection:
    @pytest.mark.parametrize("fstype", ["btrfs", "ext4", "xfs"])
    def test_supported_filesystems_get_falloc(self, fstype: str) -> None:
        assert opts.allocation_for(fstype) == "falloc"

    @pytest.mark.parametrize("fstype", ["nfs", "tmpfs", "vfat", "unknown"])
    def test_other_filesystems_fall_back(self, fstype: str) -> None:
        assert opts.allocation_for(fstype) == "none"

    def test_detects_this_machines_filesystem(self, tmp_path: Path) -> None:
        # /home here is btrfs, which is exactly why falloc matters.
        assert opts.filesystem_type(tmp_path) is not None


class TestYdlOpts:
    def test_remuxes_to_mkv_rather_than_re_encoding(self) -> None:
        # This ffmpeg build has no x264/x265 encoders; mkv accepts any codec pair
        # so the merge stays a stream copy.
        assert opts.ydl_opts("x")["merge_output_format"] == "mkv"

    def test_resumes_partial_downloads(self) -> None:
        assert opts.ydl_opts("x")["continuedl"] is True

    def test_writes_the_info_json(self) -> None:
        assert opts.ydl_opts("x")["writeinfojson"] is True

    def test_maps_aria2c_to_the_protocols_it_actually_supports(self) -> None:
        # Aria2cFD.SUPPORTED_PROTOCOLS is http/https/ftp/ftps only; an explicit
        # map avoids relying on yt-dlp's silent fallback for HLS and DASH.
        got = opts.ydl_opts("x", opts.DownloadConfig(use_aria2c=True))["external_downloader"]
        assert set(got) == {"http", "https", "ftp", "ftps"}
        assert set(got.values()) == {"aria2c"}

    def test_omits_the_external_downloader_when_aria2c_is_absent(self) -> None:
        assert "external_downloader" not in opts.ydl_opts(
            "x", opts.DownloadConfig(use_aria2c=False)
        )

    def test_raises_concurrent_fragments_for_hls_where_aria2c_cannot_help(self) -> None:
        assert opts.ydl_opts("x")["concurrent_fragment_downloads"] == 16

    def test_does_not_expand_a_playlist_by_default(self) -> None:
        # A video URL carrying &list= should transcribe that one video.
        assert opts.ydl_opts("x")["noplaylist"] is True

    def test_playlist_expansion_is_opt_in(self) -> None:
        assert opts.ydl_opts("x", opts.DownloadConfig(playlist=True))["noplaylist"] is False

    def test_height_cap_is_applied_to_the_format_selector(self) -> None:
        assert "height<=720" in opts.ydl_opts("x", opts.DownloadConfig(max_height=720))["format"]

    def test_format_prefers_progressive_https_for_aria2c_eligibility(self) -> None:
        assert "proto:https" in " ".join(opts.ydl_opts("x")["format_sort"])

    def test_cookies_from_browser_is_passed_through(self) -> None:
        got = opts.ydl_opts("x", opts.DownloadConfig(cookies_from_browser="firefox"))
        assert got["cookiesfrombrowser"] == ("firefox",)

    def test_cookies_absent_when_not_requested(self) -> None:
        assert "cookiesfrombrowser" not in opts.ydl_opts("x")

    def test_aria2c_args_are_attached_under_the_executable_name(self, tmp_path: Path) -> None:
        got = opts.ydl_opts("x", opts.DownloadConfig(use_aria2c=True), aria2_log=tmp_path / "a.log")
        assert "aria2c" in got["external_downloader_args"]
        assert any("--retry-wait" in a for a in got["external_downloader_args"]["aria2c"])


class TestJsRuntime:
    def test_defaults_to_deno_which_ships_in_the_venv(self) -> None:
        # yt-dlp's own default is {'deno': {}}, and the deno extra puts the binary
        # in the venv, so cron sees it without any PATH setup.
        assert "deno" in opts.js_runtime_spec()

    def test_an_explicit_runtime_is_honoured(self) -> None:
        assert "node" in opts.js_runtime_spec("node")

    def test_node_is_resolved_to_an_absolute_path_when_found(self) -> None:
        spec = opts.js_runtime_spec("node")
        path = spec["node"].get("path")
        if path is not None:
            assert path.startswith("/")

    def test_an_unknown_runtime_is_rejected(self) -> None:
        with pytest.raises(Exception, match="runtime"):
            opts.js_runtime_spec("rhino")

    def test_the_runtime_is_wired_into_ydl_opts(self) -> None:
        assert "js_runtimes" in opts.ydl_opts("x")


class TestBundledRuntime:
    def _exe(self, prefix: Path, name: str) -> Path:
        binary = prefix / "bin" / name
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
        return binary

    def test_finds_an_executable_beside_the_interpreter(self, tmp_path: Path) -> None:
        # The deno extra installs here. When the package is installed as a tool
        # this directory is not on PATH -- only the console script is exposed.
        deno = self._exe(tmp_path, "deno")
        assert opts.bundled("deno", tmp_path) == str(deno)

    def test_absent_is_none(self, tmp_path: Path) -> None:
        assert opts.bundled("deno", tmp_path) is None

    def test_a_non_executable_does_not_count(self, tmp_path: Path) -> None:
        binary = tmp_path / "bin" / "deno"
        binary.parent.mkdir(parents=True)
        binary.write_text("not executable")
        binary.chmod(0o644)
        assert opts.bundled("deno", tmp_path) is None

    def test_bundled_beats_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        deno = self._exe(tmp_path, "deno")

        def on_path(_n: str, _mode: int = 0, _path: str | None = None) -> str:
            return "/usr/bin/deno"

        monkeypatch.setattr(shutil, "which", on_path)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert opts.find_runtime("deno") == str(deno)

    def test_falls_back_to_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def on_path(_n: str, _mode: int = 0, _path: str | None = None) -> str:
            return "/usr/bin/deno"

        monkeypatch.setattr(shutil, "which", on_path)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert opts.find_runtime("deno") == "/usr/bin/deno"

    def test_node_keeps_its_nvm_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def found(_h: Path | None = None) -> str:
            return "/nvm/node"

        monkeypatch.setattr(opts, "find_node", found)
        assert opts.find_runtime("node") == "/nvm/node"

    def test_the_spec_carries_the_resolved_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        deno = self._exe(tmp_path, "deno")
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert opts.js_runtime_spec() == {"deno": {"path": str(deno)}}

    def test_the_spec_omits_the_path_when_nothing_is_found(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def absent(_n: str, _mode: int = 0, _path: str | None = None) -> str | None:
            return None

        monkeypatch.setattr(shutil, "which", absent)
        monkeypatch.setattr(sys, "prefix", str(tmp_path))
        assert opts.js_runtime_spec() == {"deno": {}}


def test_nvm_only_node_is_flagged_as_invisible_to_cron(tmp_path: Path) -> None:
    nvm = tmp_path / ".nvm/versions/node/v24.16.0/bin/node"
    nvm.parent.mkdir(parents=True)
    nvm.write_text("")
    assert opts.is_nvm_path(str(nvm)) is True
    assert opts.is_nvm_path("/usr/bin/node") is False


class TestFilesystemDetection:
    def test_picks_the_deepest_matching_mount(self, tmp_path: Path) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text(f"/dev/a / ext4 rw 0 0\n/dev/b {tmp_path} btrfs rw 0 0\n")
        assert opts.filesystem_type(tmp_path, mounts) == "btrfs"

    def test_malformed_lines_are_skipped(self, tmp_path: Path) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text(f"garbage\n/dev/b {tmp_path} xfs rw 0 0\n")
        assert opts.filesystem_type(tmp_path, mounts) == "xfs"

    def test_none_when_the_mount_table_is_unreadable(self, tmp_path: Path) -> None:
        assert opts.filesystem_type(tmp_path, tmp_path / "absent") is None

    def test_none_when_nothing_matches(self, tmp_path: Path) -> None:
        mounts = tmp_path / "mounts"
        mounts.write_text("/dev/a /nowhere-at-all ext4 rw 0 0\n")
        assert opts.filesystem_type(tmp_path, mounts) is None


class TestNodeDiscovery:
    def test_prefers_node_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def on_path(_n: str, _mode: int = 0, _path: str | None = None) -> str:
            return "/usr/bin/node"

        monkeypatch.setattr(shutil, "which", on_path)
        assert opts.find_node() == "/usr/bin/node"

    def test_falls_back_to_nvm(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def absent(_n: str, _mode: int = 0, _path: str | None = None) -> str | None:
            return None

        monkeypatch.setattr(shutil, "which", absent)
        node = tmp_path / ".nvm/versions/node/v24.16.0/bin/node"
        node.parent.mkdir(parents=True)
        node.write_text("")
        assert opts.find_node(tmp_path) == str(node)

    def test_picks_the_newest_nvm_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def absent(_n: str, _mode: int = 0, _path: str | None = None) -> str | None:
            return None

        monkeypatch.setattr(shutil, "which", absent)
        for version in ("v18.0.0", "v24.16.0"):
            node = tmp_path / f".nvm/versions/node/{version}/bin/node"
            node.parent.mkdir(parents=True)
            node.write_text("")
        assert "v24.16.0" in (opts.find_node(tmp_path) or "")

    def test_none_when_node_is_nowhere(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def absent(_n: str, _mode: int = 0, _path: str | None = None) -> str | None:
            return None

        monkeypatch.setattr(shutil, "which", absent)
        assert opts.find_node(tmp_path) is None

    def test_node_spec_is_empty_when_node_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_node(*_a: object, **_k: object) -> str | None:
            return None

        monkeypatch.setattr(opts, "find_node", no_node)
        assert opts.js_runtime_spec("node") == {"node": {}}


def test_a_shallower_mount_does_not_override_a_deeper_one(tmp_path: Path) -> None:
    # The longest matching mount point wins, whatever order the table lists them.
    mounts = tmp_path / "mounts"
    mounts.write_text(f"/dev/b {tmp_path} btrfs rw 0 0\n/dev/a / ext4 rw 0 0\n")
    assert opts.filesystem_type(tmp_path, mounts) == "btrfs"
