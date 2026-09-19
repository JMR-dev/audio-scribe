"""yt-dlp and aria2c option construction.

Pure functions, so the argument lists can be asserted directly rather than by
running a download.

yt-dlp's Aria2cFD already passes ``-x16 -j16 -s16 --min-split-size 1M
--file-allocation=none --http-accept-gzip=true``. -x16 is aria2's per-server cap,
so re-sending connection counts is a no-op. User arguments are appended *after*
those defaults, and aria2 takes the last occurrence, so ours win.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from audio_scribe import errors

if TYPE_CHECKING:
    from collections.abc import Sequence

# Aria2cFD.SUPPORTED_PROTOCOLS in yt-dlp 2026.08.19. HLS and DASH fragments are
# NOT included, so yt-dlp silently uses its native downloader for those.
ARIA2C_PROTOCOLS = ("http", "https", "ftp", "ftps")

# yt-dlp registers these in yt_dlp/__init__.py.
JS_RUNTIMES = ("deno", "node", "bun", "quickjs")

# fallocate is instant on these; elsewhere it would be a slow full write.
FALLOC_FILESYSTEMS = frozenset({"btrfs", "ext4", "ext3", "xfs", "f2fs", "ocfs2"})

PROC_MOUNTS = Path("/proc/mounts")

DEFAULT_CONNECTIONS = 16
DEFAULT_FRAGMENTS = 16


def filesystem_type(path: Path, mounts: Path = PROC_MOUNTS) -> str | None:
    """Filesystem backing ``path``, taken from the mount table."""
    try:
        lines = mounts.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    target = path.resolve()
    best_depth = -1
    best_type: str | None = None
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point, fstype = Path(fields[1]), fields[2]
        if target == mount_point or mount_point in target.parents:
            depth = len(mount_point.parts)
            if depth > best_depth:
                best_depth, best_type = depth, fstype
    return best_type


def allocation_for(fstype: str | None) -> str:
    """falloc where it is instant, none elsewhere."""
    return "falloc" if fstype in FALLOC_FILESYSTEMS else "none"


def aria2c_args(
    *,
    connections: int = DEFAULT_CONNECTIONS,
    lowest_speed: str = "50K",
    disk_cache: str = "64M",
    file_allocation: str = "none",
    timeout: int = 30,
    connect_timeout: int = 15,
    retry_wait: int = 3,
    auto_save: int = 20,
    log_file: Path | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    """Only the flags that actually change aria2's behaviour."""
    args = [
        # aria2 retries with zero delay by default, burning all five tries in
        # under a second against a server that is merely struggling.
        f"--retry-wait={retry_wait}",
        # A dead connection otherwise holds one of sixteen slots for a full minute.
        f"--timeout={timeout}",
        f"--connect-timeout={connect_timeout}",
        # Off by default, which means a wedged connection hangs forever.
        f"--lowest-speed-limit={lowest_speed}",
        # Sixteen writers at out-of-order offsets thrash a 16M cache.
        f"--disk-cache={disk_cache}",
        f"--file-allocation={file_allocation}",
        # Default 60s. The control file is what a resume reads, so anything not
        # checkpointed when the process is killed is re-downloaded; a kill in the
        # first minute otherwise preserves a file recording zero completed pieces.
        f"--auto-save-interval={auto_save}",
    ]
    if connections != DEFAULT_CONNECTIONS:
        args += [f"-x{connections}", f"-s{connections}"]
    if log_file is not None:
        # --console-log-level=warn swallows almost everything, so a failed
        # download otherwise leaves no forensics at all.
        args += [f"--log={log_file}", "--log-level=notice"]
    return [*args, *extra]


def is_nvm_path(path: str) -> bool:
    """nvm-managed binaries are absent from cron and systemd environments."""
    return "/.nvm/" in path


def find_node(home: Path | None = None) -> str | None:
    found = shutil.which("node")
    if found:
        return found
    root = (home or Path.home()) / ".nvm/versions/node"
    candidates = sorted(root.glob("*/bin/node")) if root.is_dir() else []
    return str(candidates[-1]) if candidates else None


def bundled(name: str, prefix: str | Path | None = None) -> str | None:
    """An executable installed alongside the running interpreter.

    The deno extra puts its binary in this environment's bin directory, which is
    *not* on PATH once the package is installed as a tool -- only the console
    script is exposed. Looking here before PATH is what makes depending on the
    extra actually self-sufficient, outside the project checkout and under cron.
    """
    candidate = Path(prefix if prefix is not None else sys.prefix) / "bin" / name
    return str(candidate) if os.access(candidate, os.X_OK) and candidate.is_file() else None


def find_runtime(name: str) -> str | None:
    """Where this JS runtime lives, preferring the one we shipped with."""
    if name == "node":
        return find_node()
    return bundled(name) or shutil.which(name)


def js_runtime_spec(preferred: str | None = None) -> dict[str, dict[str, Any]]:
    """YouTube needs a JS runtime for signature challenges.

    yt-dlp defaults to deno and resolves it on PATH. Passing the absolute path
    instead is what keeps extraction working wherever the package is installed.
    """
    name = preferred or "deno"
    if name not in JS_RUNTIMES:
        raise errors.ConfigError(
            f"unknown JS runtime {name!r}",
            hint=f"Choose one of: {', '.join(JS_RUNTIMES)}",
        )
    found = find_runtime(name)
    return {name: {"path": found} if found else {}}


@dataclass(frozen=True, slots=True)
class DownloadConfig:
    """Everything the download stage needs, in one place."""

    max_height: int | None = 1080
    use_aria2c: bool = False
    playlist: bool = False
    connections: int = DEFAULT_CONNECTIONS
    concurrent_fragments: int = DEFAULT_FRAGMENTS
    retries: int = 5
    cookies_from_browser: str | None = None
    js_runtime: str | None = None
    file_allocation: str = "none"
    lowest_speed: str = "50K"
    extra_aria2c_args: tuple[str, ...] = ()
    quiet: bool = False


DEFAULT_DOWNLOAD = DownloadConfig()


def ydl_opts(
    outtmpl: str,
    config: DownloadConfig = DEFAULT_DOWNLOAD,
    *,
    aria2_log: Path | None = None,
) -> dict[str, Any]:
    height = f"[height<={config.max_height}]" if config.max_height else ""
    options: dict[str, Any] = {
        "format": f"bv*{height}+ba/b{height}/b",
        # Prefer progressive HTTPS, which is what aria2c can actually accelerate.
        "format_sort": ["proto:https"],
        # Merging is a stream copy; mkv takes any codec pair, so this ffmpeg build
        # never needs the x264/x265 encoders it does not have.
        "merge_output_format": "mkv",
        "outtmpl": outtmpl,
        "continuedl": True,
        "writeinfojson": True,
        "noprogress": config.quiet,
        "quiet": config.quiet,
        "retries": config.retries,
        # aria2c cannot take HLS/DASH fragments, so this is the knob that matters
        # for those formats.
        "concurrent_fragment_downloads": config.concurrent_fragments,
        # A video URL that happens to carry &list= should stay one video.
        "noplaylist": not config.playlist,
        "ignoreerrors": "only_download",
        "js_runtimes": js_runtime_spec(config.js_runtime),
    }
    if config.cookies_from_browser:
        options["cookiesfrombrowser"] = (config.cookies_from_browser,)
    if config.use_aria2c:
        options["external_downloader"] = dict.fromkeys(ARIA2C_PROTOCOLS, "aria2c")
        options["external_downloader_args"] = {
            "aria2c": aria2c_args(
                connections=config.connections,
                lowest_speed=config.lowest_speed,
                file_allocation=config.file_allocation,
                log_file=aria2_log,
                extra=config.extra_aria2c_args,
            )
        }
    return options
