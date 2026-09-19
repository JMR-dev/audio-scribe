"""ffmpeg/ffprobe helpers.

Decoding is done by shelling out rather than through a Python audio library, which
keeps librosa/numba out of the dependency tree. Lifted from scripts/smoke.py, where
the same call was duplicated in scripts/bench.py.

Command construction is separated from execution so the argument lists can be
asserted on directly.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import TYPE_CHECKING, Literal

import numpy as np

from audio_scribe import errors

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

log = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
AudioProfile = Literal["source", "whisper"]
PROFILES: tuple[str, ...] = ("source", "whisper")

REQUIRED_BINARIES = ("ffmpeg", "ffprobe")


def _run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    try:
        # Argument lists are built here from fixed templates; never a shell string.
        return subprocess.run(args, capture_output=True, check=check)  # noqa: S603
    except FileNotFoundError as exc:
        raise errors.PreflightError(
            f"{args[0]} is not installed or not on PATH",
            hint="Install ffmpeg (Fedora: sudo dnf install ffmpeg-free).",
        ) from exc


def _stderr_tail(exc: subprocess.CalledProcessError, lines: int = 20) -> str:
    raw: bytes = exc.stderr or b""
    return "\n".join(raw.decode("utf-8", "replace").splitlines()[-lines:])


def has_flac_encoder() -> bool:
    return b" flac " in _run(["ffmpeg", "-hide_banner", "-encoders"], check=False).stdout


def require_binaries() -> None:
    """Fail before any job starts rather than mid-batch."""
    for name in REQUIRED_BINARIES:
        if shutil.which(name) is None:
            raise errors.PreflightError(
                f"{name} is not installed or not on PATH",
                hint="Install ffmpeg (Fedora: sudo dnf install ffmpeg-free).",
            )
    if not has_flac_encoder():
        raise errors.PreflightError(
            "this ffmpeg build has no flac encoder",
            hint="Install a build with FLAC support (Fedora: ffmpeg-free includes it).",
        )


def probe_command(path: Path, entry: str, select: str | None = None) -> list[str]:
    args = ["ffprobe", "-v", "error"]
    if select is not None:
        args += ["-select_streams", select]
    return [*args, "-show_entries", entry, "-of", "default=nw=1:nk=1", str(path)]


def probe_field(path: Path, entry: str, select: str | None = None) -> str | None:
    try:
        out = _run(probe_command(path, entry, select)).stdout
    except subprocess.CalledProcessError as exc:
        raise errors.ProbeError(
            f"ffprobe could not read {path.name}",
            hint="The file may be truncated or not a media file.",
        ) from exc
    lines = out.decode("utf-8", "replace").strip().splitlines()
    return lines[0] if lines else None


def probe_duration(path: Path) -> float | None:
    """Duration in seconds, or None when the container does not declare one."""
    for entry, select in (("format=duration", None), ("stream=duration", "a:0")):
        raw = probe_field(path, entry, select)
        if raw and raw != "N/A":
            return float(raw)
    return None


# A derived stream this far off its source means the extraction was cut short.
# The absolute floor covers short clips, where container rounding alone exceeds 0.5%.
DURATION_TOLERANCE = 0.005
MIN_TOLERANCE_S = 1.0


def durations_match(actual: float, expected: float) -> bool:
    """Whether a derived stream is as long as the source it came from."""
    return abs(actual - expected) <= max(MIN_TOLERANCE_S, expected * DURATION_TOLERANCE)


def has_audio_stream(path: Path) -> bool:
    try:
        return probe_field(path, "stream=index", "a:0") is not None
    except errors.ProbeError:
        return False


def flac_command(
    src: Path, dst: Path, *, profile: AudioProfile | str = "source", compression: int = 5
) -> list[str]:
    if profile not in PROFILES:
        raise ValueError(f"unknown audio profile {profile!r}; expected one of {PROFILES}")
    args = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "flac",
        # The FLAC encoder accepts only s16/s32, and AAC/Opus decode to fltp.
        "-sample_fmt",
        "s16",
        "-compression_level",
        str(compression),
    ]
    if profile == "whisper":
        args += ["-ar", str(SAMPLE_RATE), "-ac", "1"]
    return [*args, str(dst)]


def decode_command(path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]


def extract_flac(
    src: Path, dst: Path, *, profile: AudioProfile | str = "source", compression: int = 5
) -> None:
    """Extract the first audio track losslessly to FLAC.

    ``source`` keeps the original rate and channel count, which is the point of
    choosing FLAC; ``whisper`` writes the compact 16 kHz mono variant.
    """
    args = flac_command(src, dst, profile=profile, compression=compression)
    if not has_audio_stream(src):
        raise errors.NoAudioStreamError(
            f"{src.name} has no audio track",
            hint="Nothing to transcribe. Check the source, or pick a format with audio.",
        )
    try:
        _run(args)
    except subprocess.CalledProcessError as exc:
        raise errors.DecodeError(
            f"ffmpeg failed to extract audio from {src.name}",
            hint=_stderr_tail(exc) or "Run ffmpeg manually on the file to see why.",
        ) from exc


def decode_16k_mono(path: Path) -> npt.NDArray[np.float32]:
    """Decode to the 16 kHz mono float32 Whisper expects."""
    try:
        raw = _run(decode_command(path)).stdout
    except subprocess.CalledProcessError as exc:
        raise errors.DecodeError(
            f"ffmpeg could not decode {path.name}",
            hint=_stderr_tail(exc) or "The file may be truncated or not a media file.",
        ) from exc
    return np.frombuffer(raw, dtype=np.float32)
