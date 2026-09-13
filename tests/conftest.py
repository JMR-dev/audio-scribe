from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _run(args: list[str]) -> None:
    subprocess.run(args, capture_output=True, check=True)  # noqa: S603


@pytest.fixture(scope="session")
def sine_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One second of 440 Hz at 44.1 kHz stereo -- a real file, not a mock."""
    path = tmp_path_factory.mktemp("media") / "sine.wav"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1:sample_rate=44100",
            "-ac",
            "2",
            str(path),
        ]
    )
    return path


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A video with no audio track at all."""
    path = tmp_path_factory.mktemp("media") / "mute.mkv"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-r",
            "5",
            str(path),
        ]
    )
    return path


@pytest.fixture
def not_media(tmp_path: Path) -> Path:
    path = tmp_path / "notes.txt"
    path.write_text("this is definitely not a media file")
    return path


@pytest.fixture(scope="session")
def sine_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real mkv with audio -- truncating it leaves a readable header duration."""
    path = tmp_path_factory.mktemp("media") / "clip.mkv"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1:sample_rate=44100",
            "-r",
            "5",
            "-shortest",
            str(path),
        ]
    )
    return path
