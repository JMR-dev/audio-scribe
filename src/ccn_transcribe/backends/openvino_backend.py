"""OpenVINO GenAI backend -- the path this machine actually runs.

Model resolution is offline-first so a warm cache never touches the network, and
CACHE_DIR is anchored under XDG rather than the working directory.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ccn_transcribe import errors
from ccn_transcribe.backends import probe
from ccn_transcribe.backends.base import ToolchainStatus
from ccn_transcribe.transcript import TranscriptResult, normalize_segments

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ccn_transcribe.backends.base import TranscribeRequest

log = logging.getLogger(__name__)

DEFAULT_MODEL = "OpenVINO/whisper-large-v3-turbo-int8-ov"
DEFAULT_REVISION = "b568445dd5dc8c695bde596f8acbb4694fd6ba64"
SAMPLE_RATE = 16_000


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ccn-transcribe" / "ov_cache"


def language_token(code: str | None, allowed: set[str] | None = None) -> str | None:
    """Whisper wants ``<|en|>``, not ``en``."""
    if code is None:
        return None
    token = code if code.startswith("<|") else f"<|{code}|>"
    if allowed is not None and token not in allowed:
        valid = ", ".join(sorted(t.strip("<|>") for t in allowed)[:12])
        raise errors.ConfigError(
            f"unknown language {code!r}",
            hint=f"Valid codes include: {valid}, ...",
        )
    return token


def supported_languages(model_dir: Path) -> set[str] | None:
    config = model_dir / "generation_config.json"
    if not config.exists():
        return None
    try:
        doc = json.loads(config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    mapping = doc.get("lang_to_id")
    return set(mapping) if mapping else None


def snapshot(model_id: str, revision: str | None, *, local_only: bool) -> str:
    """One typed boundary around huggingface_hub, which ships no type information."""
    # Imported lazily so --help and doctor do not pay for it.
    from huggingface_hub import (  # pylint: disable=import-outside-toplevel
        snapshot_download,  # pyright: ignore[reportUnknownVariableType]
    )

    path: object = snapshot_download(model_id, revision=revision, local_files_only=local_only)
    return str(path)


def resolve_model(model_id: str, revision: str | None) -> Path:
    """Prefer the local cache; only reach the network when we must.

    A warm cache therefore never makes a network call, which is what lets the
    pipeline run fully offline.
    """
    try:
        cached = snapshot(model_id, revision, local_only=True)
    except Exception:  # pylint: disable=broad-exception-caught
        # Any failure here just means "not in the cache"; fall through to the network.
        log.info("%s is not cached; downloading (~790 MB for the default)", model_id)
    else:
        return Path(cached)

    try:
        fetched = snapshot(model_id, revision, local_only=False)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        raise errors.ModelUnavailableError(
            f"could not obtain model {model_id}",
            hint=f"Fetch it manually with: hf download {model_id}",
        ) from exc
    return Path(fetched)


class _OpenvinoBackend:
    name: ClassVar[str] = "openvino"
    device: ClassVar[str] = "CPU"
    default_model: ClassVar[str] = DEFAULT_MODEL

    def __init__(self, model_id: str | None = None, cache_dir: Path | None = None) -> None:
        self.model_id = model_id or self.default_model
        self.revision = DEFAULT_REVISION if self.model_id == DEFAULT_MODEL else None
        self.cache_dir = cache_dir or (cache_root() / self.device.lower())
        self.pipe: Any | None = None
        self.languages: set[str] | None = None

    @classmethod
    def probe_hardware(cls) -> bool:
        return True

    @classmethod
    def probe_toolchain(cls) -> ToolchainStatus:
        try:
            import openvino as ov  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            return ToolchainStatus(ok=False, reason=f"openvino is not importable: {exc}")
        devices = ov.Core().available_devices
        if not cls.device_present(devices):
            return ToolchainStatus(
                ok=False,
                reason=(
                    f"OpenVINO reports {devices}, without {cls.device}. The driver is "
                    "fine or clinfo would have failed; this is a plugin problem."
                ),
            )
        return ToolchainStatus(ok=True)

    @classmethod
    def device_present(cls, devices: list[str]) -> bool:
        return cls.device in devices

    def load(self) -> None:
        # Lazy: constructing a pipeline is the only thing that needs this.
        import openvino_genai as ov_genai  # pylint: disable=import-outside-toplevel

        model_dir = resolve_model(self.model_id, self.revision)
        self.languages = supported_languages(model_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.pipe = ov_genai.WhisperPipeline(
                str(model_dir), self.device, CACHE_DIR=str(self.cache_dir)
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise errors.BackendRuntimeError(
                f"OpenVINO could not build a pipeline on {self.device}: {exc}",
                hint=self.load_hint(),
            ) from exc

    def load_hint(self) -> str:
        if self.device != "GPU":
            return "Try --device cpu, or run `ccn-transcribe doctor`."
        nodes = [n for n in probe.render_nodes() if n.vendor == "intel"]
        if nodes and not nodes[0].writable:
            return (
                f"{nodes[0].path} is not writable by this user. "
                "Run: sudo usermod -aG render $USER, then log out and back in."
            )
        return "Run `ccn-transcribe doctor` to separate driver problems from plugin ones."

    def transcribe(
        self, pcm: npt.NDArray[np.float32], request: TranscribeRequest
    ) -> TranscriptResult:
        if self.pipe is None:
            self.load()
        assert self.pipe is not None  # noqa: S101 - load() raises otherwise
        kwargs: dict[str, Any] = {
            "task": request.task,
            "return_timestamps": True,
            "num_beams": request.num_beams,
        }
        token = language_token(request.language, self.languages)
        if token is not None:
            kwargs["language"] = token
        if request.initial_prompt:
            kwargs["initial_prompt"] = request.initial_prompt
        if request.hotwords:
            kwargs["hotwords"] = request.hotwords

        started = time.perf_counter()
        try:
            raw = self.pipe.generate(pcm, **kwargs)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise errors.BackendRuntimeError(
                f"{self.name}/{self.device} failed during generation: {exc}",
                hint=self.load_hint(),
            ) from exc
        wall = time.perf_counter() - started

        audio_s = len(pcm) / SAMPLE_RATE
        raw_chunks: list[Any] = list(getattr(raw, "chunks", None) or [])
        chunks: list[tuple[float | None, float | None, str]] = [
            (c.start_ts, c.end_ts, c.text) for c in raw_chunks
        ]
        segments = normalize_segments(chunks, audio_s) if chunks else ()
        return TranscriptResult(
            segments=segments,
            language=request.language,
            backend=self.name,
            device=self.device,
            model_id=self.model_id,
            model_revision=self.revision,
            audio_s=audio_s,
            wall_s=wall,
            task=request.task,
            num_beams=request.num_beams,
        )

    def close(self) -> None:
        self.pipe = None


class OpenvinoGpuBackend(_OpenvinoBackend):
    device: ClassVar[str] = "GPU"

    @classmethod
    def probe_hardware(cls) -> bool:
        return probe.has_intel_gpu()

    @classmethod
    def device_present(cls, devices: list[str]) -> bool:
        # Multi-GPU systems enumerate GPU.0, GPU.1, ...
        return any(d == "GPU" or d.startswith("GPU.") for d in devices)


class OpenvinoCpuBackend(_OpenvinoBackend):
    device: ClassVar[str] = "CPU"
