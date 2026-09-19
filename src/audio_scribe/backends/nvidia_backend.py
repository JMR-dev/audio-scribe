"""NVIDIA backend -- interface only.

Detection and routing are real: if an NVIDIA GPU is present the registry reports
it and explains why it is stepping down. The inference path is not implemented,
because the model cached on this machine is OpenVINO IR and cannot be loaded by
CUDA, and nothing here could be tested against real hardware.

Implementing it means CTranslate2 (faster-whisper) with a CTranslate2 model; note
that CTranslate2 has no ROCm support, so AMD is a genuinely separate path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from audio_scribe import errors
from audio_scribe.backends import probe
from audio_scribe.backends.base import ToolchainStatus

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from audio_scribe.backends.base import TranscribeRequest
    from audio_scribe.transcript import TranscriptResult

_NOT_IMPLEMENTED = (
    "an NVIDIA GPU was detected, but the NVIDIA backend is not implemented in this build"
)
_HINT = (
    "Implement src/audio_scribe/backends/nvidia_backend.py using faster-whisper "
    "(CTranslate2) with a CTranslate2 model. The cached OpenVINO IR model cannot "
    "be loaded on CUDA."
)


class NvidiaBackend:
    name: ClassVar[str] = "nvidia"
    device: ClassVar[str] = "cuda"
    default_model: ClassVar[str] = "large-v3-turbo"

    def __init__(self, model_id: str | None = None, cache_dir: Path | None = None) -> None:
        self.model_id = model_id or self.default_model
        self.cache_dir = cache_dir

    @classmethod
    def probe_hardware(cls) -> bool:
        return probe.has_nvidia_gpu()

    @classmethod
    def probe_toolchain(cls) -> ToolchainStatus:
        return ToolchainStatus(ok=False, reason=_NOT_IMPLEMENTED)

    def load(self) -> None:
        raise errors.BackendNotImplementedError(_NOT_IMPLEMENTED, hint=_HINT)

    def transcribe(
        self,
        pcm: npt.NDArray[np.float32],  # noqa: ARG002 - fixed by the Backend protocol
        request: TranscribeRequest,  # noqa: ARG002 - fixed by the Backend protocol
    ) -> TranscriptResult:
        raise errors.BackendNotImplementedError(_NOT_IMPLEMENTED, hint=_HINT)

    def close(self) -> None:
        return
