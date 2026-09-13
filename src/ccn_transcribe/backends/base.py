"""The backend interface.

Two-phase detection is the contract: ``probe_hardware`` must import nothing
heavy, and the registry only calls ``probe_toolchain`` once hardware is confirmed.
That ordering is what keeps torch off an Intel-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from ccn_transcribe.transcript import TranscriptResult


@dataclass(frozen=True, slots=True)
class ToolchainStatus:
    ok: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class TranscribeRequest:
    language: str | None = None
    task: str = "transcribe"
    num_beams: int = 1
    initial_prompt: str | None = None
    hotwords: str | None = None


@runtime_checkable
class Backend(Protocol):
    name: ClassVar[str]
    device: ClassVar[str]
    default_model: ClassVar[str]

    def __init__(self, model_id: str | None = None, cache_dir: Path | None = None) -> None: ...

    @classmethod
    def probe_hardware(cls) -> bool: ...

    @classmethod
    def probe_toolchain(cls) -> ToolchainStatus: ...

    def load(self) -> None: ...

    def transcribe(
        self, pcm: npt.NDArray[np.float32], request: TranscribeRequest
    ) -> TranscriptResult: ...

    def close(self) -> None: ...
