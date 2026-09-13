"""Backend selection: detection chain, runtime fallback, and demotion."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ccn_transcribe import errors
from ccn_transcribe.backends.amd_backend import AmdBackend
from ccn_transcribe.backends.nvidia_backend import NvidiaBackend
from ccn_transcribe.backends.openvino_backend import OpenvinoCpuBackend, OpenvinoGpuBackend

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt

    from ccn_transcribe.backends.base import Backend, TranscribeRequest
    from ccn_transcribe.transcript import TranscriptResult

log = logging.getLogger(__name__)

DEFAULT_CHAIN: tuple[type[Backend], ...] = (
    NvidiaBackend,
    AmdBackend,
    OpenvinoGpuBackend,
    OpenvinoCpuBackend,
)

# Process-wide: once a backend fails, a 50-job batch must not retry it 50 times.
_DEMOTED: set[str] = set()


def key(backend: type[Backend]) -> str:
    return f"{backend.name}/{backend.device}"


def demote(backend: type[Backend]) -> None:
    _DEMOTED.add(key(backend))


def reset_demotions() -> None:
    _DEMOTED.clear()


def _select_requested(
    requested: str, chain: tuple[type[Backend], ...]
) -> tuple[type[Backend], ...]:
    matches = tuple(b for b in chain if requested in (b.name, b.device.lower(), key(b).lower()))
    if not matches:
        names = ", ".join(sorted({b.name for b in chain} | {"auto"}))
        raise errors.ConfigError(f"unknown backend {requested!r}", hint=f"Choose one of: {names}")
    return matches


def candidates(
    requested: str | None = None, chain: tuple[type[Backend], ...] | None = None
) -> Iterator[type[Backend]]:
    """Backends worth trying, in order."""
    # Resolved here rather than as a default argument, which would bind the tuple
    # at import time and make DEFAULT_CHAIN impossible to override.
    chain = chain if chain is not None else DEFAULT_CHAIN
    explicit = requested is not None and requested != "auto"
    pool = _select_requested(requested, chain) if explicit and requested else chain

    for backend in pool:
        label = key(backend)
        if label in _DEMOTED:
            continue
        if not backend.probe_hardware():
            log.debug("skipping %s: no hardware", label)
            continue
        # Only now may a backend import its toolchain.
        status = backend.probe_toolchain()
        if not status.ok:
            log.warning("skipping %s: %s", label, status.reason)
            continue
        yield backend


class BackendHolder:
    """Keeps a loaded backend alive across jobs (~7s cold, ~0.7s warm)."""

    def __init__(self) -> None:
        self._loaded: dict[str, Backend] = {}

    def get(
        self,
        backend: type[Backend],
        model_id: str | None = None,
        cache_dir: Path | None = None,
    ) -> Backend:
        label = key(backend)
        instance = self._loaded.get(label)
        if instance is None:
            instance = backend(model_id, cache_dir)
            instance.load()
            self._loaded[label] = instance
        return instance

    def drop(self, backend: type[Backend]) -> None:
        instance = self._loaded.pop(key(backend), None)
        if instance is not None:
            instance.close()

    def close_all(self) -> None:
        for instance in self._loaded.values():
            instance.close()
        self._loaded.clear()


def transcribe_with_fallback(
    pcm: npt.NDArray[np.float32],
    request: TranscribeRequest,
    *,
    holder: BackendHolder | None = None,
    requested: str | None = None,
    model_id: str | None = None,
    cache_dir: Path | None = None,
    chain: tuple[type[Backend], ...] | None = None,
) -> TranscriptResult:
    """Transcribe, stepping down the chain on any runtime failure.

    Construction and the first generate() are both inside the try: a render-node
    permission problem and an OpenCL JIT failure surface at one of those points,
    never at device enumeration.
    """
    chain = chain if chain is not None else DEFAULT_CHAIN
    owned = holder or BackendHolder()
    explicit = requested is not None and requested != "auto"
    last: Exception | None = None

    for backend in candidates(requested, chain):
        label = key(backend)
        try:
            instance = owned.get(backend, model_id, cache_dir)
            return instance.transcribe(pcm, request)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Any failure -- construction or first generate -- means step down.
            owned.drop(backend)
            if explicit:
                raise errors.TranscribeError(
                    f"{label} failed and --device was explicit, so no fallback was tried",
                    hint=str(exc),
                ) from exc
            demote(backend)
            log.warning("backend %s failed (%s); demoting it and falling back", label, exc)
            last = exc

    raise errors.TranscribeError(
        "no usable backend remains",
        hint=str(last) if last else "Run `ccn-transcribe doctor` to see what was detected.",
    ) from last
