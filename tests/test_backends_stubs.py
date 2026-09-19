from __future__ import annotations

import numpy as np
import pytest

from audio_scribe import errors
from audio_scribe.backends.amd_backend import AmdBackend
from audio_scribe.backends.base import TranscribeRequest
from audio_scribe.backends.nvidia_backend import NvidiaBackend

STUBS = [NvidiaBackend, AmdBackend]


@pytest.mark.parametrize("backend", STUBS)
class TestStubs:
    def test_hardware_probe_is_real(self, backend: type[NvidiaBackend]) -> None:
        # No NVIDIA or AMD on this machine, so detection must say so honestly.
        assert backend.probe_hardware() is False

    def test_toolchain_reports_not_implemented(self, backend: type[NvidiaBackend]) -> None:
        status = backend.probe_toolchain()
        assert status.ok is False
        assert "not implemented" in status.reason

    def test_load_raises_with_an_actionable_hint(self, backend: type[NvidiaBackend]) -> None:
        with pytest.raises(errors.BackendNotImplementedError) as caught:
            backend().load()
        hint = caught.value.hint or ""
        assert "backends/" in hint

    def test_transcribe_raises_rather_than_returning_nonsense(
        self, backend: type[NvidiaBackend]
    ) -> None:
        with pytest.raises(errors.BackendNotImplementedError):
            backend().transcribe(np.zeros(16, np.float32), TranscribeRequest())

    def test_close_is_safe(self, backend: type[NvidiaBackend]) -> None:
        backend().close()

    def test_default_model_is_not_the_openvino_ir(self, backend: type[NvidiaBackend]) -> None:
        # OpenVINO IR cannot load on CUDA or ROCm, so each backend owns its default.
        assert "-ov" not in backend.default_model

    def test_a_custom_model_id_is_kept(self, backend: type[NvidiaBackend]) -> None:
        assert backend("custom/model").model_id == "custom/model"


def test_the_two_stubs_use_different_runtimes() -> None:
    # CTranslate2 has no ROCm support, so these cannot share one implementation.
    assert NvidiaBackend.device != AmdBackend.device
