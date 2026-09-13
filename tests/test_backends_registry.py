from __future__ import annotations

import sys
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from ccn_transcribe import errors
from ccn_transcribe.backends import registry
from ccn_transcribe.backends.base import ToolchainStatus, TranscribeRequest
from ccn_transcribe.transcript import Segment, TranscriptResult

if TYPE_CHECKING:
    from pathlib import Path

    import numpy.typing as npt

PCM = np.zeros(16_000, dtype=np.float32)
REQ = TranscribeRequest()


class FakeBackend:
    name: ClassVar[str] = "fake"
    device: ClassVar[str] = "FAKE"
    default_model: ClassVar[str] = "fake-model"

    hardware: ClassVar[bool] = True
    toolchain: ClassVar[ToolchainStatus] = ToolchainStatus(ok=True)
    fail_on_load: ClassVar[bool] = False
    fail_on_transcribe: ClassVar[bool] = False
    loads: ClassVar[int] = 0

    def __init__(self, model_id: str | None = None, cache_dir: Path | None = None) -> None:
        self.model_id = model_id or self.default_model
        self.cache_dir = cache_dir
        self.closed = False

    @classmethod
    def probe_hardware(cls) -> bool:
        return cls.hardware

    @classmethod
    def probe_toolchain(cls) -> ToolchainStatus:
        return cls.toolchain

    def load(self) -> None:
        type(self).loads += 1
        if self.fail_on_load:
            raise errors.BackendRuntimeError(f"{self.name} failed to load")

    def transcribe(
        self, pcm: npt.NDArray[np.float32], request: TranscribeRequest
    ) -> TranscriptResult:
        if self.fail_on_transcribe:
            raise errors.BackendRuntimeError(f"{self.name} blew up mid-generate")
        return TranscriptResult(
            segments=(Segment(0.0, 1.0, self.name),),
            language=request.language,
            backend=self.name,
            device=self.device,
            model_id=self.model_id,
            model_revision=None,
            audio_s=len(pcm) / 16_000,
            wall_s=0.1,
        )

    def close(self) -> None:
        self.closed = True


def make_backend(label: str, **attrs: object) -> type[FakeBackend]:
    return type(
        f"{label}Backend", (FakeBackend,), {"name": label, "device": label.upper(), **attrs}
    )


@pytest.fixture(autouse=True)
def _clean() -> None:
    registry.reset_demotions()


class TestCandidates:
    def test_skips_backends_with_no_hardware(self) -> None:
        a = make_backend("a", hardware=False)
        b = make_backend("b")
        assert list(registry.candidates(chain=(a, b))) == [b]

    def test_skips_backends_whose_toolchain_is_absent(self) -> None:
        a = make_backend("a", toolchain=ToolchainStatus(ok=False, reason="not installed"))
        b = make_backend("b")
        assert list(registry.candidates(chain=(a, b))) == [b]

    def test_toolchain_is_never_probed_without_hardware(self) -> None:
        # This ordering is what keeps torch from being imported on an Intel box.
        probed: list[str] = []

        class Guard(FakeBackend):
            name: ClassVar[str] = "guard"
            hardware: ClassVar[bool] = False

            @classmethod
            def probe_toolchain(cls) -> ToolchainStatus:
                probed.append(cls.name)
                return ToolchainStatus(ok=True)

        list(registry.candidates(chain=(Guard,)))
        assert probed == []

    def test_chain_order_is_preserved(self) -> None:
        a, b, c = make_backend("a"), make_backend("b"), make_backend("c")
        assert list(registry.candidates(chain=(a, b, c))) == [a, b, c]

    def test_an_explicit_request_selects_only_that_backend(self) -> None:
        a, b = make_backend("a"), make_backend("b")
        assert list(registry.candidates(requested="b", chain=(a, b))) == [b]

    def test_an_unknown_request_is_a_config_error(self) -> None:
        with pytest.raises(errors.ConfigError, match="unknown"):
            list(registry.candidates(requested="nope", chain=(make_backend("a"),)))

    def test_auto_is_not_treated_as_a_name(self) -> None:
        a = make_backend("a")
        assert list(registry.candidates(requested="auto", chain=(a,))) == [a]

    def test_demoted_backends_are_skipped(self) -> None:
        a, b = make_backend("a"), make_backend("b")
        registry.demote(a)
        assert list(registry.candidates(chain=(a, b))) == [b]


class TestFallback:
    def test_uses_the_first_working_backend(self) -> None:
        a, b = make_backend("a"), make_backend("b")
        got = registry.transcribe_with_fallback(PCM, REQ, chain=(a, b))
        assert got.backend == "a"

    def test_falls_back_when_construction_fails(self) -> None:
        a = make_backend("a", fail_on_load=True)
        b = make_backend("b")
        assert registry.transcribe_with_fallback(PCM, REQ, chain=(a, b)).backend == "b"

    def test_falls_back_when_the_first_generate_fails(self) -> None:
        # The permission and JIT failures both surface here, not at detection.
        a = make_backend("a", fail_on_transcribe=True)
        b = make_backend("b")
        assert registry.transcribe_with_fallback(PCM, REQ, chain=(a, b)).backend == "b"

    def test_a_failure_demotes_for_the_rest_of_the_run(self) -> None:
        a = make_backend("a", fail_on_transcribe=True)
        b = make_backend("b")
        registry.transcribe_with_fallback(PCM, REQ, chain=(a, b))
        a.loads = 0
        registry.transcribe_with_fallback(PCM, REQ, chain=(a, b))
        assert a.loads == 0, "a 50-job batch must not retry a dead backend 50 times"

    def test_exhausting_the_chain_raises(self) -> None:
        a = make_backend("a", fail_on_load=True)
        with pytest.raises(errors.TranscribeError, match="no usable backend"):
            registry.transcribe_with_fallback(PCM, REQ, chain=(a,))

    def test_no_candidates_at_all_raises(self) -> None:
        a = make_backend("a", hardware=False)
        with pytest.raises(errors.TranscribeError, match="no usable backend"):
            registry.transcribe_with_fallback(PCM, REQ, chain=(a,))

    def test_an_explicit_device_never_falls_back(self) -> None:
        # Asking for the GPU and silently getting the CPU makes a benchmark a lie.
        a = make_backend("a", fail_on_transcribe=True)
        b = make_backend("b")
        with pytest.raises(errors.TranscribeError):
            registry.transcribe_with_fallback(PCM, REQ, requested="a", chain=(a, b))

    def test_warnings_name_the_failing_backend(self, caplog: pytest.LogCaptureFixture) -> None:
        a = make_backend("a", fail_on_transcribe=True)
        b = make_backend("b")
        registry.transcribe_with_fallback(PCM, REQ, chain=(a, b))
        assert "a" in caplog.text.lower()


class TestHolder:
    def test_reuses_a_loaded_backend_across_jobs(self) -> None:
        a = make_backend("a")
        holder = registry.BackendHolder()
        holder.get(a)
        holder.get(a)
        assert a.loads == 1, "model load is ~7s cold; reloading per job is pure waste"

    def test_close_all_closes_instances(self) -> None:
        a = make_backend("a")
        holder = registry.BackendHolder()
        instance = holder.get(a)
        holder.close_all()
        assert isinstance(instance, FakeBackend)
        assert instance.closed is True

    def test_dropping_forces_a_reload(self) -> None:
        a = make_backend("a")
        holder = registry.BackendHolder()
        holder.get(a)
        holder.drop(a)
        holder.get(a)
        assert a.loads == 2


class TestRealChain:
    def test_default_chain_order(self) -> None:
        assert [b.name for b in registry.DEFAULT_CHAIN] == [
            "nvidia",
            "amd",
            "openvino",
            "openvino",
        ]

    def test_cpu_is_last_and_terminal(self) -> None:
        assert registry.DEFAULT_CHAIN[-1].device == "CPU"

    def test_this_machine_resolves_to_the_intel_gpu(self) -> None:
        chosen = next(iter(registry.candidates()))
        assert (chosen.name, chosen.device) == ("openvino", "GPU")

    def test_torch_is_never_imported_on_this_intel_only_box(self) -> None:
        # The structural guarantee, made executable.
        for module in ("torch", "faster_whisper", "ctranslate2", "transformers"):
            sys.modules.pop(module, None)
        list(registry.candidates())
        assert not {"torch", "faster_whisper", "ctranslate2"} & set(sys.modules)


class TestExplicitDeviceDiagnostics:
    def test_requesting_absent_hardware_is_a_config_error(self) -> None:
        # A device that cannot run is a configuration mistake, not a per-job
        # failure to repeat once per URL.
        a = make_backend("a", hardware=False)
        with pytest.raises(errors.ConfigError, match="cannot be used"):
            registry.transcribe_with_fallback(PCM, REQ, requested="a", chain=(a,))

    def test_the_hint_says_the_hardware_is_absent(self) -> None:
        a = make_backend("a", hardware=False)
        try:
            registry.transcribe_with_fallback(PCM, REQ, requested="a", chain=(a,))
        except errors.ConfigError as exc:
            assert "no a hardware detected" in (exc.hint or "")

    def test_the_hint_relays_a_toolchain_reason(self) -> None:
        a = make_backend("a", toolchain=ToolchainStatus(ok=False, reason="not implemented"))
        assert "not implemented" in registry.explain_unavailable("a", (a,))

    def test_a_runtime_failure_still_reports_as_a_transcribe_error(self) -> None:
        a = make_backend("a", fail_on_transcribe=True)
        with pytest.raises(errors.TranscribeError):
            registry.transcribe_with_fallback(PCM, REQ, requested="a", chain=(a,))

    def test_real_nvidia_request_explains_the_absence(self) -> None:
        assert "no nvidia hardware" in registry.explain_unavailable("nvidia")
