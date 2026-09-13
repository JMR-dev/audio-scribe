from __future__ import annotations

import builtins
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import openvino_genai
import pytest

from ccn_transcribe import errors
from ccn_transcribe.backends import openvino_backend as ovb
from ccn_transcribe.backends import probe as probe_mod
from ccn_transcribe.backends.base import TranscribeRequest

LIVE = pytest.mark.skipif(
    not os.environ.get("CCN_LIVE"), reason="set CCN_LIVE=1 to run against the real GPU"
)


class TestLanguageToken:
    def test_bare_code_becomes_a_token(self) -> None:
        assert ovb.language_token("en") == "<|en|>"

    def test_an_existing_token_is_left_alone(self) -> None:
        assert ovb.language_token("<|de|>") == "<|de|>"

    def test_none_stays_none_for_autodetect(self) -> None:
        assert ovb.language_token(None) is None

    def test_an_unknown_code_is_rejected_with_suggestions(self) -> None:
        with pytest.raises(errors.ConfigError, match="unknown language"):
            ovb.language_token("zz", allowed={"<|en|>", "<|de|>"})

    def test_a_known_code_passes_validation(self) -> None:
        assert ovb.language_token("en", allowed={"<|en|>"}) == "<|en|>"


class TestSupportedLanguages:
    def test_none_when_the_config_is_absent(self, tmp_path: Path) -> None:
        assert ovb.supported_languages(tmp_path) is None

    def test_none_when_the_config_is_unreadable(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text("{broken")
        assert ovb.supported_languages(tmp_path) is None

    def test_none_when_there_is_no_language_map(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text('{"max_length": 448}')
        assert ovb.supported_languages(tmp_path) is None

    def test_reads_the_language_map(self, tmp_path: Path) -> None:
        (tmp_path / "generation_config.json").write_text('{"lang_to_id": {"<|en|>": 1}}')
        assert ovb.supported_languages(tmp_path) == {"<|en|>"}

    def test_reads_the_real_cached_model(self) -> None:
        languages = ovb.supported_languages(ovb.resolve_model(ovb.DEFAULT_MODEL, None))
        assert languages is not None
        assert "<|en|>" in languages
        assert len(languages) > 90


class TestCacheRoot:
    def test_respects_xdg_cache_home(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert ovb.cache_root() == tmp_path / "ccn-transcribe/ov_cache"

    def test_falls_back_to_home_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        assert ovb.cache_root() == Path.home() / ".cache/ccn-transcribe/ov_cache"

    def test_is_not_relative_to_the_working_directory(self) -> None:
        # The benchmark scripts use a relative .ov_cache_* and only work from the
        # repo root; the pipeline must not inherit that.
        assert ovb.cache_root().is_absolute()


class TestResolveModel:
    def test_finds_the_cached_model_without_network(self) -> None:
        path = ovb.resolve_model(ovb.DEFAULT_MODEL, ovb.DEFAULT_REVISION)
        assert (path / "openvino_encoder_model.xml").exists()

    def test_a_missing_model_raises_with_a_manual_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise OSError("no such model")

        monkeypatch.setattr(ovb, "_snapshot", boom)
        with pytest.raises(errors.ModelUnavailableError) as caught:
            ovb.resolve_model("nope/nope", None)
        assert "hf download" in (caught.value.hint or "")


class TestDeviceSelection:
    def test_gpu_backend_probes_intel_hardware(self) -> None:
        assert ovb.OpenvinoGpuBackend.probe_hardware() is True

    def test_cpu_backend_always_has_hardware(self) -> None:
        assert ovb.OpenvinoCpuBackend.probe_hardware() is True

    def test_gpu_toolchain_is_available_here(self) -> None:
        assert ovb.OpenvinoGpuBackend.probe_toolchain().ok is True

    def test_cpu_toolchain_is_available_here(self) -> None:
        assert ovb.OpenvinoCpuBackend.probe_toolchain().ok is True

    def test_multi_gpu_enumeration_is_accepted(self) -> None:
        assert ovb.OpenvinoGpuBackend.device_present(["CPU", "GPU.0", "GPU.1"]) is True

    def test_a_cpu_only_enumeration_is_rejected_for_gpu(self) -> None:
        assert ovb.OpenvinoGpuBackend.device_present(["CPU"]) is False

    def test_a_missing_gpu_plugin_is_reported_as_a_plugin_problem(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def absent(_cls: type[ovb.OpenvinoGpuBackend], _devices: list[str]) -> bool:
            return False

        monkeypatch.setattr(ovb.OpenvinoGpuBackend, "device_present", classmethod(absent))
        status = ovb.OpenvinoGpuBackend.probe_toolchain()
        assert status.ok is False
        assert "plugin problem" in status.reason

    def test_an_unimportable_openvino_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = builtins.__import__

        def no_openvino(name: str, *a: object, **k: object) -> Any:
            if name == "openvino":
                raise ImportError("gone")
            return real(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", no_openvino)
        assert ovb.OpenvinoCpuBackend.probe_toolchain().ok is False


class TestTranscribeWiring:
    class _Pipe:
        def __init__(self, chunks: list[Any] | None = None, boom: bool = False) -> None:
            self.chunks = chunks
            self.boom = boom
            self.seen: dict[str, Any] = {}

        def generate(self, _pcm: object, **kwargs: object) -> Any:
            if self.boom:
                raise RuntimeError("kernel compile failed")
            self.seen = dict(kwargs)
            return type("R", (), {"chunks": self.chunks})()

    def _chunk(self, start: float, end: float, text: str) -> Any:
        return type("C", (), {"start_ts": start, "end_ts": end, "text": text})()

    def _backend(self, pipe: object) -> ovb.OpenvinoCpuBackend:
        backend = ovb.OpenvinoCpuBackend()
        backend.pipe = pipe
        backend.languages = {"<|en|>"}
        return backend

    def test_passes_timestamps_and_task(self) -> None:
        pipe = self._Pipe(chunks=[])
        self._backend(pipe).transcribe(np.zeros(16000, np.float32), TranscribeRequest())
        assert pipe.seen["return_timestamps"] is True
        assert pipe.seen["task"] == "transcribe"

    def test_converts_the_language_to_token_form(self) -> None:
        pipe = self._Pipe(chunks=[])
        self._backend(pipe).transcribe(
            np.zeros(16000, np.float32), TranscribeRequest(language="en")
        )
        assert pipe.seen["language"] == "<|en|>"

    def test_omits_language_when_autodetecting(self) -> None:
        pipe = self._Pipe(chunks=[])
        self._backend(pipe).transcribe(np.zeros(16000, np.float32), TranscribeRequest())
        assert "language" not in pipe.seen

    def test_forwards_prompt_and_hotwords_only_when_set(self) -> None:
        pipe = self._Pipe(chunks=[])
        self._backend(pipe).transcribe(
            np.zeros(16000, np.float32),
            TranscribeRequest(initial_prompt="P", hotwords="H"),
        )
        assert pipe.seen["initial_prompt"] == "P"
        assert pipe.seen["hotwords"] == "H"

    def test_normalizes_chunks_into_segments(self) -> None:
        pipe = self._Pipe(chunks=[self._chunk(0.0, -1.0, "hi")])
        got = self._backend(pipe).transcribe(np.zeros(32000, np.float32), TranscribeRequest())
        assert got.segments[0].end == pytest.approx(2.0)

    def test_reports_audio_duration_and_device(self) -> None:
        pipe = self._Pipe(chunks=[self._chunk(0.0, 1.0, "hi")])
        got = self._backend(pipe).transcribe(np.zeros(32000, np.float32), TranscribeRequest())
        assert got.audio_s == pytest.approx(2.0)
        assert got.device == "CPU"
        assert got.backend == "openvino"

    def test_empty_chunks_give_no_segments(self) -> None:
        pipe = self._Pipe(chunks=None)
        assert (
            self._backend(pipe)
            .transcribe(np.zeros(16000, np.float32), TranscribeRequest())
            .segments
            == ()
        )

    def test_a_generate_failure_becomes_a_backend_runtime_error(self) -> None:
        pipe = self._Pipe(boom=True)
        with pytest.raises(errors.BackendRuntimeError, match="during generation"):
            self._backend(pipe).transcribe(np.zeros(16000, np.float32), TranscribeRequest())

    def test_transcribe_loads_lazily(self, monkeypatch: pytest.MonkeyPatch) -> None:
        backend = ovb.OpenvinoCpuBackend()
        calls: list[int] = []

        def fake_load(self: ovb.OpenvinoCpuBackend) -> None:
            calls.append(1)
            self.pipe = TestTranscribeWiring._Pipe(chunks=[])

        monkeypatch.setattr(ovb.OpenvinoCpuBackend, "load", fake_load)
        backend.transcribe(np.zeros(16000, np.float32), TranscribeRequest())
        assert calls == [1]

    def test_close_releases_the_pipeline(self) -> None:
        backend = self._backend(self._Pipe())
        backend.close()
        assert backend.pipe is None


class TestLoadHints:
    def test_cpu_hint_points_at_doctor(self) -> None:
        assert "doctor" in ovb.OpenvinoCpuBackend().load_hint()

    def test_gpu_hint_names_the_render_group_when_the_node_is_unwritable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = probe_mod.RenderNode(
            path=Path("/dev/dri/renderD128"),
            vendor="intel",
            driver="i915",
            mode=0o660,
            writable=False,
        )

        def one_node(*_a: object, **_k: object) -> list[probe_mod.RenderNode]:
            return [node]

        monkeypatch.setattr(probe_mod, "render_nodes", one_node)
        assert "usermod -aG render" in ovb.OpenvinoGpuBackend().load_hint()

    def test_gpu_hint_falls_back_to_doctor_when_permissions_are_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_nodes(*_a: object, **_k: object) -> list[probe_mod.RenderNode]:
            return []

        monkeypatch.setattr(probe_mod, "render_nodes", no_nodes)
        assert "doctor" in ovb.OpenvinoGpuBackend().load_hint()

    def test_a_pipeline_construction_failure_is_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("no device")

        monkeypatch.setattr(openvino_genai, "WhisperPipeline", boom)
        with pytest.raises(errors.BackendRuntimeError, match="could not build a pipeline"):
            ovb.OpenvinoCpuBackend().load()


@LIVE
class TestLive:
    def test_real_transcription_on_the_default_device(self, sine_wav: Path) -> None:
        from ccn_transcribe.media import ffmpeg

        backend = ovb.OpenvinoGpuBackend()
        backend.load()
        result = backend.transcribe(ffmpeg.decode_16k_mono(sine_wav), TranscribeRequest())
        assert result.device == "GPU"


def test_a_cache_miss_falls_through_to_the_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[bool] = []

    def snapshot(_model: str, _revision: str | None, *, local_only: bool) -> str:
        calls.append(local_only)
        if local_only:
            raise OSError("not cached")
        return str(tmp_path)

    monkeypatch.setattr(ovb, "_snapshot", snapshot)
    with caplog.at_level(logging.INFO):
        assert ovb.resolve_model("some/model", None) == tmp_path
    assert calls == [True, False], "the cache must be consulted before the network"
    assert "not cached" in caplog.text
