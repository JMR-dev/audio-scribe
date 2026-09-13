"""Doctor's failure paths -- the rows that only appear on a broken machine."""

from __future__ import annotations

import builtins
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

import openvino as ov

from ccn_transcribe import doctor
from ccn_transcribe.backends import probe
from ccn_transcribe.media import ffmpeg, ytdlp_opts

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest


def const[T](value: T) -> Callable[..., T]:
    """A typed stand-in for a bare lambda in monkeypatch.setattr."""

    def _inner(*_a: object, **_k: object) -> T:
        return value

    return _inner


def _no_binary(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    real = shutil.which

    def which(name: str, _mode: int = 0, _path: str | None = None) -> str | None:
        return None if name in names else real(name)

    monkeypatch.setattr(shutil, "which", which)


class TestBinaries:
    def test_a_missing_binary_fails_with_a_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_binary(monkeypatch, "ffmpeg")
        check = doctor.binary_check("ffmpeg")
        assert check.status is doctor.Status.FAIL
        assert check.hint


class TestOpenvino:
    def test_an_unimportable_openvino_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real = builtins.__import__

        def blocked(name: str, *a: object, **k: object) -> Any:
            if name == "openvino":
                raise ImportError("not installed")
            return real(name, *a, **k)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", blocked)
        assert doctor.check_openvino().status is doctor.Status.FAIL
        assert doctor.check_devices()[0].status is doctor.Status.FAIL

    def test_an_intel_gpu_without_a_plugin_is_a_plugin_problem(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # README section 7: clinfo working while OpenVINO shows only CPU is a
        # plugin problem, not a driver problem.
        class Core:
            available_devices: ClassVar[list[str]] = ["CPU"]

        monkeypatch.setattr(ov, "Core", Core)
        monkeypatch.setattr(probe, "has_intel_gpu", const(True))
        checks = doctor.check_devices()
        assert any(c.name == "gpu plugin" and c.status is doctor.Status.FAIL for c in checks)
        assert "plugin problem" in (checks[-1].hint or "")


class TestFlacEncoder:
    def test_absent_encoder_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ffmpeg, "has_flac_encoder", const(False))
        assert doctor.check_flac_encoder().status is doctor.Status.FAIL

    def test_a_broken_ffmpeg_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom() -> bool:
            raise OSError("ffmpeg is gone")

        monkeypatch.setattr(ffmpeg, "has_flac_encoder", boom)
        assert doctor.check_flac_encoder().status is doctor.Status.FAIL


class TestJsRuntime:
    def test_no_runtime_at_all_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_binary(monkeypatch, "deno")
        monkeypatch.setattr(ytdlp_opts, "find_node", const(None))
        check = doctor.check_js_runtime()
        assert check.status is doctor.Status.FAIL
        assert "YouTube" in (check.hint or "")

    def test_nvm_only_node_warns_about_cron(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_binary(monkeypatch, "deno")
        monkeypatch.setattr(
            ytdlp_opts, "find_node", const("/home/u/.nvm/versions/node/v24/bin/node")
        )
        check = doctor.check_js_runtime()
        assert check.status is doctor.Status.WARN
        assert "cron" in (check.hint or "")

    def test_system_node_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_binary(monkeypatch, "deno")
        monkeypatch.setattr(ytdlp_opts, "find_node", const("/usr/bin/node"))
        assert doctor.check_js_runtime().status is doctor.Status.PASS


class TestRenderNodes:
    def test_no_nodes_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, "render_nodes", const([]))
        assert doctor.check_render_nodes()[0].status is doctor.Status.WARN

    def test_an_unwritable_node_fails_with_the_group_fix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = probe.RenderNode(
            path=Path("/dev/dri/renderD128"),
            vendor="intel",
            driver="i915",
            mode=0o660,
            writable=False,
        )
        monkeypatch.setattr(probe, "render_nodes", const([node]))
        checks = doctor.check_render_nodes()
        failing = [c for c in checks if c.status is doctor.Status.FAIL]
        assert failing
        assert "usermod -aG render" in (failing[0].hint or "")

    def test_a_properly_permissioned_node_is_silent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = probe.RenderNode(
            path=Path("/dev/dri/renderD128"),
            vendor="intel",
            driver="i915",
            mode=0o660,
            writable=True,
        )
        monkeypatch.setattr(probe, "render_nodes", const([node]))
        assert all(c.status is doctor.Status.PASS for c in doctor.check_render_nodes())


class TestClinfo:
    def test_missing_clinfo_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _no_binary(monkeypatch, "clinfo")
        assert doctor.check_clinfo().status is doctor.Status.WARN

    def test_no_opencl_device_points_at_the_compute_runtime(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def empty(*_a: object, **_k: object) -> Any:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", empty)
        check = doctor.check_clinfo()
        assert check.status is doctor.Status.FAIL
        assert "intel-compute-runtime" in (check.hint or "")

    def test_a_crashing_clinfo_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise OSError("cannot execute")

        monkeypatch.setattr(subprocess, "run", boom)
        assert doctor.check_clinfo().status is doctor.Status.WARN


class TestModelAndCache:
    def test_an_uncached_model_warns_with_the_download_command(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*_a: object, **_k: object) -> None:
            raise OSError("not cached")

        monkeypatch.setattr(doctor, "snapshot", boom)
        check = doctor.check_model()
        assert check.status is doctor.Status.WARN
        assert "hf download" in (check.hint or "")

    def test_an_unwritable_cache_dir_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(doctor, "cache_root", const(Path("/proc/nope/cache")))
        assert doctor.check_cache_dir().status is doctor.Status.FAIL

    def test_low_disk_warns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Usage(NamedTuple):
            total: int
            used: int
            free: int

        def tiny(_p: object) -> Usage:
            return Usage(total=10**9, used=10**9, free=10**8)

        monkeypatch.setattr(shutil, "disk_usage", tiny)
        assert doctor.check_disk().status is doctor.Status.WARN


class TestVendorRows:
    def test_a_present_nvidia_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, "has_nvidia_gpu", const(True))
        rows = {c.name: c for c in doctor.check_vendors()}
        assert rows["nvidia"].status is doctor.Status.PASS

    def test_a_present_amd_is_reported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(probe, "has_amd_gpu", const(True))
        rows = {c.name: c for c in doctor.check_vendors()}
        assert rows["amd"].status is doctor.Status.PASS
