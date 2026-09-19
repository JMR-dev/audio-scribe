"""Environment diagnosis.

README-intel.md's troubleshooting table turned into code, deliberately two-phase
so it can tell a driver problem from a plugin problem.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from audio_scribe.backends import probe
from audio_scribe.backends.openvino_backend import (
    DEFAULT_MODEL,
    cache_root,
    snapshot,
)
from audio_scribe.media import ffmpeg, ytdlp_opts


class Status(StrEnum):
    PASS = "PASS"  # noqa: S105 - a status, not a credential
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None


def binary_check(name: str) -> Check:
    found = shutil.which(name)
    if found is None:
        return Check(name, Status.FAIL, "not on PATH", f"Install {name}.")
    return Check(name, Status.PASS, found)


def check_python() -> Check:
    return Check("python", Status.PASS, sys.version.split()[0])


def check_openvino() -> Check:
    try:
        import openvino as ov  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        return Check("openvino", Status.FAIL, str(exc), "pip install openvino-genai")
    return Check("openvino", Status.PASS, ov.__version__.split("-")[0])


def check_flac_encoder() -> Check:
    try:
        ok = ffmpeg.has_flac_encoder()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return Check("flac encoder", Status.FAIL, str(exc), "Install ffmpeg.")
    if not ok:
        return Check("flac encoder", Status.FAIL, "absent", "Install an ffmpeg with FLAC.")
    return Check("flac encoder", Status.PASS, "present")


def check_js_runtime() -> Check:
    """YouTube needs one for signature challenges; yt-dlp defaults to deno only."""
    deno = ytdlp_opts.find_runtime("deno")
    if deno:
        return Check("JS runtime", Status.PASS, f"deno at {deno}")
    node = ytdlp_opts.find_node()
    if node and ytdlp_opts.is_nvm_path(node):
        return Check(
            "JS runtime",
            Status.WARN,
            f"only nvm node at {node}",
            "nvm paths are invisible to cron and systemd. Install the deno extra: "
            "uv sync (yt-dlp[deno]).",
        )
    if node:
        return Check("JS runtime", Status.PASS, f"node at {node}")
    return Check(
        "JS runtime",
        Status.FAIL,
        "none found",
        "YouTube extraction will fail. Install it with: uv sync (yt-dlp[deno]).",
    )


def check_render_nodes() -> list[Check]:
    nodes = probe.render_nodes()
    if not nodes:
        return [Check("render nodes", Status.WARN, "none", "No GPU render node found.")]
    checks: list[Check] = []
    for node in nodes:
        checks.append(
            Check(f"render {node.path.name}", Status.PASS, f"{node.vendor}/{node.driver}")
        )
        if not node.writable:
            checks.append(
                Check(
                    f"permission {node.path.name}",
                    Status.FAIL,
                    f"mode {node.mode:o}",
                    "sudo usermod -aG render $USER, then log out and back in.",
                )
            )
        elif node.mode == 0o666:
            checks.append(
                Check(
                    f"permission {node.path.name}",
                    Status.WARN,
                    "mode 0666 (world-writable)",
                    "GPU access works only because the node is world-writable. If an "
                    "update restores 0660, OpenVINO will silently fall back to CPU. "
                    "sudo usermod -aG render $USER makes it robust.",
                )
            )
    return checks


def check_clinfo() -> Check:
    if shutil.which("clinfo") is None:
        return Check("clinfo", Status.WARN, "not installed", "dnf install clinfo")
    try:
        out = subprocess.run(
            ["clinfo", "-l"], capture_output=True, check=False, timeout=20
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as exc:
        return Check("clinfo", Status.WARN, str(exc))
    if "Device" not in out:
        return Check(
            "clinfo",
            Status.FAIL,
            "no OpenCL device",
            "Compute runtime missing: dnf install intel-compute-runtime",
        )
    return Check("clinfo", Status.PASS, out.strip().splitlines()[-1].strip())


def check_devices() -> list[Check]:
    try:
        import openvino as ov  # pylint: disable=import-outside-toplevel

        devices = list(ov.Core().available_devices)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return [Check("openvino devices", Status.FAIL, str(exc))]

    checks = [Check("openvino devices", Status.PASS, ", ".join(devices))]
    has_gpu = any(d == "GPU" or d.startswith("GPU.") for d in devices)
    if probe.has_intel_gpu() and not has_gpu:
        # README-intel.md section 7: the two checks are deliberately separate.
        checks.append(
            Check(
                "gpu plugin",
                Status.FAIL,
                "an Intel GPU is present but OpenVINO shows none",
                "This is a plugin problem, not a driver problem. Check you are in the "
                "venv and not shadowed by a distro python3-openvino.",
            )
        )
    return checks


def check_vendors() -> list[Check]:
    return [
        Check(
            "nvidia",
            Status.PASS if probe.has_nvidia_gpu() else Status.WARN,
            "present" if probe.has_nvidia_gpu() else "absent",
            None if probe.has_nvidia_gpu() else "The NVIDIA backend is interface-only.",
        ),
        Check(
            "amd",
            Status.PASS if probe.has_amd_gpu() else Status.WARN,
            "present" if probe.has_amd_gpu() else "absent",
            None if probe.has_amd_gpu() else "The AMD backend is interface-only.",
        ),
    ]


def check_model() -> Check:
    try:
        path = snapshot(DEFAULT_MODEL, None, local_only=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return Check("model", Status.WARN, "not cached", f"hf download {DEFAULT_MODEL}")
    return Check("model", Status.PASS, path)


def check_cache_dir() -> Check:
    root = cache_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check("kernel cache", Status.FAIL, str(exc))
    return Check("kernel cache", Status.PASS, str(root))


def check_disk(path: Path | None = None) -> Check:
    usage = shutil.disk_usage(path or Path.cwd())
    free_gb = usage.free / 1e9
    status = Status.PASS if free_gb > 5 else Status.WARN
    return Check("disk free", status, f"{free_gb:.0f} GB")


def check_start_method() -> Check:
    import multiprocessing  # pylint: disable=import-outside-toplevel

    method = multiprocessing.get_start_method(allow_none=True) or "default"
    return Check("mp start method", Status.PASS, str(method))


def run_checks() -> list[Check]:
    checks: list[Check] = [check_python(), check_openvino()]
    checks += [binary_check("ffmpeg"), binary_check("ffprobe"), check_flac_encoder()]
    checks += [binary_check("aria2c"), check_js_runtime()]
    checks += check_render_nodes()
    checks.append(check_clinfo())
    checks += check_devices()
    checks += check_vendors()
    checks += [check_model(), check_cache_dir(), check_disk(), check_start_method()]
    return checks


def render(checks: list[Check]) -> str:
    width = max(len(c.name) for c in checks)
    lines: list[str] = []
    for check in checks:
        lines.append(f"{check.status.value:4}  {check.name:<{width}}  {check.detail}")
        if check.hint and check.status is not Status.PASS:
            lines.append(f"      {'':<{width}}  -> {check.hint}")
    failures = sum(1 for c in checks if c.status is Status.FAIL)
    lines.append("")
    lines.append("all checks passed" if not failures else f"{failures} check(s) FAILED")
    return "\n".join(lines)


def worst(checks: list[Check]) -> int:
    return 2 if any(c.status is Status.FAIL for c in checks) else 0
