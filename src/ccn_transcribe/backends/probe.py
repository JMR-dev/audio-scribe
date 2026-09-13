"""Hardware probing: cheap, and it imports nothing heavier than pathlib.

The registry calls these before any toolchain import, which is what structurally
guarantees torch is never imported on a machine with no NVIDIA or AMD GPU.

Detection reads each render node's *bound driver*, not loaded kernel modules:
/sys/module/xe can exist with zero bound devices while i915 actually owns the
card, so a module-presence check false-positives.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

VENDORS = {"0x8086": "intel", "0x10de": "nvidia", "0x1002": "amd"}
INTEL_DRIVERS = ("i915", "xe")


@dataclass(frozen=True, slots=True)
class SysfsRoots:
    drm: Path
    dev_dri: Path
    kfd: Path


DEFAULT_ROOTS = SysfsRoots(
    drm=Path("/sys/class/drm"),
    dev_dri=Path("/dev/dri"),
    kfd=Path("/dev/kfd"),
)

DEFAULT_PROC_NVIDIA = Path("/proc/driver/nvidia/version")


@dataclass(frozen=True, slots=True)
class RenderNode:
    path: Path
    vendor: str
    driver: str
    mode: int
    writable: bool


def render_nodes(roots: SysfsRoots = DEFAULT_ROOTS) -> list[RenderNode]:
    if not roots.drm.is_dir():
        return []
    nodes: list[RenderNode] = []
    for link in sorted(roots.drm.glob("renderD*")):
        try:
            vendor_id = (link / "device" / "vendor").read_text().strip()
            driver = (link / "device" / "driver").resolve().name
        except OSError:
            continue
        device = roots.dev_dri / link.name
        mode = 0
        writable = False
        if device.exists():
            mode = stat.S_IMODE(device.stat().st_mode)
            writable = os.access(device, os.R_OK | os.W_OK)
        nodes.append(
            RenderNode(
                path=device,
                vendor=VENDORS.get(vendor_id, vendor_id),
                driver=driver,
                mode=mode,
                writable=writable,
            )
        )
    return nodes


def has_intel_gpu(roots: SysfsRoots = DEFAULT_ROOTS) -> bool:
    return any(n.vendor == "intel" and n.driver in INTEL_DRIVERS for n in render_nodes(roots))


def has_amd_gpu(roots: SysfsRoots = DEFAULT_ROOTS) -> bool:
    if roots.kfd.exists():
        return True
    return any(n.vendor == "amd" for n in render_nodes(roots))


def has_nvidia_gpu(
    roots: SysfsRoots = DEFAULT_ROOTS, proc_nvidia: Path = DEFAULT_PROC_NVIDIA
) -> bool:
    if proc_nvidia.exists():
        return True
    return any(n.vendor == "nvidia" for n in render_nodes(roots))
