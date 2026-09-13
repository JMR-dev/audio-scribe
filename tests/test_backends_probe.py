from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ccn_transcribe.backends import probe

if TYPE_CHECKING:
    from pathlib import Path

INTEL = "0x8086"
NVIDIA = "0x10de"
AMD = "0x1002"


@pytest.fixture
def sysfs(tmp_path: Path) -> probe.SysfsRoots:
    (tmp_path / "sys").mkdir()
    (tmp_path / "dev").mkdir()
    return probe.SysfsRoots(drm=tmp_path / "sys", dev_dri=tmp_path / "dev", kfd=tmp_path / "kfd")


def add_node(roots: probe.SysfsRoots, name: str, vendor: str, driver: str) -> None:
    node = roots.drm / name
    device = node / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    # The bound driver is a symlink; its basename is the driver name.
    driver_dir = roots.drm.parent / "drivers" / driver
    driver_dir.mkdir(parents=True, exist_ok=True)
    (device / "driver").symlink_to(driver_dir)
    (roots.dev_dri / name).write_text("")


class TestRenderNodes:
    def test_empty_when_there_is_no_drm_tree(self, tmp_path: Path) -> None:
        roots = probe.SysfsRoots(drm=tmp_path / "nope", dev_dri=tmp_path / "d", kfd=tmp_path / "k")
        assert probe.render_nodes(roots) == []

    def test_reads_vendor_and_bound_driver(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        [node] = probe.render_nodes(sysfs)
        assert node.vendor == "intel"
        assert node.driver == "i915"

    def test_unknown_vendor_id_is_passed_through(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", "0xbeef", "weird")
        assert probe.render_nodes(sysfs)[0].vendor == "0xbeef"

    def test_nodes_without_a_vendor_file_are_skipped(self, sysfs: probe.SysfsRoots) -> None:
        (sysfs.drm / "renderD200" / "device").mkdir(parents=True)
        assert probe.render_nodes(sysfs) == []

    def test_only_render_nodes_are_listed(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        add_node(sysfs, "card1", INTEL, "i915")
        assert [n.path.name for n in probe.render_nodes(sysfs)] == ["renderD128"]

    def test_reports_permissions(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        (sysfs.dev_dri / "renderD128").chmod(0o666)
        node = probe.render_nodes(sysfs)[0]
        assert node.mode == 0o666
        assert node.writable is True

    def test_a_node_with_no_device_file_is_not_writable(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        (sysfs.dev_dri / "renderD128").unlink()
        assert probe.render_nodes(sysfs)[0].writable is False


class TestVendorProbes:
    def test_intel_detected_via_i915(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        assert probe.has_intel_gpu(sysfs) is True

    def test_intel_detected_via_xe(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "xe")
        assert probe.has_intel_gpu(sysfs) is True

    def test_intel_not_detected_when_the_driver_is_unbound(self, sysfs: probe.SysfsRoots) -> None:
        # /sys/module/xe can exist with zero bound devices, which is why the
        # bound-driver symlink is the probe rather than module presence.
        add_node(sysfs, "renderD128", INTEL, "vfio-pci")
        assert probe.has_intel_gpu(sysfs) is False

    def test_no_intel_when_the_box_has_none(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", AMD, "amdgpu")
        assert probe.has_intel_gpu(sysfs) is False

    def test_amd_detected_via_render_node(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", AMD, "amdgpu")
        assert probe.has_amd_gpu(sysfs) is True

    def test_amd_detected_via_kfd(self, sysfs: probe.SysfsRoots) -> None:
        sysfs.kfd.write_text("")
        assert probe.has_amd_gpu(sysfs) is True

    def test_no_amd_on_an_intel_box(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        assert probe.has_amd_gpu(sysfs) is False

    def test_nvidia_detected_via_proc(self, sysfs: probe.SysfsRoots, tmp_path: Path) -> None:
        proc = tmp_path / "nvidia_version"
        proc.write_text("NVRM version ...")
        assert probe.has_nvidia_gpu(sysfs, proc_nvidia=proc) is True

    def test_nvidia_detected_via_render_node(self, sysfs: probe.SysfsRoots) -> None:
        # The proprietary stack does not guarantee a DRM render node, so sysfs is
        # the fallback rather than the primary signal.
        add_node(sysfs, "renderD128", NVIDIA, "nvidia-drm")
        assert probe.has_nvidia_gpu(sysfs, proc_nvidia=sysfs.drm / "absent") is True

    def test_no_nvidia_on_an_intel_box(self, sysfs: probe.SysfsRoots) -> None:
        add_node(sysfs, "renderD128", INTEL, "i915")
        assert probe.has_nvidia_gpu(sysfs, proc_nvidia=sysfs.drm / "absent") is False


class TestAgainstTheRealMachine:
    def test_defaults_point_at_the_real_system(self) -> None:
        assert probe.DEFAULT_ROOTS.drm.name == "drm"

    def test_intel_igpu_is_detected_here(self) -> None:
        # This box is an Iris Xe on i915; if this fails the probe is broken.
        assert probe.has_intel_gpu() is True

    def test_no_nvidia_or_amd_here(self) -> None:
        assert probe.has_nvidia_gpu() is False
        assert probe.has_amd_gpu() is False
