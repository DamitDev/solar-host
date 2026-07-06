"""Tests for resources/usage.py — actual usage collection helpers (S-034)."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from solar_host.resources.usage import (
    _dir_size_gb,
    _get_container_pid,
    collect_container_ram_gb,
    collect_container_vram_gb,
    collect_workspace_disk_gb,
)

# ---------------------------------------------------------------------------
# _dir_size_gb
# ---------------------------------------------------------------------------


class TestDirSizeGb:
    def test_sums_files_in_directory(self, tmp_path: Path):
        (tmp_path / "a.bin").write_bytes(b"\x00" * (1024**2))  # 1 MB
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.bin").write_bytes(b"\x00" * (1024**2))  # 1 MB
        result = _dir_size_gb(tmp_path)
        assert result is not None
        # 2 MB = 0.001953 GB; rounded to 4 decimal places = 0.002
        assert result == pytest.approx(2 * 1024**2 / 1024**3, abs=5e-4)

    def test_nonexistent_path_returns_none(self, tmp_path: Path):
        assert _dir_size_gb(tmp_path / "ghost") is None


# ---------------------------------------------------------------------------
# collect_workspace_disk_gb
# ---------------------------------------------------------------------------


class TestCollectWorkspaceDiskGb:
    def test_returns_size_of_workspace_directory(self, tmp_path: Path):
        workspace = tmp_path / "job-001"
        workspace.mkdir()
        (workspace / "file.bin").write_bytes(b"\x00" * (1024**2))
        result = asyncio.run(collect_workspace_disk_gb("job-001", str(tmp_path)))
        assert result is not None
        # 1 MB = 0.000977 GB; rounded to 4 decimal places = 0.001
        assert result == pytest.approx(1024**2 / 1024**3, abs=5e-4)

    def test_missing_workspace_returns_none(self, tmp_path: Path):
        result = asyncio.run(collect_workspace_disk_gb("job-ghost", str(tmp_path)))
        assert result is None


# ---------------------------------------------------------------------------
# collect_container_ram_gb
# ---------------------------------------------------------------------------


class TestCollectContainerRamGb:
    def test_happy_path_subtracts_cache(self):
        docker_svc = MagicMock()
        docker_svc.container_stats.return_value = {
            "memory_stats": {
                "usage": int(6.0 * 1024**3),  # 6 GB raw
                "stats": {"cache": int(2.0 * 1024**3)},  # 2 GB cache (cgroup v1)
            }
        }
        result = asyncio.run(collect_container_ram_gb(docker_svc, "cid-abc"))
        assert result is not None
        assert result == pytest.approx(4.0, rel=1e-3)

    def test_cgroup_v2_subtracts_inactive_file(self):
        """cgroup v2 has no 'cache' key — inactive_file is the page cache."""
        docker_svc = MagicMock()
        docker_svc.container_stats.return_value = {
            "memory_stats": {
                "usage": int(6.0 * 1024**3),  # 6 GB raw
                "stats": {
                    "inactive_file": int(2.0 * 1024**3),  # 2 GB reclaimable cache
                    "active_file": int(1.0 * 1024**3),
                },
            }
        }
        result = asyncio.run(collect_container_ram_gb(docker_svc, "cid-abc"))
        assert result is not None
        assert result == pytest.approx(4.0, rel=1e-3)

    def test_missing_usage_returns_none(self):
        docker_svc = MagicMock()
        docker_svc.container_stats.return_value = {"memory_stats": {}}
        result = asyncio.run(collect_container_ram_gb(docker_svc, "cid-abc"))
        assert result is None

    def test_empty_stats_returns_none(self):
        docker_svc = MagicMock()
        docker_svc.container_stats.return_value = {}
        result = asyncio.run(collect_container_ram_gb(docker_svc, "cid-abc"))
        assert result is None


# ---------------------------------------------------------------------------
# collect_container_vram_gb
# ---------------------------------------------------------------------------


class TestCollectContainerVramGb:
    def test_sums_gpu_memory_for_pid_tree(self):
        docker_svc = MagicMock()
        # Simulate container with root PID 1000 and child 1001
        container_mock = MagicMock()
        container_mock.attrs = {"State": {"Pid": 1000}}
        docker_svc._client.containers.get.return_value = container_mock

        with (
            patch(
                "solar_host.resources.usage._collect_pid_tree",
                return_value={1000, 1001},
            ),
            patch(
                "solar_host.resources.usage.get_gpu_process_memory",
                return_value={
                    1000: int(2.0 * 1024**3),
                    1001: int(1.5 * 1024**3),
                    9999: int(4.0 * 1024**3),  # unrelated PID — should be ignored
                },
            ),
        ):
            result = asyncio.run(collect_container_vram_gb(docker_svc, "cid-abc"))

        assert result is not None
        assert result == pytest.approx(3.5, rel=1e-3)

    def test_no_gpu_memory_returns_none(self):
        docker_svc = MagicMock()
        container_mock = MagicMock()
        container_mock.attrs = {"State": {"Pid": 1000}}
        docker_svc._client.containers.get.return_value = container_mock

        with (
            patch(
                "solar_host.resources.usage._collect_pid_tree",
                return_value={1000},
            ),
            patch(
                "solar_host.resources.usage.get_gpu_process_memory",
                return_value={},
            ),
        ):
            result = asyncio.run(collect_container_vram_gb(docker_svc, "cid-abc"))

        # GPU memory map is empty — returns None (no GPU usage data at all)
        assert result is None

    def test_pid_zero_returns_none(self):
        docker_svc = MagicMock()
        container_mock = MagicMock()
        container_mock.attrs = {"State": {"Pid": 0}}
        docker_svc._client.containers.get.return_value = container_mock

        result = asyncio.run(collect_container_vram_gb(docker_svc, "cid-abc"))
        assert result is None


# ---------------------------------------------------------------------------
# _get_container_pid
# ---------------------------------------------------------------------------


class TestGpuProcessMemorySentinel:
    """get_gpu_process_memory must discard NVML sentinel / None values so a
    process whose per-PID memory is unavailable doesn't inflate the total."""

    def test_sentinel_and_none_values_excluded(self, monkeypatch):
        from solar_host.memory_monitor import get_gpu_process_memory

        class _Proc:
            def __init__(self, pid: int, mem) -> None:
                self.pid = pid
                self.usedGpuMemory = mem

        fake = types.ModuleType("pynvml")
        fake.nvmlInit = lambda: None
        fake.nvmlShutdown = lambda: None
        fake.nvmlDeviceGetCount = lambda: 1
        fake.nvmlDeviceGetHandleByIndex = lambda i: object()
        fake.nvmlDeviceGetComputeRunningProcesses = lambda h: [
            _Proc(100, 2 * 1024**3),  # valid
            _Proc(101, 2**64 - 1),  # NVML_VALUE_NOT_AVAILABLE sentinel
            _Proc(102, None),  # binding returned None
        ]
        fake.nvmlDeviceGetGraphicsRunningProcesses = lambda h: []
        monkeypatch.setitem(sys.modules, "pynvml", fake)

        result = get_gpu_process_memory()
        assert result == {100: 2 * 1024**3}


class TestGetContainerPid:
    def test_extracts_pid_from_container_attrs(self):
        docker_svc = MagicMock()
        container_mock = MagicMock()
        container_mock.attrs = {"State": {"Pid": 12345}}
        docker_svc._client.containers.get.return_value = container_mock

        result = _get_container_pid(docker_svc, "cid-abc")
        assert result == 12345

    def test_returns_none_on_exception(self):
        docker_svc = MagicMock()
        docker_svc._client.containers.get.side_effect = RuntimeError("fail")

        result = _get_container_pid(docker_svc, "cid-abc")
        assert result is None
