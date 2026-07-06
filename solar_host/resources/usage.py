"""Per-job actual resource usage collection (S-034).

Collects:
- RAM:  container.stats(stream=False) → memory_stats.usage minus cache
- VRAM: pynvml per-PID accounting → sum over container host-PID tree
- Disk: recursive size walk of the job workspace directory
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from solar_host.memory_monitor import get_gpu_process_memory

if TYPE_CHECKING:
    from solar_host.docker.service import DockerService

logger = logging.getLogger(__name__)


async def collect_container_ram_gb(
    docker_service: "DockerService", container_id: str
) -> Optional[float]:
    """Return RAM used by *container_id* in GB, or None on failure.

    Subtracts the reclaimable page cache from the raw ``memory_stats.usage``
    value so that only resident working-set memory is counted (mirrors
    cAdvisor behaviour). Handles both cgroup layouts:

    - cgroup v1 exposes ``memory_stats.stats.cache``.
    - cgroup v2 has no ``cache`` key; the equivalent reclaimable page cache is
      ``inactive_file`` (matching Kubernetes' working-set definition).
    """
    try:
        stats = await asyncio.to_thread(docker_service.container_stats, container_id)
        mem = stats.get("memory_stats", {})
        usage = mem.get("usage")
        if usage is None:
            return None
        stats_map = mem.get("stats") or {}
        if "cache" in stats_map:
            cache = stats_map.get("cache") or 0
        else:
            # cgroup v2: no "cache"; inactive_file is the reclaimable page cache.
            cache = stats_map.get("inactive_file") or 0
        net_bytes = max(0, usage - cache)
        return round(net_bytes / (1024**3), 4)
    except Exception as exc:
        logger.debug("collect_container_ram_gb(%s): %s", container_id, exc)
        return None


async def collect_container_vram_gb(
    docker_service: "DockerService", container_id: str
) -> Optional[float]:
    """Return GPU VRAM used by the container's host-PID tree, in GB.

    Steps:
    1. Retrieve the container's root host PID via the Docker API.
    2. Collect root + all recursive children via psutil.
    3. Sum per-PID GPU memory reported by pynvml across the matching PID set.

    Returns None when pynvml is unavailable or the container has no GPU usage.
    """
    try:
        container_pid = await asyncio.to_thread(
            _get_container_pid, docker_service, container_id
        )
        if not container_pid:
            return None

        pid_set = await asyncio.to_thread(_collect_pid_tree, container_pid)
        if not pid_set:
            return None

        gpu_mem = await asyncio.to_thread(get_gpu_process_memory)
        if not gpu_mem:
            return None

        total_bytes = sum(gpu_mem.get(pid, 0) for pid in pid_set)
        return round(total_bytes / (1024**3), 4)
    except Exception as exc:
        logger.debug("collect_container_vram_gb(%s): %s", container_id, exc)
        return None


async def collect_workspace_disk_gb(job_id: str, jobs_dir: str) -> Optional[float]:
    """Return disk usage of the job workspace directory in GB, or None."""
    try:
        return await asyncio.to_thread(_dir_size_gb, Path(jobs_dir) / job_id)
    except Exception as exc:
        logger.debug("collect_workspace_disk_gb(%s): %s", job_id, exc)
        return None


# ---------------------------------------------------------------------------
# Synchronous helpers (run via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _get_container_pid(
    docker_service: "DockerService", container_id: str
) -> Optional[int]:
    """Return the host-namespace root PID for *container_id*, or None."""
    try:
        container = docker_service._client.containers.get(container_id)
        container.reload()
        pid = container.attrs.get("State", {}).get("Pid")
        return int(pid) if pid else None
    except Exception:
        return None


def _collect_pid_tree(root_pid: int) -> set[int]:
    """Return *root_pid* plus all its recursive children via psutil."""
    try:
        import psutil

        proc = psutil.Process(root_pid)
        pids: set[int] = {root_pid}
        for child in proc.children(recursive=True):
            pids.add(child.pid)
        return pids
    except Exception:
        return {root_pid}


def _dir_size_gb(path: Path) -> Optional[float]:
    """Recursively sum file sizes under *path* and return GB, or None."""
    if not path.exists():
        return None
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return round(total / (1024**3), 4)
