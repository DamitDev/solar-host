"""
Memory monitoring for GPU VRAM (NVIDIA) and system RAM (macOS),
GPU type detection, and disk usage reporting.
"""

import platform
import shutil
import time
from pathlib import Path
from typing import Optional, Dict, Union
import psutil

# Cache for memory info to avoid excessive polling
_memory_cache: Optional[Dict] = None
_cache_timestamp: float = 0
CACHE_DURATION = 5.0  # seconds

# GPU type is constant for the lifetime of the process
_gpu_type_cache: Optional[str] = None


def get_memory_info() -> Optional[Dict[str, Union[float, str]]]:
    """
    Get memory information based on platform.

    Returns dict with:
    - used_gb: Used memory in GB
    - total_gb: Total memory in GB
    - available_gb: Memory available for new workloads (total - used)
    - percent: Usage percentage
    - memory_type: "VRAM" or "RAM"

    Returns None if memory info cannot be obtained.
    """
    global _memory_cache, _cache_timestamp

    # Return cached data if still valid
    current_time = time.time()
    if _memory_cache and (current_time - _cache_timestamp) < CACHE_DURATION:
        return _memory_cache

    system = platform.system()

    if system == "Darwin":
        result = _get_mac_memory()
    else:
        result = _get_nvidia_memory()
        if result is None:
            result = _get_system_memory()

    # Update cache
    if result:
        _memory_cache = result
        _cache_timestamp = current_time

    return result


def detect_gpu_type() -> str:
    """Detect the acceleration backend available on this host.

    Returns one of: "nvidia_cuda", "apple_mps", "cpu".
    Result is cached for the lifetime of the process.
    """
    global _gpu_type_cache
    if _gpu_type_cache is not None:
        return _gpu_type_cache

    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            pynvml.nvmlShutdown()
            _gpu_type_cache = "nvidia_cuda"
            return _gpu_type_cache
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if platform.system() == "Darwin":
        _gpu_type_cache = "apple_mps"
        return _gpu_type_cache

    _gpu_type_cache = "cpu"
    return _gpu_type_cache


def _get_nvidia_memory() -> Optional[Dict[str, Union[float, str]]]:
    """Get combined VRAM from all NVIDIA GPUs."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                return None

            total_used = 0
            total_capacity = 0
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_used += info.used
                total_capacity += info.total
        finally:
            pynvml.nvmlShutdown()

        used_gb = total_used / (1024**3)
        total_gb = total_capacity / (1024**3)
        percent = (total_used / total_capacity * 100) if total_capacity > 0 else 0
        available_gb = total_gb - used_gb

        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent": round(percent, 2),
            "memory_type": "VRAM",
        }
    except Exception:
        return None


def _get_system_memory() -> Optional[Dict[str, Union[float, str]]]:
    """Get system RAM info via psutil (fallback for Linux without NVIDIA)."""
    try:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        available_gb = total_gb - used_gb
        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent": round(mem.percent, 2),
            "memory_type": "RAM",
        }
    except Exception:
        return None


def _get_mac_memory() -> Optional[Dict[str, Union[float, str]]]:
    """Get unified memory info on macOS."""
    try:
        mem = psutil.virtual_memory()

        # Convert bytes to GB
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        percent = mem.percent

        available_gb = total_gb - used_gb

        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent": round(percent, 2),
            "memory_type": "RAM",
        }
    except Exception:
        return None


def get_disk_info(path: str) -> Optional[Dict[str, float]]:
    """Return disk usage stats (in GB) for the filesystem containing *path*.

    Walks up to the nearest existing parent if *path* itself doesn't exist.
    Returns None if usage cannot be determined.
    """
    try:
        target = Path(path).resolve()
        while not target.exists():
            target = target.parent
        usage = shutil.disk_usage(target)
        return {
            "total_gb": round(usage.total / (1024**3), 2),
            "used_gb": round(usage.used / (1024**3), 2),
            "available_gb": round(usage.free / (1024**3), 2),
        }
    except Exception:
        return None
