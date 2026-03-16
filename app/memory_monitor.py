"""
Memory monitoring for GPU VRAM (NVIDIA) and system RAM (macOS).
"""

import platform
import time
from typing import Optional, Dict, Union
import psutil

# Cache for memory info to avoid excessive polling
_memory_cache: Optional[Dict] = None
_cache_timestamp: float = 0
CACHE_DURATION = 5.0  # seconds


def get_memory_info() -> Optional[Dict[str, Union[float, str]]]:
    """
    Get memory information based on platform.

    Returns dict with:
    - used_gb: Used memory in GB
    - total_gb: Total memory in GB
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

    if system == "Darwin":  # macOS
        result = _get_mac_memory()
    else:  # Windows or Linux
        result = _get_nvidia_memory()

    # Update cache
    if result:
        _memory_cache = result
        _cache_timestamp = current_time

    return result


def _get_nvidia_memory() -> Optional[Dict[str, Union[float, str]]]:
    """Get combined VRAM from all NVIDIA GPUs."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
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

        pynvml.nvmlShutdown()

        # Convert bytes to GB
        used_gb = total_used / (1024**3)
        total_gb = total_capacity / (1024**3)
        percent = (total_used / total_capacity * 100) if total_capacity > 0 else 0

        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "percent": round(percent, 2),
            "memory_type": "VRAM",
        }
    except Exception:
        # No NVIDIA GPU or driver not available
        return None


def _get_mac_memory() -> Optional[Dict[str, Union[float, str]]]:
    """Get unified memory info on macOS."""
    try:
        mem = psutil.virtual_memory()

        # Convert bytes to GB
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        percent = mem.percent

        return {
            "used_gb": round(used_gb, 2),
            "total_gb": round(total_gb, 2),
            "percent": round(percent, 2),
            "memory_type": "RAM",
        }
    except Exception:
        return None
