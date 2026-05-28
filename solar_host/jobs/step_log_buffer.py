"""Bounded per-step in-memory log buffer with batched Socket.IO emission.

Mirrors the deque + queue + flush pattern used by ProcessManager for instance
logs, but keyed by (job_id, step_name) instead of instance_id.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections import deque
from datetime import UTC, datetime

from solar_host.jobs.models import StepLogMessage
from solar_host.ws_client import broadcast_step_log_batch

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_S = 0.1
MAX_QUEUE_SIZE = 10_000


class StepLogBuffer:
    """Bounded per-step log buffer with thread-safe append and async flush."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._maxlen = maxlen
        self._lock = threading.Lock()

        # (job_id, step_name) → bounded deque[StepLogMessage]
        self._buffers: dict[tuple[str, str], deque[StepLogMessage]] = {}
        # (job_id, step_name) → monotonic seq counter
        self._seqs: dict[tuple[str, str], int] = {}

        # Emit queue stores full dicts ready for broadcast (no StepLogMessage overhead).
        self._queue: queue.Queue[dict] = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    # ------------------------------------------------------------------
    # Thread-safe write API (called from executor thread pool)
    # ------------------------------------------------------------------

    def append(
        self,
        job_id: str,
        step_name: str,
        step_index: int,
        stream: str,
        line: str,
    ) -> None:
        """Append one log line to the deque and enqueue for emission."""
        key = (job_id, step_name)
        now = datetime.now(UTC)

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = deque(maxlen=self._maxlen)
                self._seqs[key] = 0

            seq = self._seqs[key]
            self._seqs[key] = seq + 1

            msg = StepLogMessage(seq=seq, timestamp=now, stream=stream, line=line)  # type: ignore[arg-type]
            self._buffers[key].append(msg)

        entry = {
            "job_id": job_id,
            "step_name": step_name,
            "step_index": step_index,
            "stream": stream,
            "seq": seq,
            "timestamp": now.isoformat(),
            "line": line,
        }
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            pass

    def mark_completed(
        self,
        job_id: str,
        step_name: str,
        step_index: int,
        exit_code: int | None,
    ) -> None:
        """Enqueue a completion marker entry for this step."""
        key = (job_id, step_name)
        now = datetime.now(UTC)

        with self._lock:
            seq = self._seqs.get(key, 0)
            self._seqs[key] = seq + 1

            # Also store completion in the in-memory buffer.
            if key in self._buffers:
                msg = StepLogMessage(
                    seq=seq,
                    timestamp=now,
                    stream="stdout",
                    line="",
                    completed=True,
                    exit_code=exit_code,
                )
                self._buffers[key].append(msg)

        entry = {
            "job_id": job_id,
            "step_name": step_name,
            "step_index": step_index,
            "stream": "stdout",
            "seq": seq,
            "timestamp": now.isoformat(),
            "line": "",
            "completed": True,
            "exit_code": exit_code,
        }
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get_buffer(self, job_id: str, step_name: str) -> list[StepLogMessage]:
        """Return a snapshot of the in-memory buffer for a step."""
        key = (job_id, step_name)
        with self._lock:
            buf = self._buffers.get(key)
            return list(buf) if buf else []

    def remove(self, job_id: str) -> None:
        """Purge all step buffers for a finished job."""
        with self._lock:
            keys = [k for k in self._buffers if k[0] == job_id]
            for k in keys:
                del self._buffers[k]
                self._seqs.pop(k, None)

    # ------------------------------------------------------------------
    # Async flush helpers (called from event loop)
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Drain the emit queue and broadcast entries to Solar Control."""
        entries: list[dict] = []
        while True:
            try:
                entries.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if entries:
            await broadcast_step_log_batch(entries)


# Module singleton — imported by step_executor and main.py.
def _make_step_log_buffer() -> StepLogBuffer:
    from solar_host.config import settings

    return StepLogBuffer(maxlen=settings.log_buffer_size)


step_log_buffer: StepLogBuffer = _make_step_log_buffer()


async def flush_step_logs() -> None:
    """Drain the singleton emit queue and broadcast to Solar Control."""
    await step_log_buffer.flush()


async def step_log_flush_loop() -> None:
    """Flush step log batches every FLUSH_INTERVAL_S seconds."""
    while True:
        try:
            await asyncio.sleep(FLUSH_INTERVAL_S)
            await flush_step_logs()
        except asyncio.CancelledError:
            await flush_step_logs()
            break
        except Exception:
            logger.exception("Error in step log flush loop")
            await asyncio.sleep(1)
