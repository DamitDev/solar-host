"""Lifecycle event builders and emitters for job/step state transitions.

Each ``emit_*`` function constructs a typed payload and calls
:func:`~solar_host.ws_client.broadcast_job_lifecycle` (fire-and-forget).
Events are emitted directly by name so Solar Control can register
per-event handlers (S-032).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from solar_host.ws_client import broadcast_job_lifecycle, get_client

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _host_id() -> str | None:
    """Return the registered host_id, or None if not yet connected/registered."""
    client = get_client()
    return client.host_id if client else None


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Payload builders (pure functions – easy to unit-test)
# ---------------------------------------------------------------------------


def build_job_started(job_id: str, name: str, timestamp: datetime) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "name": name,
        "status": "running",
        "timestamp": _iso(timestamp),
    }


def build_job_completed(
    job_id: str,
    timestamp: datetime,
    workspace_path: str,
    retention_hours: float,
) -> dict:
    retention_deadline = (timestamp + timedelta(hours=retention_hours)).replace(
        tzinfo=UTC if timestamp.tzinfo is None else timestamp.tzinfo
    )
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "status": "completed",
        "timestamp": _iso(timestamp),
        "workspace_path": workspace_path,
        "retention_deadline": _iso(retention_deadline),
    }


def build_job_failed(
    job_id: str, timestamp: datetime, error_message: str | None
) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "status": "failed",
        "timestamp": _iso(timestamp),
        "error_message": error_message,
    }


def build_job_cancelled(job_id: str, timestamp: datetime) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "status": "cancelled",
        "timestamp": _iso(timestamp),
    }


def build_step_started(
    job_id: str, step_name: str, step_index: int, timestamp: datetime
) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "step_name": step_name,
        "step_index": step_index,
        "status": "running",
        "timestamp": _iso(timestamp),
    }


def build_step_completed(
    job_id: str,
    step_name: str,
    step_index: int,
    timestamp: datetime,
    duration_s: float,
    exit_code: int,
) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "step_name": step_name,
        "step_index": step_index,
        "status": "completed",
        "timestamp": _iso(timestamp),
        "duration_s": duration_s,
        "exit_code": exit_code,
    }


def build_step_failed(
    job_id: str,
    step_name: str,
    step_index: int,
    timestamp: datetime,
    duration_s: float,
    exit_code: int | None,
    error_summary: str | None,
) -> dict:
    return {
        "job_id": job_id,
        "host_id": _host_id(),
        "step_name": step_name,
        "step_index": step_index,
        "status": "failed",
        "timestamp": _iso(timestamp),
        "duration_s": duration_s,
        "exit_code": exit_code,
        "error_summary": error_summary,
    }


# ---------------------------------------------------------------------------
# Async emitters
# ---------------------------------------------------------------------------


async def emit_job_started(job_id: str, name: str, timestamp: datetime) -> None:
    await broadcast_job_lifecycle(
        "job_started", build_job_started(job_id, name, timestamp)
    )


async def emit_job_completed(
    job_id: str,
    timestamp: datetime,
    workspace_path: str,
    retention_hours: float,
) -> None:
    await broadcast_job_lifecycle(
        "job_completed",
        build_job_completed(job_id, timestamp, workspace_path, retention_hours),
    )


async def emit_job_failed(
    job_id: str, timestamp: datetime, error_message: str | None
) -> None:
    await broadcast_job_lifecycle(
        "job_failed", build_job_failed(job_id, timestamp, error_message)
    )


async def emit_job_cancelled(job_id: str, timestamp: datetime) -> None:
    await broadcast_job_lifecycle(
        "job_cancelled", build_job_cancelled(job_id, timestamp)
    )


async def emit_step_started(
    job_id: str, step_name: str, step_index: int, timestamp: datetime
) -> None:
    await broadcast_job_lifecycle(
        "step_started", build_step_started(job_id, step_name, step_index, timestamp)
    )


async def emit_step_completed(
    job_id: str,
    step_name: str,
    step_index: int,
    timestamp: datetime,
    duration_s: float,
    exit_code: int,
) -> None:
    await broadcast_job_lifecycle(
        "step_completed",
        build_step_completed(
            job_id, step_name, step_index, timestamp, duration_s, exit_code
        ),
    )


async def emit_step_failed(
    job_id: str,
    step_name: str,
    step_index: int,
    timestamp: datetime,
    duration_s: float,
    exit_code: int | None,
    error_summary: str | None,
) -> None:
    await broadcast_job_lifecycle(
        "step_failed",
        build_step_failed(
            job_id, step_name, step_index, timestamp, duration_s, exit_code, error_summary
        ),
    )
