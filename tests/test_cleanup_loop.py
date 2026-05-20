"""Unit tests for solar_host.jobs.executor.cleanup_loop."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from solar_host.jobs.executor import cleanup_loop
from solar_host.jobs.models import JobState, JobStatus
from solar_host.jobs.store import JobStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXECUTOR_MODULE = "solar_host.jobs.executor"


def _make_job(
    job_id: str,
    status: JobStatus,
    finished_at: datetime | None,
    retention_hours: float = 1.0,
    workspace_path: str = "/tmp/ws/job1",
) -> JobState:
    return JobState(
        job_id=job_id,
        name=f"job-{job_id}",
        status=status,
        workspace_path=workspace_path,
        finished_at=finished_at,
        retention_hours=retention_hours,
    )


def _expired_at(hours_ago: float = 2.0) -> datetime:
    """Return a finished_at time that has expired."""
    return datetime.now(UTC) - timedelta(hours=hours_ago)


def _recent() -> datetime:
    """Return a finished_at time that has NOT expired (retention window not elapsed)."""
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cleanup_removes_expired_terminal_jobs() -> None:
    """Expired completed/failed/cancelled jobs are deleted and removed from store."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.completed, _expired_at(), retention_hours=1.0))
    store.add(_make_job("j2", JobStatus.failed, _expired_at(), retention_hours=1.0))
    store.add(_make_job("j3", JobStatus.cancelled, _expired_at(), retention_hours=1.0))

    # sleep: first call succeeds (loop body runs), second raises CancelledError (loop exits).
    with patch(f"{_EXECUTOR_MODULE}.delete_workspace") as mock_delete, \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        await cleanup_loop(store, poll_interval_s=0)

    assert store.get("j1") is None
    assert store.get("j2") is None
    assert store.get("j3") is None
    assert mock_delete.call_count == 3


@pytest.mark.anyio
async def test_cleanup_keeps_non_expired_jobs() -> None:
    """Jobs within retention window are left in the store."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.completed, _recent(), retention_hours=24.0))

    with patch(f"{_EXECUTOR_MODULE}.delete_workspace") as mock_delete, \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        await cleanup_loop(store, poll_interval_s=0)

    assert store.get("j1") is not None
    mock_delete.assert_not_called()


@pytest.mark.anyio
async def test_cleanup_skips_running_jobs() -> None:
    """Running jobs are never cleaned up, even if they somehow have a finished_at."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.running, _expired_at(), retention_hours=0.0))

    with patch(f"{_EXECUTOR_MODULE}.delete_workspace") as mock_delete, \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        await cleanup_loop(store, poll_interval_s=0)

    assert store.get("j1") is not None
    mock_delete.assert_not_called()


@pytest.mark.anyio
async def test_cleanup_skips_jobs_without_finished_at() -> None:
    """Terminal jobs with no finished_at are skipped (defensive)."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.completed, finished_at=None, retention_hours=0.0))

    with patch(f"{_EXECUTOR_MODULE}.delete_workspace") as mock_delete, \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        await cleanup_loop(store, poll_interval_s=0)

    assert store.get("j1") is not None
    mock_delete.assert_not_called()


@pytest.mark.anyio
async def test_cleanup_skips_workspace_delete_when_empty_path() -> None:
    """Jobs with empty workspace_path skip the delete_workspace call but are still removed."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.completed, _expired_at(), workspace_path=""))

    with patch(f"{_EXECUTOR_MODULE}.delete_workspace") as mock_delete, \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = [None, asyncio.CancelledError()]
        await cleanup_loop(store, poll_interval_s=0)

    assert store.get("j1") is None
    mock_delete.assert_not_called()


@pytest.mark.anyio
async def test_cleanup_exits_cleanly_on_cancelled_error() -> None:
    """The loop exits without propagating CancelledError when cancelled during sleep."""
    store = JobStore()

    with patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        mock_sleep.side_effect = asyncio.CancelledError()
        await cleanup_loop(store, poll_interval_s=0)  # must return, not raise


@pytest.mark.anyio
async def test_cleanup_runs_multiple_cycles() -> None:
    """Cleanup processes jobs on each iteration; expired job is removed in first cycle."""
    store = JobStore()
    store.add(_make_job("j1", JobStatus.completed, _expired_at(), retention_hours=1.0))

    call_count = 0

    async def controlled_sleep(_interval: float) -> None:
        nonlocal call_count
        call_count += 1
        if call_count >= 3:
            raise asyncio.CancelledError()

    with patch(f"{_EXECUTOR_MODULE}.delete_workspace"), \
         patch(f"{_EXECUTOR_MODULE}.asyncio.sleep", side_effect=controlled_sleep):
        await cleanup_loop(store, poll_interval_s=0)

    # Loop ran 3 sleep calls total; body executed twice (calls 1 and 2 succeeded).
    assert call_count == 3
    # Job was removed in the first body execution.
    assert store.get("j1") is None
