"""Unit tests for solar_host.jobs.JobExecutor (job-level orchestration).

Per-step Docker behaviour is tested in test_job_step_executor.py.
All Docker calls are mocked — no real Docker daemon or filesystem writes needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solar_host.config import Settings
from solar_host.docker.errors import ContainerNonZeroExitError, ContainerStartError
from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.executor import JobExecutor
from solar_host.jobs.models import (
    JobDefinition,
    JobStatus,
    StepDefinition,
    StepStatus,
)
from solar_host.jobs.store import JobStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_SETTINGS = Settings(
    jobs_dir="/tmp/solar-test-jobs",
    container_uid=1000,
    container_gid=1000,
    hf_cache_dir="/tmp/solar-test-hf-cache",
    harbor_url="http://harbor.example.com",
    harbor_username="user",
    harbor_password="pass",
    hf_token="hf-token-xyz",
)

_WORKSPACE = Path("/tmp/solar-test-jobs/test-job-001")

_EXECUTOR_MODULE = "solar_host.jobs.executor"
_STEP_MODULE = "solar_host.jobs.step_executor"


def _make_step(
    name: str,
    is_preparation_step: bool = False,
    environment: dict[str, str] | None = None,
) -> StepDefinition:
    return StepDefinition(
        name=name,
        image="test/img:latest",
        is_preparation_step=is_preparation_step,
        environment=environment or {},
    )


def _make_job(
    steps: list[StepDefinition], job_id: str = "test-job-001"
) -> JobDefinition:
    return JobDefinition(job_id=job_id, name="Test Job", steps=steps)


def _make_executor(
    docker_service: MagicMock | None = None,
    store: JobStore | None = None,
) -> tuple[JobExecutor, MagicMock, JobStore]:
    ds = docker_service or MagicMock()
    ds.create_container.return_value = "container-abc"
    ds.start_container.return_value = None
    ds.wait_container.return_value = 0
    ds.remove_container.return_value = None
    ds.stop_container.return_value = None
    ds.stream_logs.return_value = iter([])

    st = store or JobStore()
    return JobExecutor(docker_service=ds, store=st, settings=_TEST_SETTINGS), ds, st


# Standard workspace patches applied to every test in this file.
_WORKSPACE_PATCHES = (
    patch(f"{_EXECUTOR_MODULE}.validate_job_id"),
    patch(f"{_EXECUTOR_MODULE}.check_disk_space"),
    patch(f"{_EXECUTOR_MODULE}.create_workspace", return_value=_WORKSPACE),
    patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"),
)


# ---------------------------------------------------------------------------
# Successful 3-step job
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_three_step_success() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b"), _make_step("c")])
    ds.create_container.side_effect = ["c1", "c2", "c3"]

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.completed
    assert all(s.status == StepStatus.completed for s in result.steps)
    assert ds.create_container.call_count == 3
    assert ds.remove_container.call_count == 3


# ---------------------------------------------------------------------------
# Fail-fast: step 2 fails → step 3 is cancelled
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_fail_fast_on_step_failure() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b"), _make_step("c")])
    ds.create_container.side_effect = ["c1", "c2", "c3"]
    ds.wait_container.side_effect = [
        0,
        ContainerNonZeroExitError("c2", 1, ["ERROR line"]),
        0,
    ]

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.failed
    assert result.steps[0].status == StepStatus.completed
    assert result.steps[1].status == StepStatus.failed
    assert result.steps[1].exit_code == 1
    assert "ERROR line" in (result.steps[1].error_message or "")
    assert result.steps[2].status == StepStatus.cancelled
    assert ds.create_container.call_count == 2  # step-c never created


# ---------------------------------------------------------------------------
# Cancellation during step 2
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancellation_during_step_2() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b"), _make_step("c")])
    ds.create_container.side_effect = ["c1", "c2", "c3"]

    original_to_thread = asyncio.to_thread

    async def patched_to_thread(fn, *args, **kwargs):  # type: ignore[misc]
        if fn is ds.wait_container and args and args[0] == "c2":
            await executor.cancel_job(job_def.job_id)
            return 0
        return await original_to_thread(fn, *args, **kwargs)

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch("asyncio.to_thread", side_effect=patched_to_thread),
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.cancelled
    assert result.steps[2].status == StepStatus.cancelled


# ---------------------------------------------------------------------------
# InsufficientDiskError raised before a step
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_insufficient_disk_raises_and_fails_job() -> None:
    executor, _, store = _make_executor()
    job_def = _make_job([_make_step("a")])
    disk_error = InsufficientDiskError(required_gb=10.0, available_gb=2.0)

    # First call (pre-workspace) passes; second call (pre-step) raises.
    with (
        patch(f"{_EXECUTOR_MODULE}.validate_job_id"),
        patch(f"{_EXECUTOR_MODULE}.check_disk_space", side_effect=[None, disk_error]),
        patch(f"{_EXECUTOR_MODULE}.create_workspace", return_value=_WORKSPACE),
        _WORKSPACE_PATCHES[3],
        pytest.raises(InsufficientDiskError),
    ):
        await executor.run_job(job_def)

    assert store.get(job_def.job_id).status == JobStatus.failed  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Container start failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_container_start_failure_fails_job() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b")])
    ds.create_container.side_effect = ContainerStartError("solar-job-…-a", "API error")

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.failed
    assert result.steps[0].status == StepStatus.failed
    assert result.steps[1].status == StepStatus.cancelled


# ---------------------------------------------------------------------------
# Lifecycle event emission
# ---------------------------------------------------------------------------

_BROADCAST = "solar_host.jobs.events.broadcast_job_lifecycle"


@pytest.mark.anyio
async def test_job_started_emitted_on_run() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a")])
    ds.create_container.return_value = "c1"

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        await executor.run_job(job_def)

    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "job_started" in calls


@pytest.mark.anyio
async def test_job_completed_emitted_on_success() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a")])
    ds.create_container.return_value = "c1"

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.completed
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "job_completed" in calls
    assert "job_failed" not in calls
    assert "job_cancelled" not in calls


@pytest.mark.anyio
async def test_job_failed_emitted_on_step_failure() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a")])
    ds.wait_container.side_effect = ContainerNonZeroExitError("c1", 1, ["err"])

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.failed
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "job_failed" in calls
    assert "job_completed" not in calls


@pytest.mark.anyio
async def test_job_cancelled_emitted_on_cancellation() -> None:
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b")])
    ds.create_container.side_effect = ["c1", "c2"]

    original_to_thread = asyncio.to_thread

    async def patched_to_thread(fn, *args, **kwargs):  # type: ignore[misc]
        if fn is ds.wait_container and args and args[0] == "c1":
            await executor.cancel_job(job_def.job_id)
            return 0
        return await original_to_thread(fn, *args, **kwargs)

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch("asyncio.to_thread", side_effect=patched_to_thread),
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.cancelled
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "job_cancelled" in calls
    assert "job_completed" not in calls
    assert "job_failed" not in calls


@pytest.mark.anyio
async def test_no_duplicate_terminal_job_events_on_success() -> None:
    """job_completed emitted exactly once when all steps succeed."""
    executor, ds, _ = _make_executor()
    job_def = _make_job([_make_step("a"), _make_step("b")])
    ds.create_container.side_effect = ["c1", "c2"]

    with (
        _WORKSPACE_PATCHES[0],
        _WORKSPACE_PATCHES[1],
        _WORKSPACE_PATCHES[2],
        _WORKSPACE_PATCHES[3],
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        await executor.run_job(job_def)

    terminal_events = [
        call.args[0]
        for call in mock_bc.call_args_list
        if call.args[0] in ("job_completed", "job_failed", "job_cancelled")
    ]
    assert terminal_events.count("job_completed") == 1
    assert len(terminal_events) == 1
