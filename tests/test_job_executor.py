"""Unit tests for solar_host.jobs.JobExecutor.

All Docker calls are mocked — no real Docker daemon or filesystem writes needed.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from solar_host.config import Settings
from solar_host.docker.errors import ContainerNonZeroExitError, ContainerStartError
from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.executor import JobExecutor
from solar_host.jobs.models import (
    JobDefinition,
    JobState,
    JobStatus,
    StepDefinition,
    StepStatus,
)
from solar_host.jobs.store import JobStore

# ---------------------------------------------------------------------------
# Fixtures / helpers
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


def _make_step(
    name: str,
    image: str = "test/img:latest",
    gpu: bool = False,
    is_preparation_step: bool = False,
    environment: dict[str, str] | None = None,
) -> StepDefinition:
    return StepDefinition(
        name=name,
        image=image,
        gpu=gpu,
        is_preparation_step=is_preparation_step,
        environment=environment or {},
    )


def _make_job(
    steps: list[StepDefinition],
    job_id: str = "test-job-001",
    name: str = "Test Job",
) -> JobDefinition:
    return JobDefinition(job_id=job_id, name=name, steps=steps)


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
    executor = JobExecutor(docker_service=ds, store=st, settings=_TEST_SETTINGS)
    return executor, ds, st


def _patch_workspace() -> Any:
    """Context manager patches: validate_job_id, check_disk_space, create_workspace."""
    workspace_path = Path("/tmp/solar-test-jobs/test-job-001")
    patches = [
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch(
            "solar_host.jobs.executor.create_workspace",
            return_value=workspace_path,
        ),
    ]
    return patches, workspace_path


# ---------------------------------------------------------------------------
# Helper to run all patches together
# ---------------------------------------------------------------------------

def _run_with_workspace_patches(coro: Any) -> Any:
    """Apply standard workspace patches and run coroutine, returning result."""
    patches, workspace_path = _patch_workspace()
    import contextlib

    @contextlib.contextmanager
    def all_patches():
        with patches[0], patches[1], patches[2]:
            yield workspace_path

    return all_patches, coro


# ---------------------------------------------------------------------------
# Successful 3-step job
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_three_step_success() -> None:
    executor, ds, store = _make_executor()
    steps = [_make_step("step-a"), _make_step("step-b"), _make_step("step-c")]
    job_def = _make_job(steps)

    ds.create_container.side_effect = [
        "container-1",
        "container-2",
        "container-3",
    ]

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
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
    executor, ds, store = _make_executor()
    steps = [_make_step("step-a"), _make_step("step-b"), _make_step("step-c")]
    job_def = _make_job(steps)

    # step-a succeeds, step-b raises non-zero exit, step-c should not run.
    ds.create_container.side_effect = ["container-1", "container-2", "container-3"]
    ds.wait_container.side_effect = [
        0,
        ContainerNonZeroExitError("container-2", 1, ["ERROR line"]),
        0,
    ]

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.failed
    assert result.steps[0].status == StepStatus.completed
    assert result.steps[1].status == StepStatus.failed
    assert result.steps[1].exit_code == 1
    assert "ERROR line" in (result.steps[1].error_message or "")
    assert result.steps[2].status == StepStatus.cancelled
    # step-c container should never have been created.
    assert ds.create_container.call_count == 2


# ---------------------------------------------------------------------------
# Cancellation during step 2
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_cancellation_during_step_2() -> None:
    executor, ds, store = _make_executor()
    steps = [_make_step("step-a"), _make_step("step-b"), _make_step("step-c")]
    job_def = _make_job(steps)

    ds.create_container.side_effect = ["container-1", "container-2", "container-3"]

    cancel_called = asyncio.Event()

    async def _wait_side_effect(container_id: str) -> int:
        if container_id == "container-2":
            # Signal cancellation then let wait_container "block" until stopped.
            await executor.cancel_job(job_def.job_id)
            # Return 0 to simulate container finishing after stop.
        return 0

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        # Replace wait_container with an async wrapper via to_thread mock.
        original_to_thread = asyncio.to_thread

        async def patched_to_thread(fn, *args, **kwargs):  # type: ignore[misc]
            if fn is ds.wait_container and args and args[0] == "container-2":
                await executor.cancel_job(job_def.job_id)
                return 0
            return await original_to_thread(fn, *args, **kwargs)

        with patch("asyncio.to_thread", side_effect=patched_to_thread):
            result = await executor.run_job(job_def)

    assert result.status == JobStatus.cancelled
    # step-c must be cancelled.
    assert result.steps[2].status == StepStatus.cancelled


# ---------------------------------------------------------------------------
# InsufficientDiskError before a step
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_insufficient_disk_error_before_step() -> None:
    executor, ds, store = _make_executor()
    steps = [_make_step("step-a")]
    job_def = _make_job(steps)

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    disk_error = InsufficientDiskError(required_gb=10.0, available_gb=2.0)

    # First check_disk_space call (before workspace creation) passes;
    # second call (pre-step) raises.
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch(
            "solar_host.jobs.executor.check_disk_space",
            side_effect=[None, disk_error],
        ),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        with pytest.raises(InsufficientDiskError):
            await executor.run_job(job_def)

    job = store.get(job_def.job_id)
    assert job is not None
    assert job.status == JobStatus.failed


# ---------------------------------------------------------------------------
# Container start failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_container_start_failure() -> None:
    executor, ds, store = _make_executor()
    steps = [_make_step("step-a"), _make_step("step-b")]
    job_def = _make_job(steps)

    ds.create_container.side_effect = ContainerStartError(
        "solar-job-test-job-001-step-a", "API error"
    )

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.failed
    assert result.steps[0].status == StepStatus.failed
    assert result.steps[1].status == StepStatus.cancelled


# ---------------------------------------------------------------------------
# Environment building
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_environment_building() -> None:
    executor, ds, store = _make_executor()
    steps = [_make_step("train", environment={"CUSTOM_VAR": "custom_val"})]
    job_def = _make_job(steps)

    captured_envs: list[dict[str, str]] = []
    original_create = ds.create_container

    def capture_create(image, job_id, step_name, environment, gpu=False, is_preparation_step=False):  # type: ignore[misc]
        captured_envs.append(environment)
        return "container-1"

    ds.create_container.side_effect = capture_create

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        await executor.run_job(job_def)

    assert len(captured_envs) == 1
    env = captured_envs[0]

    # Workspace paths.
    assert env["JOB_ID"] == "test-job-001"
    assert env["WORKSPACE_MODELS"] == "/workspace/models"
    assert env["WORKSPACE_DATA"] == "/workspace/data"
    assert env["WORKSPACE_OUTPUT"] == "/workspace/output"
    assert env["WORKSPACE_CONFIG"] == "/workspace/config"
    assert env["JOB_CONFIG"] == "/workspace/config/job.json"
    assert env["STEP_NAME"] == "train"
    assert env["STEP_INDEX"] == "0"

    # Infrastructure credentials from settings.
    assert env["HARBOR_URL"] == "http://harbor.example.com"
    assert env["HARBOR_USERNAME"] == "user"
    assert env["HARBOR_PASSWORD"] == "pass"
    assert env["HF_TOKEN"] == "hf-token-xyz"
    assert env["HF_HOME"] == "/workspace/.cache/huggingface"

    # Step-specific override.
    assert env["CUSTOM_VAR"] == "custom_val"


# ---------------------------------------------------------------------------
# is_preparation_step forwarded to create_container
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_preparation_step_forwarded() -> None:
    executor, ds, store = _make_executor()
    steps = [
        _make_step("download", is_preparation_step=True),
        _make_step("train", is_preparation_step=False),
    ]
    job_def = _make_job(steps)

    ds.create_container.side_effect = ["container-1", "container-2"]
    recorded_flags: list[bool] = []

    original_create = ds.create_container.side_effect

    def _capture(image, job_id, step_name, environment, gpu=False, is_preparation_step=False):  # type: ignore[misc]
        recorded_flags.append(is_preparation_step)
        return ["container-1", "container-2"][len(recorded_flags) - 1]

    ds.create_container.side_effect = _capture

    workspace = Path("/tmp/solar-test-jobs/test-job-001")
    with (
        patch("solar_host.jobs.executor.validate_job_id"),
        patch("solar_host.jobs.executor.check_disk_space"),
        patch("solar_host.jobs.executor.create_workspace", return_value=workspace),
        patch("solar_host.jobs.executor.JobExecutor._stream_logs_to_file"),
    ):
        result = await executor.run_job(job_def)

    assert result.status == JobStatus.completed
    assert recorded_flags == [True, False]
