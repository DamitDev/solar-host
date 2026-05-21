"""Unit tests for solar_host.jobs.step_executor.JobStepExecutor.

Tests focus on single-step behaviour: container lifecycle, log streaming,
environment construction, and is_preparation_step forwarding.
All Docker calls are mocked — no real Docker daemon or filesystem writes needed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from solar_host.config import Settings
from solar_host.docker.errors import ContainerNonZeroExitError, ContainerStartError
from solar_host.jobs.models import (
    GpuOptions,
    JobState,
    JobStatus,
    StepDefinition,
    StepState,
    StepStatus,
)
from solar_host.jobs.step_executor import JobStepExecutor
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

_JOB_ID = "step-test-job"
_WORKSPACE = Path("/tmp/solar-test-jobs/step-test-job")

_STEP_MODULE = "solar_host.jobs.step_executor"


def _make_step(
    name: str = "train",
    image: str = "test/img:latest",
    gpu: GpuOptions | None = None,
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


def _make_store_with_job(steps: list[StepDefinition]) -> JobStore:
    """Return a JobStore pre-populated with a running job."""
    store = JobStore()
    store.add(
        JobState(
            job_id=_JOB_ID,
            name="Test Job",
            status=JobStatus.running,
            steps=[StepState(name=s.name) for s in steps],
        )
    )
    return store


def _make_step_executor(
    docker_service: MagicMock | None = None,
    store: JobStore | None = None,
    active_containers: dict[str, str] | None = None,
) -> tuple[JobStepExecutor, MagicMock, JobStore]:
    ds = docker_service or MagicMock()
    ds.create_container.return_value = "container-xyz"
    ds.start_container.return_value = None
    ds.wait_container.return_value = 0
    ds.remove_container.return_value = None
    ds.stream_logs.return_value = iter([])

    steps = [_make_step()]
    st = store or _make_store_with_job(steps)
    ac: dict[str, str] = active_containers if active_containers is not None else {}

    executor = JobStepExecutor(
        docker_service=ds,
        store=st,
        settings=_TEST_SETTINGS,
        active_containers=ac,
        lock=threading.Lock(),
    )
    return executor, ds, st


# ---------------------------------------------------------------------------
# Successful step
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_successful_step_returns_false() -> None:
    """A zero-exit step returns False (no fail-fast)."""
    step_exec, ds, store = _make_step_executor()

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is False
    assert store.get(_JOB_ID).steps[0].status == StepStatus.completed  # type: ignore[union-attr]
    assert store.get(_JOB_ID).steps[0].exit_code == 0  # type: ignore[union-attr]
    ds.create_container.assert_called_once()
    ds.start_container.assert_called_once()
    ds.remove_container.assert_called_once()


# ---------------------------------------------------------------------------
# Non-zero exit → fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_nonzero_exit_returns_true_and_marks_step_failed() -> None:
    step_exec, ds, store = _make_step_executor()
    ds.wait_container.side_effect = ContainerNonZeroExitError(
        "container-xyz", 2, ["error output"]
    )

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is True
    step = store.get(_JOB_ID).steps[0]  # type: ignore[union-attr]
    assert step.status == StepStatus.failed
    assert step.exit_code == 2
    assert "error output" in (step.error_message or "")
    assert store.get(_JOB_ID).status == JobStatus.failed  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# ContainerStartError → fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_container_start_error_returns_true_and_marks_step_failed() -> None:
    step_exec, ds, store = _make_step_executor()
    ds.create_container.side_effect = ContainerStartError("ctr", "API error")

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is True
    assert store.get(_JOB_ID).steps[0].status == StepStatus.failed  # type: ignore[union-attr]
    assert store.get(_JOB_ID).status == JobStatus.failed  # type: ignore[union-attr]
    # Container was never created so remove_container should not be called.
    ds.remove_container.assert_not_called()


# ---------------------------------------------------------------------------
# remove_container always called (even on failure)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_remove_container_called_even_on_nonzero_exit() -> None:
    step_exec, ds, store = _make_step_executor()
    ds.wait_container.side_effect = ContainerNonZeroExitError("container-xyz", 1, [])

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    ds.remove_container.assert_called_once_with("container-xyz", True)


# ---------------------------------------------------------------------------
# active_containers updated and cleared
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_active_container_cleared_after_step() -> None:
    active: dict[str, str] = {}
    step_exec, ds, store = _make_step_executor(active_containers=active)

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert _JOB_ID not in active


# ---------------------------------------------------------------------------
# Environment building
# ---------------------------------------------------------------------------


def test_build_environment_workspace_paths() -> None:
    step_exec, _, _ = _make_step_executor()
    step = _make_step(name="train", environment={"CUSTOM": "val"})
    env = step_exec.build_environment(_JOB_ID, 3, step, _WORKSPACE)

    assert env["JOB_ID"] == _JOB_ID
    assert env["WORKSPACE_MODELS"] == "/workspace/models"
    assert env["WORKSPACE_DATA"] == "/workspace/data"
    assert env["WORKSPACE_OUTPUT"] == "/workspace/output"
    assert env["WORKSPACE_CONFIG"] == "/workspace/config"
    assert env["JOB_CONFIG"] == "/workspace/config/job.json"
    assert env["STEP_NAME"] == "train"
    assert env["STEP_INDEX"] == "3"


def test_build_environment_credentials() -> None:
    step_exec, _, _ = _make_step_executor()
    env = step_exec.build_environment(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert env["HARBOR_URL"] == "http://harbor.example.com"
    assert env["HARBOR_USERNAME"] == "user"
    assert env["HARBOR_PASSWORD"] == "pass"
    assert env["HF_TOKEN"] == "hf-token-xyz"
    assert env["HF_HOME"] == "/workspace/.cache/huggingface"


def test_build_environment_step_vars_override_infra() -> None:
    step_exec, _, _ = _make_step_executor()
    step = _make_step(environment={"HF_TOKEN": "override-token", "CUSTOM_VAR": "x"})
    env = step_exec.build_environment(_JOB_ID, 0, step, _WORKSPACE)

    assert env["HF_TOKEN"] == "override-token"
    assert env["CUSTOM_VAR"] == "x"


# ---------------------------------------------------------------------------
# is_preparation_step forwarded to create_container
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_preparation_step_flag_forwarded() -> None:
    steps = [_make_step(name="download", is_preparation_step=True)]
    store = _make_store_with_job(steps)
    step_exec, ds, _ = _make_step_executor(store=store)

    recorded: list[bool] = []

    def _capture(image, job_id, step_name, environment, gpu=None, is_preparation_step=False):  # type: ignore[misc]
        recorded.append(is_preparation_step)
        return "container-xyz"

    ds.create_container.side_effect = _capture

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        await step_exec.run(_JOB_ID, 0, steps[0], _WORKSPACE)

    assert recorded == [True]


@pytest.mark.anyio
async def test_consumption_step_flag_forwarded() -> None:
    steps = [_make_step(name="train", is_preparation_step=False)]
    store = _make_store_with_job(steps)
    step_exec, ds, _ = _make_step_executor(store=store)

    recorded: list[bool] = []

    def _capture(image, job_id, step_name, environment, gpu=None, is_preparation_step=False):  # type: ignore[misc]
        recorded.append(is_preparation_step)
        return "container-xyz"

    ds.create_container.side_effect = _capture

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs_to_file"):
        await step_exec.run(_JOB_ID, 0, steps[0], _WORKSPACE)

    assert recorded == [False]
