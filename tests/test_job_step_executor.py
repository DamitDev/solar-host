"""Unit tests for solar_host.jobs.step_executor.JobStepExecutor.

Tests focus on single-step behaviour: container lifecycle, log streaming,
environment construction, and is_preparation_step forwarding.
All Docker calls are mocked — no real Docker daemon or filesystem writes needed.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from solar_host.config import Settings
from solar_host.docker.errors import (
    ContainerNonZeroExitError,
    ContainerStartError,
    GpuUnavailableError,
)
from solar_host.jobs.errors import GpuValidationError
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

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
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

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
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

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
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

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
        await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    ds.remove_container.assert_called_once_with("container-xyz", True)


# ---------------------------------------------------------------------------
# active_containers updated and cleared
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_active_container_cleared_after_step() -> None:
    active: dict[str, str] = {}
    step_exec, ds, store = _make_step_executor(active_containers=active)

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
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

    def _capture(
        image, job_id, step_name, environment, gpu=None, is_preparation_step=False
    ):  # type: ignore[misc]
        recorded.append(is_preparation_step)
        return "container-xyz"

    ds.create_container.side_effect = _capture

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
        await step_exec.run(_JOB_ID, 0, steps[0], _WORKSPACE)

    assert recorded == [True]


@pytest.mark.anyio
async def test_consumption_step_flag_forwarded() -> None:
    steps = [_make_step(name="train", is_preparation_step=False)]
    store = _make_store_with_job(steps)
    step_exec, ds, _ = _make_step_executor(store=store)

    recorded: list[bool] = []

    def _capture(
        image, job_id, step_name, environment, gpu=None, is_preparation_step=False
    ):  # type: ignore[misc]
        recorded.append(is_preparation_step)
        return "container-xyz"

    ds.create_container.side_effect = _capture

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
        await step_exec.run(_JOB_ID, 0, steps[0], _WORKSPACE)

    assert recorded == [False]


# ---------------------------------------------------------------------------
# _validate_gpu_options
# ---------------------------------------------------------------------------

_TWO_GPU_INVENTORY = [
    {
        "index": 0,
        "uuid": "GPU-uuid-0",
        "name": "RTX 4090",
        "total_gb": 24.0,
        "used_gb": 0.0,
    },
    {
        "index": 1,
        "uuid": "GPU-uuid-1",
        "name": "RTX 4090",
        "total_gb": 24.0,
        "used_gb": 0.0,
    },
]


def test_validate_gpu_options_valid_count() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        step_exec._validate_gpu_options(GpuOptions(count=2))  # no exception


def test_validate_gpu_options_count_exceeds_inventory_raises() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        with pytest.raises(GpuValidationError):
            step_exec._validate_gpu_options(GpuOptions(count=5))


def test_validate_gpu_options_count_minus_one_skips_count_check() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        step_exec._validate_gpu_options(GpuOptions(count=-1))  # no exception


def test_validate_gpu_options_valid_device_index() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        step_exec._validate_gpu_options(
            GpuOptions(device_ids=["0", "1"])
        )  # no exception


def test_validate_gpu_options_valid_device_uuid() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        step_exec._validate_gpu_options(
            GpuOptions(device_ids=["GPU-uuid-0"])
        )  # no exception


def test_validate_gpu_options_invalid_device_id_raises() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=_TWO_GPU_INVENTORY):
        with pytest.raises(GpuValidationError):
            step_exec._validate_gpu_options(GpuOptions(device_ids=["99"]))


def test_validate_gpu_options_empty_inventory_skips() -> None:
    step_exec, _, _ = _make_step_executor()
    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=[]):
        step_exec._validate_gpu_options(
            GpuOptions(count=99)
        )  # no exception — no inventory


# ---------------------------------------------------------------------------
# GpuUnavailableError propagates as step failure
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_gpu_unavailable_error_marks_step_failed() -> None:
    step = _make_step(gpu=GpuOptions(count=1))
    store = _make_store_with_job([step])
    step_exec, ds, _ = _make_step_executor(store=store)
    ds.create_container.side_effect = GpuUnavailableError("toolkit missing")

    with patch(f"{_STEP_MODULE}.get_gpu_devices", return_value=[]):
        with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
            result = await step_exec.run(_JOB_ID, 0, step, _WORKSPACE)

    assert result is True
    assert store.get(_JOB_ID).steps[0].status == StepStatus.failed  # type: ignore[union-attr]
    assert "toolkit missing" in (store.get(_JOB_ID).steps[0].error_message or "")  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# build_environment — NVIDIA env vars
# ---------------------------------------------------------------------------


def test_build_environment_nvidia_vars_when_gpu_set() -> None:
    step_exec, _, _ = _make_step_executor()
    step = _make_step(gpu=GpuOptions(count=1))
    env = step_exec.build_environment(_JOB_ID, 0, step, _WORKSPACE)
    assert env["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"


def test_build_environment_no_nvidia_vars_when_gpu_none() -> None:
    step_exec, _, _ = _make_step_executor()
    step = _make_step(gpu=None)
    env = step_exec.build_environment(_JOB_ID, 0, step, _WORKSPACE)
    assert "NVIDIA_VISIBLE_DEVICES" not in env
    assert "NVIDIA_DRIVER_CAPABILITIES" not in env


def test_build_environment_step_env_overrides_nvidia_vars() -> None:
    step_exec, _, _ = _make_step_executor()
    step = _make_step(
        gpu=GpuOptions(count=1),
        environment={
            "NVIDIA_VISIBLE_DEVICES": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
        },
    )
    env = step_exec.build_environment(_JOB_ID, 0, step, _WORKSPACE)
    assert env["NVIDIA_VISIBLE_DEVICES"] == "0"
    assert env["NVIDIA_DRIVER_CAPABILITIES"] == "all"


# ---------------------------------------------------------------------------
# _stream_logs — buffer integration
# ---------------------------------------------------------------------------

_BUF_MODULE = "solar_host.jobs.step_executor.step_log_buffer"


def test_stream_logs_appends_to_buffer(tmp_path: Path) -> None:
    """_stream_logs dual-writes to file and calls step_log_buffer.append per chunk."""
    step_exec, ds, _ = _make_step_executor()
    ds.stream_logs.return_value = iter([("stdout", "hello\n"), ("stderr", "err\n")])

    log_path = tmp_path / "test.log"
    mock_buf = MagicMock()

    with patch(_BUF_MODULE, mock_buf):
        step_exec._stream_logs(_JOB_ID, 0, "train", "ctr-xyz", log_path)

    assert mock_buf.append.call_count == 2
    mock_buf.append.assert_any_call(_JOB_ID, "train", 0, "stdout", "hello\n")
    mock_buf.append.assert_any_call(_JOB_ID, "train", 0, "stderr", "err\n")

    content = log_path.read_text()
    assert "hello" in content
    assert "err" in content


def test_stream_logs_writes_file_without_double_newline(tmp_path: Path) -> None:
    """Lines that already end with \\n are not double-newlined in the log file."""
    step_exec, ds, _ = _make_step_executor()
    ds.stream_logs.return_value = iter([("stdout", "already\n")])

    log_path = tmp_path / "test.log"
    with patch(_BUF_MODULE):
        step_exec._stream_logs(_JOB_ID, 0, "train", "ctr-xyz", log_path)

    assert log_path.read_text() == "already\n"


def test_stream_logs_failure_does_not_raise(tmp_path: Path) -> None:
    """A streaming exception is caught; the method returns without raising."""
    step_exec, ds, _ = _make_step_executor()
    ds.stream_logs.side_effect = RuntimeError("docker broken")

    log_path = tmp_path / "train.log"
    with patch(_BUF_MODULE):
        step_exec._stream_logs(_JOB_ID, 0, "train", "ctr-xyz", log_path)  # no raise


# ---------------------------------------------------------------------------
# mark_completed called on success and non-zero exit
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_mark_completed_called_on_success() -> None:
    step_exec, ds, _ = _make_step_executor()
    mock_buf = MagicMock()

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
        with patch(_BUF_MODULE, mock_buf):
            await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    mock_buf.mark_completed.assert_called_once_with(_JOB_ID, "train", 0, exit_code=0)


@pytest.mark.anyio
async def test_mark_completed_called_on_nonzero_exit() -> None:
    step_exec, ds, _ = _make_step_executor()
    ds.wait_container.side_effect = ContainerNonZeroExitError("ctr", 5, [])
    mock_buf = MagicMock()

    with patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"):
        with patch(_BUF_MODULE, mock_buf):
            await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    mock_buf.mark_completed.assert_called_once_with(_JOB_ID, "train", 0, exit_code=5)


# ---------------------------------------------------------------------------
# Lifecycle event emission from step executor
# ---------------------------------------------------------------------------

_BROADCAST = "solar_host.jobs.events.broadcast_job_lifecycle"


@pytest.mark.anyio
async def test_step_started_emitted_on_run() -> None:
    step_exec, _, _ = _make_step_executor()

    with (
        patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"),
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "step_started" in calls


@pytest.mark.anyio
async def test_step_completed_emitted_on_success() -> None:
    step_exec, _, _ = _make_step_executor()

    with (
        patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"),
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is False
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "step_completed" in calls
    assert "step_failed" not in calls

    completed_payload = next(
        call.args[1]
        for call in mock_bc.call_args_list
        if call.args[0] == "step_completed"
    )
    assert completed_payload["exit_code"] == 0
    assert completed_payload["step_name"] == "train"


@pytest.mark.anyio
async def test_step_failed_emitted_on_nonzero_exit() -> None:
    step_exec, ds, _ = _make_step_executor()
    ds.wait_container.side_effect = ContainerNonZeroExitError("ctr", 2, ["FAIL"])

    with (
        patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"),
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is True
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "step_failed" in calls
    assert "step_completed" not in calls

    failed_payload = next(
        call.args[1] for call in mock_bc.call_args_list if call.args[0] == "step_failed"
    )
    assert failed_payload["exit_code"] == 2
    assert "FAIL" in (failed_payload["error_summary"] or "")


@pytest.mark.anyio
async def test_step_failed_emitted_on_container_start_error() -> None:
    step_exec, ds, _ = _make_step_executor()
    ds.create_container.side_effect = ContainerStartError("ctr", "API error")

    with (
        patch(f"{_STEP_MODULE}.JobStepExecutor._stream_logs"),
        patch(_BROADCAST, new_callable=AsyncMock) as mock_bc,
    ):
        result = await step_exec.run(_JOB_ID, 0, _make_step(), _WORKSPACE)

    assert result is True
    calls = [call.args[0] for call in mock_bc.call_args_list]
    assert "step_failed" in calls
