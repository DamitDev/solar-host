"""Unit tests for solar_host.docker.DockerService.

All tests mock docker.from_env so no real Docker daemon is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import docker.errors as _docker_errors
import pytest

from solar_host.config import Settings
from solar_host.docker.errors import (
    ContainerNonZeroExitError,
    ContainerStartError,
    DaemonUnavailableError,
    GpuUnavailableError,
    ImagePullError,
)
from solar_host.docker.service import ContainerStatus, DockerService
from solar_host.jobs.models import GpuOptions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_SETTINGS = Settings(
    jobs_dir="/tmp/solar-test-jobs",
    container_uid=1000,
    container_gid=1000,
    hf_cache_dir="/tmp/solar-test-hf-cache",
)


def _make_service(mock_client: MagicMock) -> DockerService:
    """Build a DockerService whose internal client is replaced by *mock_client*."""
    with patch("solar_host.docker.service.docker.from_env", return_value=mock_client):
        return DockerService(settings=_TEST_SETTINGS)


# ---------------------------------------------------------------------------
# Daemon connectivity
# ---------------------------------------------------------------------------


def test_daemon_unavailable_raises():
    with patch(
        "solar_host.docker.service.docker.from_env",
        side_effect=_docker_errors.DockerException("daemon down"),
    ):
        with pytest.raises(DaemonUnavailableError):
            DockerService(settings=_TEST_SETTINGS)


# ---------------------------------------------------------------------------
# pull_image
# ---------------------------------------------------------------------------


def test_pull_image_success():
    client = MagicMock()
    svc = _make_service(client)
    svc.pull_image("alpine", "3.18")
    client.images.pull.assert_called_once_with("alpine", tag="3.18")


def test_pull_image_with_tag_in_name():
    """When the caller embeds the tag in the image name, tag kwarg should be None."""
    client = MagicMock()
    svc = _make_service(client)
    svc.pull_image("alpine:3.18")
    client.images.pull.assert_called_once_with("alpine:3.18", tag=None)


def test_pull_image_api_error_raises():
    client = MagicMock()
    client.images.pull.side_effect = _docker_errors.APIError("404 not found")
    svc = _make_service(client)
    with pytest.raises(ImagePullError) as exc_info:
        svc.pull_image("nonexistent", "latest")
    assert exc_info.value.image_ref == "nonexistent:latest"


# ---------------------------------------------------------------------------
# create_container
# ---------------------------------------------------------------------------


def test_create_container_returns_id():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    cid = svc.create_container(
        image="alpine:3.18",
        job_id="job-1",
        step_name="train",
        environment={"FOO": "bar"},
    )
    assert cid == "abc123"


def test_create_container_bind_mounts():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container(
        image="alpine:3.18",
        job_id="job-1",
        step_name="train",
        environment={},
    )

    kwargs = client.containers.create.call_args[1]
    volumes: dict = kwargs["volumes"]

    container_paths = {v["bind"] for v in volumes.values()}
    assert "/workspace/models" in container_paths
    assert "/workspace/data" in container_paths
    assert "/workspace/output" in container_paths
    assert "/workspace/config" in container_paths
    assert "/workspace/.cache/huggingface" in container_paths

    # default (consumption) step: models and data must be read-only
    models_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/models")
    data_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/data")
    assert models_entry["mode"] == "ro"
    assert data_entry["mode"] == "ro"


def _volumes_for(client: MagicMock, svc: DockerService, **kwargs: object) -> dict:
    """Helper: call create_container and return the volumes kwarg."""
    svc.create_container(
        image="alpine:3.18",
        job_id="job-1",
        step_name="train",
        environment={},
        **kwargs,  # type: ignore[arg-type]
    )
    return client.containers.create.call_args[1]["volumes"]


def test_create_container_preparation_step_models_and_data_rw():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    volumes = _volumes_for(client, svc, is_preparation_step=True)

    models_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/models")
    data_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/data")
    assert models_entry["mode"] == "rw"
    assert data_entry["mode"] == "rw"


def test_create_container_consumption_step_models_and_data_ro():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    volumes = _volumes_for(client, svc, is_preparation_step=False)

    models_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/models")
    data_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/data")
    assert models_entry["mode"] == "ro"
    assert data_entry["mode"] == "ro"


def test_create_container_output_and_config_always_rw():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    for prep_flag in (True, False):
        client.containers.create.reset_mock()
        client.containers.create.return_value = mock_container
        volumes = _volumes_for(client, svc, is_preparation_step=prep_flag)

        output_entry = next(
            v for v in volumes.values() if v["bind"] == "/workspace/output"
        )
        config_entry = next(
            v for v in volumes.values() if v["bind"] == "/workspace/config"
        )
        assert (
            output_entry["mode"] == "rw"
        ), f"output not rw when is_preparation_step={prep_flag}"
        assert (
            config_entry["mode"] == "rw"
        ), f"config not rw when is_preparation_step={prep_flag}"


def test_create_container_user_set():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {})
    kwargs = client.containers.create.call_args[1]
    assert kwargs["user"] == "1000:1000"


def test_create_container_no_gpu():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {}, gpu=None)
    kwargs = client.containers.create.call_args[1]
    assert kwargs["device_requests"] is None


def test_create_container_name_format():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container("alpine", "my-job", "my-step", {})
    kwargs = client.containers.create.call_args[1]
    assert kwargs["name"] == "solar-job-my-job-my-step"


def test_create_container_not_privileged():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {})
    kwargs = client.containers.create.call_args[1]
    assert kwargs.get("privileged") is False


def test_create_container_api_error():
    client = MagicMock()
    client.containers.create.side_effect = _docker_errors.APIError("image not found")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError):
        svc.create_container("bad-image", "job-1", "train", {})


# ---------------------------------------------------------------------------
# start_container
# ---------------------------------------------------------------------------


def test_start_container_calls_start():
    client = MagicMock()
    mock_container = MagicMock()
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    svc.start_container("abc123")
    mock_container.start.assert_called_once()


def test_start_container_not_found_raises():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError):
        svc.start_container("ghost")


# ---------------------------------------------------------------------------
# stop_container
# ---------------------------------------------------------------------------


def test_stop_container_calls_stop_with_timeout():
    client = MagicMock()
    mock_container = MagicMock()
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    svc.stop_container("abc123", timeout=10)
    mock_container.stop.assert_called_once_with(timeout=10)


def test_stop_container_not_found_does_not_raise():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    svc.stop_container("ghost")  # should not raise


# ---------------------------------------------------------------------------
# remove_container
# ---------------------------------------------------------------------------


def test_remove_container_calls_remove():
    client = MagicMock()
    mock_container = MagicMock()
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    svc.remove_container("abc123", force=True)
    mock_container.remove.assert_called_once_with(force=True)


def test_remove_container_not_found_does_not_raise():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    svc.remove_container("ghost")  # should not raise


# ---------------------------------------------------------------------------
# inspect_container
# ---------------------------------------------------------------------------


def _mock_container_with_state(
    container_id: str = "abc123",
    status: str = "running",
    exit_code: int | None = None,
    started_at: str | None = "2024-01-01T00:00:00Z",
    finished_at: str | None = None,
) -> MagicMock:
    container = MagicMock()
    container.id = container_id
    container.status = status
    container.attrs = {
        "State": {
            "ExitCode": exit_code,
            "StartedAt": started_at,
            "FinishedAt": finished_at,
        }
    }
    return container


def test_inspect_running_container():
    client = MagicMock()
    mock_container = _mock_container_with_state(status="running", exit_code=0)
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    status = svc.inspect_container("abc123")
    assert isinstance(status, ContainerStatus)
    assert status.status == "running"
    assert status.exit_code == 0


def test_inspect_exited_container():
    client = MagicMock()
    mock_container = _mock_container_with_state(
        status="exited", exit_code=1, finished_at="2024-01-01T01:00:00Z"
    )
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    status = svc.inspect_container("abc123")
    assert status.status == "exited"
    assert status.exit_code == 1


def test_inspect_not_found_raises():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError):
        svc.inspect_container("ghost")


# ---------------------------------------------------------------------------
# stream_logs
# ---------------------------------------------------------------------------


def test_stream_logs_yields_decoded_lines():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([b"hello\n", b"world\n"])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    lines = list(svc.stream_logs("abc123"))
    assert lines == ["hello\n", "world\n"]


def test_stream_logs_default_tail_passed_to_docker():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    list(svc.stream_logs("abc123"))
    kwargs = mock_container.logs.call_args[1]
    assert kwargs["tail"] == 50


def test_stream_logs_custom_tail():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    list(svc.stream_logs("abc123", tail=10))
    kwargs = mock_container.logs.call_args[1]
    assert kwargs["tail"] == 10


def test_stream_logs_tail_zero_means_all():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    list(svc.stream_logs("abc123", tail=0))
    kwargs = mock_container.logs.call_args[1]
    assert kwargs["tail"] == "all"


def test_stream_logs_not_found_raises():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError):
        list(svc.stream_logs("ghost"))


# ---------------------------------------------------------------------------
# wait_container
# ---------------------------------------------------------------------------


def test_wait_container_success_returns_zero():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    code = svc.wait_container("abc123")
    assert code == 0


def test_wait_container_nonzero_raises():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 2}
    mock_container.logs.return_value = b"something failed"
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    with pytest.raises(ContainerNonZeroExitError) as exc_info:
        svc.wait_container("abc123")
    assert exc_info.value.exit_code == 2
    assert exc_info.value.container_id == "abc123"


def test_wait_container_not_found_raises():
    client = MagicMock()
    client.containers.get.side_effect = _docker_errors.NotFound("not found")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError):
        svc.wait_container("ghost")


# ---------------------------------------------------------------------------
# is_nvidia_toolkit_available
# ---------------------------------------------------------------------------


def test_nvidia_toolkit_available_when_nvidia_in_runtimes():
    client = MagicMock()
    client.info.return_value = {"Runtimes": {"nvidia": {}, "runc": {}}}
    svc = _make_service(client)
    assert svc.is_nvidia_toolkit_available() is True


def test_nvidia_toolkit_unavailable_when_nvidia_not_in_runtimes():
    client = MagicMock()
    client.info.return_value = {"Runtimes": {"runc": {}}}
    svc = _make_service(client)
    assert svc.is_nvidia_toolkit_available() is False


def test_nvidia_toolkit_unavailable_when_runtimes_missing():
    client = MagicMock()
    client.info.return_value = {}
    svc = _make_service(client)
    assert svc.is_nvidia_toolkit_available() is False


def test_nvidia_toolkit_unavailable_on_exception():
    client = MagicMock()
    client.info.side_effect = Exception("daemon error")
    svc = _make_service(client)
    assert svc.is_nvidia_toolkit_available() is False


# ---------------------------------------------------------------------------
# create_container — GPU wiring
# ---------------------------------------------------------------------------


def _make_container(client: MagicMock, cid: str = "gpu-ctr") -> MagicMock:
    mock = MagicMock()
    mock.id = cid
    client.containers.create.return_value = mock
    return mock


def test_create_container_gpu_none_no_device_requests():
    client = MagicMock()
    _make_container(client)
    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {}, gpu=None)
    kwargs = client.containers.create.call_args[1]
    assert kwargs["device_requests"] is None


def test_create_container_gpu_count_builds_device_request():
    client = MagicMock()
    _make_container(client)
    client.info.return_value = {"Runtimes": {"nvidia": {}}}
    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {}, gpu=GpuOptions(count=2))
    kwargs = client.containers.create.call_args[1]
    reqs = kwargs["device_requests"]
    assert reqs is not None and len(reqs) == 1
    assert reqs[0].count == 2
    assert not reqs[0].device_ids  # Docker SDK normalises absent device_ids to []


def test_create_container_gpu_count_minus_one():
    client = MagicMock()
    _make_container(client)
    client.info.return_value = {"Runtimes": {"nvidia": {}}}
    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {}, gpu=GpuOptions(count=-1))
    kwargs = client.containers.create.call_args[1]
    reqs = kwargs["device_requests"]
    assert reqs is not None
    assert reqs[0].count == -1


def test_create_container_gpu_device_ids_builds_device_request():
    client = MagicMock()
    _make_container(client)
    client.info.return_value = {"Runtimes": {"nvidia": {}}}
    svc = _make_service(client)
    svc.create_container(
        "alpine", "job-1", "train", {}, gpu=GpuOptions(device_ids=["0", "1"])
    )
    kwargs = client.containers.create.call_args[1]
    reqs = kwargs["device_requests"]
    assert reqs is not None and len(reqs) == 1
    assert reqs[0].device_ids == ["0", "1"]
    assert reqs[0].count == 0  # Docker SDK normalises absent count to 0


def test_create_container_gpu_raises_when_toolkit_missing():
    client = MagicMock()
    _make_container(client)
    client.info.return_value = {"Runtimes": {"runc": {}}}
    svc = _make_service(client)
    with pytest.raises(GpuUnavailableError):
        svc.create_container("alpine", "job-1", "train", {}, gpu=GpuOptions(count=1))


# ---------------------------------------------------------------------------
# stream_logs — demux
# ---------------------------------------------------------------------------


def test_stream_logs_demux_yields_stream_tuples():
    """demux=True yields (stream, chunk) tuples for stdout and stderr."""
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter(
        [
            (b"stdout line\n", None),
            (None, b"stderr line\n"),
            (b"more stdout\n", b"more stderr\n"),
        ]
    )
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    results = list(svc.stream_logs("abc123", follow=True, tail=0, demux=True))

    assert ("stdout", "stdout line\n") in results
    assert ("stderr", "stderr line\n") in results
    assert ("stdout", "more stdout\n") in results
    assert ("stderr", "more stderr\n") in results


def test_stream_logs_demux_skips_none_chunks():
    """When one of the demux tuple elements is None it is not yielded."""
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter(
        [
            (b"only stdout\n", None),
            (None, None),  # both None — nothing should be yielded
        ]
    )
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    results = list(svc.stream_logs("abc123", follow=True, tail=0, demux=True))
    assert results == [("stdout", "only stdout\n")]


def test_stream_logs_demux_false_preserves_plain_str_output():
    """demux=False (default) still yields plain strings — backward-compat."""
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([b"plain\n"])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    results = list(svc.stream_logs("abc123"))
    assert results == ["plain\n"]


def test_stream_logs_demux_passes_demux_param_to_docker():
    """demux=True passes demux=True to docker-py container.logs()."""
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = iter([])
    client.containers.get.return_value = mock_container

    svc = _make_service(client)
    list(svc.stream_logs("abc123", follow=True, tail=0, demux=True))

    kwargs = mock_container.logs.call_args[1]
    assert kwargs.get("demux") is True


# ---------------------------------------------------------------------------
# Auto-pull on ImageNotFound
# ---------------------------------------------------------------------------


def test_create_container_auto_pulls_when_image_not_found():
    """When the image is not found locally, pull_image is attempted and the
    container is re-created.  On success the container ID is returned."""
    client = MagicMock()
    client.info.return_value = {}

    # First create → ImageNotFound (triggers pull + retry).
    # Second create → succeeds.
    mock_c1 = MagicMock()
    mock_c1.id = "pulled-container-id"
    client.containers.create.side_effect = [
        _docker_errors.ImageNotFound("not found"),
        mock_c1,
    ]

    svc = _make_service(client)
    cid = svc.create_container(
        image="missing/image:latest",
        job_id="job-1",
        step_name="train",
        environment={},
    )
    assert cid == "pulled-container-id"
    client.images.pull.assert_called_once_with("missing/image:latest", tag=None)
    assert client.containers.create.call_count == 2


def test_create_container_auto_pull_failure_raises_container_start_error():
    """If both the initial create AND the pull fail, ContainerStartError is raised."""
    client = MagicMock()
    client.info.return_value = {}
    client.containers.create.side_effect = _docker_errors.ImageNotFound("not found")
    client.images.pull.side_effect = _docker_errors.APIError("registry down")

    svc = _make_service(client)
    with pytest.raises(ContainerStartError) as exc_info:
        svc.create_container(
            image="bad/image:latest",
            job_id="job-1",
            step_name="train",
            environment={},
        )
    assert "pull failed" in str(exc_info.value).lower()
