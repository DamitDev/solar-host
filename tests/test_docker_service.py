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
    ImagePullError,
)
from solar_host.docker.service import ContainerStatus, DockerService

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

    # data mount must be read-only
    data_entry = next(v for v in volumes.values() if v["bind"] == "/workspace/data")
    assert data_entry["mode"] == "ro"


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
    svc.create_container("alpine", "job-1", "train", {}, gpu=False)
    kwargs = client.containers.create.call_args[1]
    assert kwargs["device_requests"] is None


def test_create_container_with_gpu():
    client = MagicMock()
    mock_container = MagicMock()
    mock_container.id = "abc123"
    client.containers.create.return_value = mock_container

    svc = _make_service(client)
    svc.create_container("alpine", "job-1", "train", {}, gpu=True)
    kwargs = client.containers.create.call_args[1]
    assert kwargs["device_requests"] is not None
    assert len(kwargs["device_requests"]) == 1


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
