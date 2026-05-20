"""DockerService: lifecycle primitives wrapping docker-py."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import docker
import docker.errors as _docker_errors

from solar_host.config import Settings, settings as _default_settings
from solar_host.docker.errors import (
    ContainerNonZeroExitError,
    ContainerStartError,
    DaemonUnavailableError,
    ImagePullError,
)

logger = logging.getLogger(__name__)

# Mounts whose read/write mode depends on is_preparation_step.
_TOGGLED_MOUNTS: list[tuple[str, str]] = [
    ("models", "/workspace/models"),
    ("data", "/workspace/data"),
]

# Mounts that are always read-write regardless of step type.
_ALWAYS_RW_MOUNTS: list[tuple[str, str]] = [
    ("output", "/workspace/output"),
    ("config", "/workspace/config"),
]


@dataclass
class ContainerStatus:
    container_id: str
    status: str  # "created", "running", "exited", etc.
    exit_code: int | None
    started_at: str | None
    finished_at: str | None


class DockerService:
    """Thin synchronous wrapper around docker-py.

    Intended to be called via ``asyncio.to_thread`` from the async step
    executor (S-023) so that blocking Docker I/O never stalls the event loop.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or _default_settings
        try:
            self._client: docker.DockerClient = docker.from_env()
        except _docker_errors.DockerException as exc:
            raise DaemonUnavailableError(
                f"Cannot connect to Docker daemon: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def pull_image(self, image: str, tag: str = "latest") -> None:
        """Pull *image:tag* from a registry."""
        image_ref = f"{image}:{tag}" if ":" not in image else image
        logger.info("Pulling image %s", image_ref)
        try:
            self._client.images.pull(image, tag=tag if ":" not in image else None)
        except _docker_errors.APIError as exc:
            raise ImagePullError(image_ref, str(exc)) from exc
        except _docker_errors.DockerException as exc:
            raise ImagePullError(image_ref, str(exc)) from exc

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def create_container(
        self,
        image: str,
        job_id: str,
        step_name: str,
        environment: dict[str, str],
        gpu: bool = False,
        is_preparation_step: bool = False,
    ) -> str:
        """Create (but don't start) a container and return its ID.

        Args:
            is_preparation_step: When True, ``models/`` and ``data/`` are
                mounted read-write (the step is allowed to populate them).
                When False (default), those directories are read-only.
                ``output/`` and ``config/`` are always read-write.
        """
        s = self._settings
        job_path = Path(s.jobs_dir) / job_id

        volumes: dict[str, dict[str, str]] = {}

        toggled_mode = "rw" if is_preparation_step else "ro"
        for sub_dir, container_path in _TOGGLED_MOUNTS:
            host_path = str((job_path / sub_dir).resolve())
            volumes[host_path] = {"bind": container_path, "mode": toggled_mode}

        for sub_dir, container_path in _ALWAYS_RW_MOUNTS:
            host_path = str((job_path / sub_dir).resolve())
            volumes[host_path] = {"bind": container_path, "mode": "rw"}

        hf_cache_host = str(Path(s.hf_cache_dir).resolve())
        volumes[hf_cache_host] = {
            "bind": "/workspace/.cache/huggingface",
            "mode": "rw",
        }

        device_requests = None
        if gpu:
            device_requests = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]

        container_name = f"solar-job-{job_id}-{step_name}"
        logger.info(
            "Creating container %s from image %s (gpu=%s)", container_name, image, gpu
        )
        try:
            container = self._client.containers.create(
                image,
                name=container_name,
                environment=environment,
                volumes=volumes,
                user=f"{s.container_uid}:{s.container_gid}",
                network_mode="bridge",
                privileged=False,
                device_requests=device_requests,
            )
        except _docker_errors.APIError as exc:
            raise ContainerStartError(container_name, str(exc)) from exc

        return container.id  # type: ignore[return-value]

    def start_container(self, container_id: str) -> None:
        """Start a previously created container."""
        logger.info("Starting container %s", container_id)
        try:
            container = self._client.containers.get(container_id)
            container.start()
        except _docker_errors.NotFound as exc:
            raise ContainerStartError(container_id, f"Not found: {exc}") from exc
        except _docker_errors.APIError as exc:
            raise ContainerStartError(container_id, str(exc)) from exc

    def stop_container(self, container_id: str, timeout: int = 30) -> None:
        """Send SIGTERM to the container and wait up to *timeout* seconds."""
        logger.info("Stopping container %s (timeout=%ds)", container_id, timeout)
        try:
            container = self._client.containers.get(container_id)
            container.stop(timeout=timeout)
        except _docker_errors.NotFound:
            logger.warning("Container %s not found during stop; ignoring", container_id)
        except _docker_errors.APIError as exc:
            logger.warning("Error stopping container %s: %s", container_id, exc)

    def remove_container(self, container_id: str, force: bool = False) -> None:
        """Remove a container, optionally with ``force=True``."""
        logger.info("Removing container %s (force=%s)", container_id, force)
        try:
            container = self._client.containers.get(container_id)
            container.remove(force=force)
        except _docker_errors.NotFound:
            logger.warning(
                "Container %s not found during remove; ignoring", container_id
            )
        except _docker_errors.APIError as exc:
            logger.warning("Error removing container %s: %s", container_id, exc)

    def inspect_container(self, container_id: str) -> ContainerStatus:
        """Return a ``ContainerStatus`` snapshot for *container_id*."""
        try:
            container = self._client.containers.get(container_id)
            container.reload()
            state = container.attrs.get("State", {})
            raw_exit = state.get("ExitCode")
            exit_code = int(raw_exit) if raw_exit is not None else None
            return ContainerStatus(
                container_id=container.id,  # type: ignore[arg-type]
                status=container.status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=state.get("StartedAt"),
                finished_at=state.get("FinishedAt"),
            )
        except _docker_errors.NotFound as exc:
            raise ContainerStartError(container_id, f"Not found: {exc}") from exc

    def stream_logs(
        self, container_id: str, follow: bool = True, tail: int = 50
    ) -> Iterator[str]:
        """Yield decoded log lines from the container.

        Args:
            container_id: ID of the target container.
            follow: Keep streaming until the container stops.
            tail: Number of lines to show from the end of existing logs
                before streaming new output. Pass ``0`` for all lines.
        """
        try:
            container = self._client.containers.get(container_id)
            log_stream = container.logs(
                stream=True,
                follow=follow,
                stdout=True,
                stderr=True,
                tail=tail if tail > 0 else "all",
            )
            for chunk in log_stream:
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="replace")
                else:
                    yield chunk  # type: ignore[misc]
        except _docker_errors.NotFound as exc:
            raise ContainerStartError(container_id, f"Not found: {exc}") from exc

    def wait_container(self, container_id: str) -> int:
        """Block until the container stops and return its exit code.

        Raises ``ContainerNonZeroExitError`` when the exit code is non-zero.
        """
        try:
            container = self._client.containers.get(container_id)
            result = container.wait()
            exit_code: int = result.get("StatusCode", -1)
        except _docker_errors.NotFound as exc:
            raise ContainerStartError(container_id, f"Not found: {exc}") from exc
        except _docker_errors.APIError as exc:
            raise ContainerStartError(container_id, str(exc)) from exc

        if exit_code != 0:
            stderr_lines: list[str] = []
            try:
                raw = container.logs(stdout=False, stderr=True, tail=20)
                stderr_lines = raw.decode("utf-8", errors="replace").splitlines()
            except Exception:
                pass
            raise ContainerNonZeroExitError(container_id, exit_code, stderr_lines)

        return exit_code
