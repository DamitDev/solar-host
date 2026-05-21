"""Structured exception hierarchy for the Docker execution layer."""

from __future__ import annotations


class DockerServiceError(Exception):
    """Base class for all Docker layer errors."""


class DaemonUnavailableError(DockerServiceError):
    """Docker daemon is not reachable."""


class ImagePullError(DockerServiceError):
    """Failed to pull a Docker image."""

    def __init__(self, image_ref: str, reason: str) -> None:
        self.image_ref = image_ref
        self.reason = reason
        super().__init__(f"Failed to pull image {image_ref!r}: {reason}")


class ContainerStartError(DockerServiceError):
    """Container failed to start."""

    def __init__(self, container_id: str, reason: str) -> None:
        self.container_id = container_id
        self.reason = reason
        super().__init__(f"Container {container_id!r} failed to start: {reason}")


class ContainerNonZeroExitError(DockerServiceError):
    """Container exited with a non-zero exit code."""

    def __init__(
        self,
        container_id: str,
        exit_code: int,
        last_stderr_lines: list[str] | None = None,
    ) -> None:
        self.container_id = container_id
        self.exit_code = exit_code
        self.last_stderr_lines: list[str] = last_stderr_lines or []
        tail = "\n".join(self.last_stderr_lines[-10:])
        super().__init__(
            f"Container {container_id!r} exited with code {exit_code}.\n{tail}".strip()
        )


class GpuUnavailableError(DockerServiceError):
    """NVIDIA Container Toolkit is not available on this host."""
