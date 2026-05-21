"""Structured exception hierarchy for the jobs execution layer."""

from __future__ import annotations


class JobsError(Exception):
    """Base class for all jobs-layer errors."""


class WorkspaceError(JobsError):
    """Base class for workspace-related failures."""

    def __init__(self, job_id: str, reason: str) -> None:
        self.job_id = job_id
        self.reason = reason
        super().__init__(f"Workspace error for job {job_id!r}: {reason}")


class InvalidJobIdError(WorkspaceError):
    """Job ID contains forbidden characters or patterns."""

    def __init__(self, job_id: str, reason: str) -> None:
        super().__init__(job_id, reason)


class InsufficientDiskError(JobsError):
    """Available disk space is below the required minimum."""

    def __init__(self, required_gb: float, available_gb: float) -> None:
        self.required_gb = required_gb
        self.available_gb = available_gb
        super().__init__(
            f"Insufficient disk space: required {required_gb:.1f} GB, "
            f"available {available_gb:.1f} GB"
        )


class GpuValidationError(JobsError):
    """Requested GPU device ID or count exceeds the host inventory."""

    def __init__(self, requested: str, available_count: int) -> None:
        self.requested = requested
        self.available_count = available_count
        super().__init__(
            f"GPU validation failed: {requested!r} not available "
            f"(host has {available_count} device(s))"
        )
