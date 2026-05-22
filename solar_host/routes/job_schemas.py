"""Pydantic request / response schemas for the /jobs REST endpoints (S-027)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from solar_host.config import settings
from solar_host.jobs.models import JobDefinition, JobState, StepLogMessage, StepState
from solar_host.jobs.step_log_buffer import step_log_buffer

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

# JobSubmitRequest is the full JobDefinition shape (submission_id / correlation_id
# are already present on JobDefinition after US-002).
JobSubmitRequest = JobDefinition

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    workspace_path: str
    submission_id: str | None = None
    correlation_id: str | None = None


class StepStateResponse(BaseModel):
    """StepState enriched with the host log-file path and a recent-log snippet."""

    name: str
    status: str
    container_id: str | None = None
    started_at: Any = None
    finished_at: Any = None
    duration_s: float | None = None
    exit_code: int | None = None
    error_message: str | None = None
    log_file: str
    recent_logs: list[StepLogMessage] = []

    @classmethod
    def from_step_state(
        cls,
        step: StepState,
        job_id: str,
        *,
        max_recent: int = 100,
    ) -> "StepStateResponse":
        from pathlib import Path

        log_file = str(Path(settings.jobs_dir) / job_id / "logs" / f"{step.name}.log")
        recent = step_log_buffer.get_buffer(job_id, step.name)
        return cls(
            name=step.name,
            status=step.status.value,
            container_id=step.container_id,
            started_at=step.started_at,
            finished_at=step.finished_at,
            duration_s=step.duration_s,
            exit_code=step.exit_code,
            error_message=step.error_message,
            log_file=log_file,
            recent_logs=recent[-max_recent:],
        )


class JobStateResponse(BaseModel):
    """JobState enriched with per-step log paths and recent log snippets."""

    job_id: str
    name: str
    status: str
    steps: list[StepStateResponse] = []
    current_step_index: int
    workspace_path: str
    created_at: Any = None
    started_at: Any = None
    finished_at: Any = None
    retention_hours: float
    error_message: str | None = None
    submission_id: str | None = None
    correlation_id: str | None = None

    @classmethod
    def from_job_state(cls, job: JobState) -> "JobStateResponse":
        return cls(
            job_id=job.job_id,
            name=job.name,
            status=job.status.value,
            steps=[StepStateResponse.from_step_state(s, job.job_id) for s in job.steps],
            current_step_index=job.current_step_index,
            workspace_path=job.workspace_path,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            retention_hours=job.retention_hours,
            error_message=job.error_message,
            submission_id=job.submission_id,
            correlation_id=job.correlation_id,
        )
