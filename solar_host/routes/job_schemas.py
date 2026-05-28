"""Pydantic request / response schemas for the /jobs REST endpoints (S-027)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

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
    job_id: str = Field(examples=["job-abc123"])
    status: str = Field(examples=["running"])
    workspace_path: str = Field(examples=["/opt/projects/solar-host/jobs/job-abc123"])
    submission_id: str | None = Field(default=None, examples=["sub-42"])
    correlation_id: str | None = Field(default=None, examples=["corr-99"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job-abc123",
                    "status": "running",
                    "workspace_path": "/opt/projects/solar-host/jobs/job-abc123",
                    "submission_id": "sub-42",
                    "correlation_id": "corr-99",
                }
            ]
        }
    }


class StepStateResponse(BaseModel):
    """StepState enriched with the host log-file path and a recent-log snippet."""

    name: str = Field(examples=["download_model"])
    status: str = Field(examples=["completed"])
    container_id: str | None = Field(default=None, examples=["a1b2c3d4e5f6"])
    started_at: Any = None
    finished_at: Any = None
    duration_s: float | None = Field(default=None, examples=[12.5])
    exit_code: int | None = Field(default=None, examples=[0])
    error_message: str | None = None
    log_file: str = Field(
        examples=["/opt/projects/solar-host/jobs/job-abc123/logs/download_model.log"]
    )
    recent_logs: list[StepLogMessage] = []

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "download_model",
                    "status": "completed",
                    "container_id": "a1b2c3d4e5f6",
                    "started_at": "2026-05-28T10:00:00Z",
                    "finished_at": "2026-05-28T10:05:30Z",
                    "duration_s": 330.0,
                    "exit_code": 0,
                    "error_message": None,
                    "log_file": "/opt/projects/solar-host/jobs/job-abc123/logs/download_model.log",
                    "recent_logs": [],
                }
            ]
        }
    }

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

    job_id: str = Field(examples=["job-abc123"])
    name: str = Field(examples=["phi-3.5-finetune"])
    status: str = Field(examples=["completed"])
    steps: list[StepStateResponse] = []
    current_step_index: int = Field(examples=[0])
    workspace_path: str = Field(
        examples=["/opt/projects/solar-host/jobs/job-abc123"]
    )
    created_at: Any = None
    started_at: Any = None
    finished_at: Any = None
    retention_hours: float = Field(examples=[24.0])
    error_message: str | None = None
    submission_id: str | None = Field(default=None, examples=["sub-42"])
    correlation_id: str | None = Field(default=None, examples=["corr-99"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job-abc123",
                    "name": "phi-3.5-finetune",
                    "status": "completed",
                    "steps": [
                        {
                            "name": "download_model",
                            "status": "completed",
                            "container_id": "a1b2c3d4e5f6",
                            "started_at": "2026-05-28T10:00:00Z",
                            "finished_at": "2026-05-28T10:05:30Z",
                            "duration_s": 330.0,
                            "exit_code": 0,
                            "error_message": None,
                            "log_file": "/opt/projects/solar-host/jobs/job-abc123/logs/download_model.log",
                            "recent_logs": [],
                        }
                    ],
                    "current_step_index": 0,
                    "workspace_path": "/opt/projects/solar-host/jobs/job-abc123",
                    "created_at": "2026-05-28T10:00:00Z",
                    "started_at": "2026-05-28T10:00:00Z",
                    "finished_at": "2026-05-28T10:05:31Z",
                    "retention_hours": 24.0,
                    "error_message": None,
                    "submission_id": "sub-42",
                    "correlation_id": "corr-99",
                }
            ]
        }
    }

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
