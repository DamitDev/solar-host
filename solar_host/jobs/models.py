"""Pydantic models and enums for job/step definitions and runtime state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class StepStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


# ---------------------------------------------------------------------------
# Input models (submitted by caller)
# ---------------------------------------------------------------------------


class StepDefinition(BaseModel):
    """Definition of a single pipeline step."""

    name: str
    image: str
    environment: dict[str, str] = Field(default_factory=dict)
    gpu: bool = False
    is_preparation_step: bool = False


class JobDefinition(BaseModel):
    """Full job definition submitted to the executor."""

    job_id: str
    name: str
    steps: list[StepDefinition]
    retention_hours: float = 24.0
    min_free_disk_gb: float | None = None
    base_model_uri: str | None = None
    training_data_uri: str | None = None
    training_config: dict[str, Any] | None = None
    model_selection: dict[str, Any] | None = None
    deployment: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# State models (runtime tracking)
# ---------------------------------------------------------------------------


class StepState(BaseModel):
    """Runtime state of a single step."""

    name: str
    status: StepStatus = StepStatus.pending
    container_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    error_message: str | None = None


class JobState(BaseModel):
    """Runtime state of an entire job."""

    job_id: str
    name: str
    status: JobStatus = JobStatus.pending
    steps: list[StepState] = Field(default_factory=list)
    current_step_index: int = -1
    workspace_path: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    retention_hours: float = 24.0
    error_message: str | None = None
