"""Pydantic models and enums for job/step definitions and runtime state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
# Log models
# ---------------------------------------------------------------------------


class StepLogMessage(BaseModel):
    """A single log line captured from a step container."""

    seq: int
    timestamp: datetime
    stream: Literal["stdout", "stderr"]
    line: str
    completed: bool = False
    exit_code: int | None = None


# ---------------------------------------------------------------------------
# Input models (submitted by caller)
# ---------------------------------------------------------------------------


class GpuOptions(BaseModel):
    """GPU execution options for a step container.

    Maps directly to Docker SDK DeviceRequest semantics:
    - At least one of count or device_ids must be set.
    - device_ids takes precedence over count when both are set.
    - count=-1 means all available GPUs (must be set explicitly).
    """

    count: int | None = None
    device_ids: list[str] | None = None

    @model_validator(mode="after")
    def _require_count_or_device_ids(self) -> "GpuOptions":
        if self.count is None and self.device_ids is None:
            raise ValueError("gpu options require count or device_ids")
        return self


class StepDefinition(BaseModel):
    """Definition of a single pipeline step."""

    name: str = Field(examples=["download_model"])
    image: str = Field(examples=["imgrepo.damit.hu/supernova/download-model:v1"])
    environment: dict[str, str] = Field(
        default_factory=dict,
        examples=[{"MODEL_URI": "huggingface://microsoft/Phi-3.5-mini-instruct"}],
    )
    gpu: GpuOptions | None = Field(default=None, examples=[None])
    is_preparation_step: bool = False


class JobDefinition(BaseModel):
    """Full job definition submitted to the executor."""

    job_id: str = Field(examples=["job-abc123"])
    name: str = Field(examples=["phi-3.5-finetune"])
    steps: list[StepDefinition]
    retention_hours: float = Field(default=24.0, examples=[24.0])
    min_free_disk_gb: float | None = Field(default=None, examples=[5.0])
    base_model_uri: str | None = Field(
        default=None, examples=["huggingface://microsoft/Phi-3.5-mini-instruct"]
    )
    training_data_uri: str | None = Field(
        default=None, examples=["repo://training-dataset:v2"]
    )
    training_config: dict[str, Any] | None = Field(
        default=None,
        examples=[{"epochs": 3, "batch_size": 4, "learning_rate": 2e-5}],
    )
    model_selection: dict[str, Any] | None = Field(
        default=None,
        examples=[{"strategy": "best_loss", "metric": "eval_loss", "direction": "min"}],
    )
    deployment: dict[str, Any] | None = Field(
        default=None,
        examples=[{"target_model_name": "phi-3.5-finetuned", "replicas": 1}],
    )
    submission_id: str | None = Field(default=None, examples=["sub-42"])
    correlation_id: str | None = Field(default=None, examples=["corr-99"])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "job-abc123",
                    "name": "phi-3.5-finetune",
                    "steps": [
                        {
                            "name": "download_model",
                            "image": "imgrepo.damit.hu/supernova/download-model:v1",
                            "environment": {
                                "MODEL_URI": "huggingface://microsoft/Phi-3.5-mini-instruct"
                            },
                            "is_preparation_step": True,
                        },
                        {
                            "name": "download_dataset",
                            "image": "imgrepo.damit.hu/supernova/download-dataset:v1",
                            "environment": {
                                "DATASET_URI": "repo://training-dataset:v2"
                            },
                            "is_preparation_step": True,
                        },
                        {
                            "name": "train",
                            "image": "imgrepo.damit.hu/supernova/train:v1",
                            "environment": {},
                            "gpu": {"count": 1},
                        },
                    ],
                    "retention_hours": 24.0,
                    "min_free_disk_gb": 5.0,
                    "base_model_uri": "huggingface://microsoft/Phi-3.5-mini-instruct",
                    "training_data_uri": "repo://training-dataset:v2",
                    "training_config": {
                        "epochs": 3,
                        "batch_size": 4,
                        "learning_rate": 2e-5,
                    },
                    "model_selection": {
                        "strategy": "best_loss",
                        "metric": "eval_loss",
                        "direction": "min",
                    },
                    "deployment": {
                        "target_model_name": "phi-3.5-finetuned",
                        "replicas": 1,
                    },
                    "submission_id": "sub-42",
                    "correlation_id": "corr-99",
                }
            ]
        }
    }


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
    submission_id: str | None = None
    correlation_id: str | None = None
