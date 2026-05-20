"""Unit tests for solar_host.jobs models and enums."""

from __future__ import annotations

from datetime import datetime

import pytest

from solar_host.jobs.models import (
    JobDefinition,
    JobState,
    JobStatus,
    StepDefinition,
    StepState,
    StepStatus,
)


# ---------------------------------------------------------------------------
# Enum values
# ---------------------------------------------------------------------------


def test_job_status_values():
    assert JobStatus.pending == "pending"
    assert JobStatus.running == "running"
    assert JobStatus.completed == "completed"
    assert JobStatus.failed == "failed"
    assert JobStatus.cancelled == "cancelled"


def test_step_status_values():
    assert StepStatus.pending == "pending"
    assert StepStatus.running == "running"
    assert StepStatus.completed == "completed"
    assert StepStatus.failed == "failed"
    assert StepStatus.cancelled == "cancelled"


def test_job_status_is_str():
    assert isinstance(JobStatus.running, str)


def test_step_status_is_str():
    assert isinstance(StepStatus.running, str)


# ---------------------------------------------------------------------------
# StepDefinition
# ---------------------------------------------------------------------------


def test_step_definition_required_fields():
    step = StepDefinition(name="train", image="acme/trainer:latest")
    assert step.name == "train"
    assert step.image == "acme/trainer:latest"


def test_step_definition_defaults():
    step = StepDefinition(name="train", image="acme/trainer:latest")
    assert step.environment == {}
    assert step.gpu is False
    assert step.is_preparation_step is False


def test_step_definition_with_all_fields():
    step = StepDefinition(
        name="download",
        image="acme/downloader:1.0",
        environment={"HF_MODEL": "llama3"},
        gpu=True,
        is_preparation_step=True,
    )
    assert step.name == "download"
    assert step.environment == {"HF_MODEL": "llama3"}
    assert step.gpu is True
    assert step.is_preparation_step is True


def test_step_definition_missing_required_raises():
    with pytest.raises(Exception):
        StepDefinition(name="train")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# JobDefinition
# ---------------------------------------------------------------------------


def _make_step(**kwargs) -> StepDefinition:
    return StepDefinition(name="train", image="acme/trainer:latest", **kwargs)


def test_job_definition_required_fields():
    job = JobDefinition(job_id="j-001", name="My Job", steps=[_make_step()])
    assert job.job_id == "j-001"
    assert job.name == "My Job"
    assert len(job.steps) == 1


def test_job_definition_defaults():
    job = JobDefinition(job_id="j-001", name="My Job", steps=[])
    assert job.retention_hours == 24.0
    assert job.min_free_disk_gb is None
    assert job.base_model_uri is None
    assert job.training_data_uri is None
    assert job.training_config is None
    assert job.model_selection is None
    assert job.deployment is None


def test_job_definition_optional_fields():
    job = JobDefinition(
        job_id="j-002",
        name="Full Job",
        steps=[_make_step()],
        retention_hours=48.0,
        min_free_disk_gb=10.0,
        base_model_uri="repo://models/llama3",
        training_data_uri="s3://bucket/data",
        training_config={"lr": 0.001},
        model_selection={"strategy": "best"},
        deployment={"endpoint": "/v1"},
    )
    assert job.retention_hours == 48.0
    assert job.min_free_disk_gb == 10.0
    assert job.base_model_uri == "repo://models/llama3"
    assert job.training_data_uri == "s3://bucket/data"
    assert job.training_config == {"lr": 0.001}
    assert job.model_selection == {"strategy": "best"}
    assert job.deployment == {"endpoint": "/v1"}


def test_job_definition_multiple_steps():
    steps = [
        StepDefinition(name="download", image="acme/dl:1.0", is_preparation_step=True),
        StepDefinition(name="train", image="acme/train:1.0"),
        StepDefinition(name="convert", image="acme/convert:1.0"),
    ]
    job = JobDefinition(job_id="j-003", name="Pipeline", steps=steps)
    assert len(job.steps) == 3
    assert job.steps[0].is_preparation_step is True
    assert job.steps[1].is_preparation_step is False


# ---------------------------------------------------------------------------
# StepState
# ---------------------------------------------------------------------------


def test_step_state_defaults():
    state = StepState(name="train")
    assert state.name == "train"
    assert state.status == StepStatus.pending
    assert state.container_id is None
    assert state.started_at is None
    assert state.finished_at is None
    assert state.duration_s is None
    assert state.exit_code is None
    assert state.error_message is None


def test_step_state_with_all_fields():
    now = datetime(2024, 1, 1, 12, 0, 0)
    later = datetime(2024, 1, 1, 12, 5, 0)
    state = StepState(
        name="train",
        status=StepStatus.completed,
        container_id="abc123",
        started_at=now,
        finished_at=later,
        duration_s=300.0,
        exit_code=0,
    )
    assert state.status == StepStatus.completed
    assert state.container_id == "abc123"
    assert state.duration_s == 300.0
    assert state.exit_code == 0


def test_step_state_failed():
    state = StepState(
        name="train",
        status=StepStatus.failed,
        exit_code=1,
        error_message="OOM",
    )
    assert state.status == StepStatus.failed
    assert state.exit_code == 1
    assert state.error_message == "OOM"


# ---------------------------------------------------------------------------
# JobState
# ---------------------------------------------------------------------------


def test_job_state_defaults():
    state = JobState(job_id="j-001", name="My Job")
    assert state.job_id == "j-001"
    assert state.name == "My Job"
    assert state.status == JobStatus.pending
    assert state.steps == []
    assert state.current_step_index == -1
    assert state.workspace_path == ""
    assert isinstance(state.created_at, datetime)
    assert state.started_at is None
    assert state.finished_at is None
    assert state.retention_hours == 24.0
    assert state.error_message is None


def test_job_state_with_steps():
    steps = [StepState(name="train"), StepState(name="convert")]
    state = JobState(
        job_id="j-001",
        name="My Job",
        status=JobStatus.running,
        steps=steps,
        current_step_index=0,
        workspace_path="/jobs/j-001",
    )
    assert state.status == JobStatus.running
    assert len(state.steps) == 2
    assert state.current_step_index == 0
    assert state.workspace_path == "/jobs/j-001"


def test_job_state_serialization():
    state = JobState(
        job_id="j-001",
        name="My Job",
        status=JobStatus.completed,
        workspace_path="/jobs/j-001",
    )
    data = state.model_dump()
    assert data["job_id"] == "j-001"
    assert data["status"] == "completed"
    assert data["workspace_path"] == "/jobs/j-001"


def test_job_state_json_roundtrip():
    state = JobState(
        job_id="j-001",
        name="My Job",
        steps=[StepState(name="train", status=StepStatus.completed, exit_code=0)],
    )
    json_str = state.model_dump_json()
    restored = JobState.model_validate_json(json_str)
    assert restored.job_id == state.job_id
    assert restored.steps[0].status == StepStatus.completed
    assert restored.steps[0].exit_code == 0
