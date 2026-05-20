"""Unit tests for solar_host.jobs.workspace."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.models import JobDefinition, StepDefinition
from solar_host.jobs.workspace import (
    build_job_json,
    check_disk_space,
    create_workspace,
    delete_workspace,
    validate_job_id,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(tmp_path: Path, uid: int = 0, gid: int = 0):
    """Return a minimal settings-like namespace backed by *tmp_path*."""

    class _Settings:
        jobs_dir = str(tmp_path / "jobs")
        container_uid = uid
        container_gid = gid
        min_free_disk_gb = 2.0

    return _Settings()


def _make_job(**kwargs) -> JobDefinition:
    defaults = {
        "job_id": "job-abc123",
        "name": "Test Job",
        "steps": [StepDefinition(name="train", image="acme/trainer:latest")],
    }
    defaults.update(kwargs)
    return JobDefinition(**defaults)


# ---------------------------------------------------------------------------
# validate_job_id
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "valid_id",
    [
        "job-abc123",
        "j_001",
        "job.v2",
        "A1B2C3",
        "a",
        "123",
    ],
)
def test_validate_job_id_valid(valid_id: str):
    validate_job_id(valid_id)  # must not raise


@pytest.mark.parametrize(
    "bad_id",
    [
        "",
        "job/with/slash",
        "job/../traversal",
        "job id",  # space
        "job@domain",  # @
        "job!bad",  # !
        "job:bad",  # :
        "../escape",
        "..leading",
    ],
)
def test_validate_job_id_invalid(bad_id: str):
    with pytest.raises(ValueError):
        validate_job_id(bad_id)


def test_validate_job_id_slash_raises():
    with pytest.raises(ValueError, match="/"):
        validate_job_id("a/b")


def test_validate_job_id_double_dot_raises():
    with pytest.raises(ValueError, match=r"\.\."):
        validate_job_id("a..b")


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------


def _fake_disk_usage(free_bytes: int):
    class _Usage:
        free = free_bytes
        total = free_bytes * 2
        used = free_bytes

    return _Usage()


def test_check_disk_space_sufficient(tmp_path: Path):
    # 10 GB free, 2 GB required — should not raise
    with patch("shutil.disk_usage", return_value=_fake_disk_usage(10 * 1024**3)):
        check_disk_space(tmp_path, min_free_gb=2.0)


def test_check_disk_space_insufficient(tmp_path: Path):
    # 1 GB free, 2 GB required — must raise
    with patch("shutil.disk_usage", return_value=_fake_disk_usage(1 * 1024**3)):
        with pytest.raises(InsufficientDiskError) as exc_info:
            check_disk_space(tmp_path, min_free_gb=2.0)
    err = exc_info.value
    assert err.required_gb == 2.0
    assert err.available_gb == pytest.approx(1.0, abs=0.01)


def test_check_disk_space_exactly_at_threshold(tmp_path: Path):
    # Exactly at threshold — should NOT raise
    with patch("shutil.disk_usage", return_value=_fake_disk_usage(2 * 1024**3)):
        check_disk_space(tmp_path, min_free_gb=2.0)


def test_check_disk_space_just_below_threshold(tmp_path: Path):
    # 1 byte below 2 GB — must raise
    slightly_below = 2 * 1024**3 - 1
    with patch("shutil.disk_usage", return_value=_fake_disk_usage(slightly_below)):
        with pytest.raises(InsufficientDiskError):
            check_disk_space(tmp_path, min_free_gb=2.0)


# ---------------------------------------------------------------------------
# build_job_json
# ---------------------------------------------------------------------------


def test_build_job_json_required_keys():
    job = _make_job()
    data = build_job_json(job)
    required = {
        "job_id",
        "name",
        "created_at",
        "pipeline",
        "base_model_uri",
        "training_data_uri",
        "model_selection",
        "deployment",
        "retention_hours",
        "steps",
    }
    assert required.issubset(data.keys())


def test_build_job_json_values():
    job = _make_job(
        job_id="j-001",
        name="Pipeline",
        steps=[
            StepDefinition(name="download", image="acme/dl:1.0"),
            StepDefinition(name="train", image="acme/trainer:1.0"),
        ],
        base_model_uri="repo://models/llama3",
        training_data_uri="s3://data",
        model_selection={"strategy": "best"},
        deployment={"endpoint": "/v1"},
        retention_hours=48.0,
    )
    data = build_job_json(job)
    assert data["job_id"] == "j-001"
    assert data["name"] == "Pipeline"
    assert data["pipeline"] == ["download", "train"]
    assert data["base_model_uri"] == "repo://models/llama3"
    assert data["training_data_uri"] == "s3://data"
    assert data["model_selection"] == {"strategy": "best"}
    assert data["deployment"] == {"endpoint": "/v1"}
    assert data["retention_hours"] == 48.0
    assert data["steps"] == {}


def test_build_job_json_steps_empty_dict():
    job = _make_job()
    data = build_job_json(job)
    assert data["steps"] == {}


# ---------------------------------------------------------------------------
# create_workspace
# ---------------------------------------------------------------------------


def test_create_workspace_directory_structure(tmp_path: Path):
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-struct")
    workspace = create_workspace(job, settings)

    assert workspace.is_dir()
    for sub in ("models", "data", "output", "config", "logs"):
        assert (workspace / sub).is_dir(), f"Missing subdir: {sub}"


def test_create_workspace_returns_correct_path(tmp_path: Path):
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-path")
    workspace = create_workspace(job, settings)
    assert workspace == Path(settings.jobs_dir) / "job-path"


def test_create_workspace_writes_job_json(tmp_path: Path):
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-json", name="JSON Test")
    workspace = create_workspace(job, settings)

    job_json_path = workspace / "config" / "job.json"
    assert job_json_path.is_file()
    data = json.loads(job_json_path.read_text())
    assert data["job_id"] == "job-json"
    assert data["name"] == "JSON Test"
    assert "pipeline" in data
    assert data["steps"] == {}


def test_create_workspace_no_training_json_when_absent(tmp_path: Path):
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-notrain")
    workspace = create_workspace(job, settings)
    assert not (workspace / "config" / "training.json").exists()


def test_create_workspace_writes_training_json(tmp_path: Path):
    settings = _make_settings(tmp_path)
    training_cfg = {"lr": 0.001, "epochs": 3}
    job = _make_job(job_id="job-train", training_config=training_cfg)
    workspace = create_workspace(job, settings)

    training_json_path = workspace / "config" / "training.json"
    assert training_json_path.is_file()
    data = json.loads(training_json_path.read_text())
    assert data == training_cfg


def test_create_workspace_directory_permissions(tmp_path: Path):
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-perms")
    workspace = create_workspace(job, settings)

    for sub in ("models", "data", "output", "config", "logs"):
        subpath = workspace / sub
        mode = subpath.stat().st_mode & 0o777
        assert mode == 0o755, f"{sub} has mode {oct(mode)}, expected 0o755"


def test_create_workspace_idempotent(tmp_path: Path):
    """Calling create_workspace twice for the same job should not raise."""
    settings = _make_settings(tmp_path)
    job = _make_job(job_id="job-idem")
    create_workspace(job, settings)
    create_workspace(job, settings)  # must not raise


# ---------------------------------------------------------------------------
# delete_workspace
# ---------------------------------------------------------------------------


def test_delete_workspace_removes_directory(tmp_path: Path):
    ws = tmp_path / "job-del"
    ws.mkdir()
    (ws / "file.txt").write_text("data")
    delete_workspace(ws)
    assert not ws.exists()


def test_delete_workspace_no_error_on_missing(tmp_path: Path):
    """Deleting a non-existent workspace must not raise."""
    missing = tmp_path / "does-not-exist"
    delete_workspace(missing)  # must not raise


def test_delete_workspace_no_error_on_repeated_delete(tmp_path: Path):
    ws = tmp_path / "job-twice"
    ws.mkdir()
    delete_workspace(ws)
    delete_workspace(ws)  # second call must not raise
