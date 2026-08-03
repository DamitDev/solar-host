"""Tests for POST /jobs, GET /jobs/{id}, DELETE /jobs/{id} endpoints (S-027)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.models import JobState, JobStatus, StepState
from solar_host.jobs.store import JobStore
from solar_host.main import app

API_KEY = "test-jobs-key-s027"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _make_job_state(
    job_id: str = "job-001",
    status: JobStatus = JobStatus.running,
    **overrides,
) -> JobState:
    now = datetime.now(UTC)
    defaults: dict = {
        "job_id": job_id,
        "name": "Test Job",
        "status": status,
        "steps": [StepState(name="step1")],
        "current_step_index": -1,
        "workspace_path": f"/tmp/solar-jobs/{job_id}",
        "created_at": now,
        "started_at": now,
        "retention_hours": 24.0,
    }
    defaults.update(overrides)
    return JobState(**defaults)


def _submit_body(**overrides) -> dict:
    base = {
        "job_id": "job-001",
        "name": "Test Job",
        "steps": [{"name": "step1", "image": "test/img:latest"}],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch):
    """Patch settings so each test gets a fresh tmp jobs dir, no WS clients,
    and a fixed API key."""
    monkeypatch.setattr("solar_host.config.settings.jobs_dir", str(tmp_path / "jobs"))
    monkeypatch.setattr("solar_host.config.settings.solar_control_url", "")
    monkeypatch.setattr("solar_host.config.settings.api_key", API_KEY)


@pytest.fixture()
def mock_store() -> JobStore:
    return JobStore()


@pytest.fixture()
def mock_executor() -> MagicMock:
    ex = MagicMock()
    ex.submit_job = AsyncMock()
    ex.cancel_job = AsyncMock()
    ex.await_job = AsyncMock()
    ex.await_all = AsyncMock()
    return ex


@pytest.fixture()
def client(mock_executor: MagicMock, mock_store: JobStore):
    """Full-lifespan TestClient with DockerService patched out and app state
    replaced with fresh mocks so each test is fully isolated."""
    with (
        patch("solar_host.docker.service.DockerService"),
        TestClient(app, raise_server_exceptions=True) as c,
    ):
        app.state.job_executor = mock_executor
        app.state.job_store = mock_store
        yield c


# ---------------------------------------------------------------------------
# Authentication smoke tests
# ---------------------------------------------------------------------------


class TestAuth:
    def test_post_missing_api_key_returns_401(self, client: TestClient):
        resp = client.post("/jobs", json=_submit_body())
        assert resp.status_code == 401

    def test_post_wrong_api_key_returns_401(self, client: TestClient):
        resp = client.post("/jobs", json=_submit_body(), headers={"X-API-Key": "bad"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /jobs
# ---------------------------------------------------------------------------


class TestPostJob:
    def test_happy_path_returns_202(self, client: TestClient, mock_executor: MagicMock):
        state = _make_job_state()
        mock_executor.submit_job.return_value = state

        resp = client.post("/jobs", json=_submit_body(), headers=_headers())

        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-001"
        assert body["status"] == "running"
        assert body["workspace_path"] == state.workspace_path
        mock_executor.submit_job.assert_awaited_once()

    def test_empty_steps_returns_400(
        self, client: TestClient, mock_executor: MagicMock
    ):
        mock_executor.submit_job.side_effect = ValueError(
            "steps list must not be empty"
        )
        resp = client.post("/jobs", json=_submit_body(steps=[]), headers=_headers())
        assert resp.status_code == 400

    def test_traversal_job_id_returns_400(
        self, client: TestClient, mock_executor: MagicMock
    ):
        mock_executor.submit_job.side_effect = ValueError("invalid job_id: ../etc")
        resp = client.post(
            "/jobs", json=_submit_body(job_id="../etc"), headers=_headers()
        )
        assert resp.status_code == 400

    def test_duplicate_job_id_returns_409(
        self, client: TestClient, mock_executor: MagicMock
    ):
        mock_executor.submit_job.side_effect = KeyError("already exists in store")
        resp = client.post("/jobs", json=_submit_body(), headers=_headers())
        assert resp.status_code == 409

    def test_executor_unavailable_returns_503(self, client: TestClient):
        app.state.job_executor = None
        try:
            resp = client.post("/jobs", json=_submit_body(), headers=_headers())
            assert resp.status_code == 503
            assert resp.json()["detail"]["error"] == "executor_unavailable"
        finally:
            app.state.job_executor = client.app.state.__dict__.get(
                "_mock_executor_restore"
            )

    def test_insufficient_disk_returns_507(
        self, client: TestClient, mock_executor: MagicMock
    ):
        mock_executor.submit_job.side_effect = InsufficientDiskError(
            required_gb=10.0, available_gb=3.5
        )
        resp = client.post("/jobs", json=_submit_body(), headers=_headers())
        assert resp.status_code == 507
        detail = resp.json()["detail"]
        assert detail["error"] == "insufficient_storage"
        assert detail["required_gb"] == 10.0
        assert detail["available_gb"] == 3.5

    def test_correlation_ids_round_trip(
        self, client: TestClient, mock_executor: MagicMock
    ):
        state = _make_job_state(submission_id="sub-42", correlation_id="corr-99")
        mock_executor.submit_job.return_value = state

        body = _submit_body(submission_id="sub-42", correlation_id="corr-99")
        resp = client.post("/jobs", json=body, headers=_headers())

        assert resp.status_code == 202
        data = resp.json()
        assert data["submission_id"] == "sub-42"
        assert data["correlation_id"] == "corr-99"


# ---------------------------------------------------------------------------
# GET /jobs/{id}
# ---------------------------------------------------------------------------


class TestGetJob:
    def test_happy_path_returns_200(
        self, client: TestClient, mock_store: JobStore, tmp_path: Path
    ):
        state = _make_job_state(submission_id="sub-1", correlation_id="corr-1")
        mock_store.add(state)

        resp = client.get("/jobs/job-001", headers=_headers())

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-001"
        assert data["status"] == "running"
        assert data["workspace_path"] == state.workspace_path
        assert data["submission_id"] == "sub-1"
        assert data["correlation_id"] == "corr-1"

        # Per-step log_file path must follow the expected pattern.
        step = data["steps"][0]
        assert step["name"] == "step1"
        jobs_dir = str(tmp_path / "jobs")
        assert step["log_file"] == f"{jobs_dir}/job-001/logs/step1.log"

        # recent_logs empty — buffer has nothing for this job.
        assert step["recent_logs"] == []

    def test_unknown_id_returns_404(self, client: TestClient):
        resp = client.get("/jobs/no-such-job", headers=_headers())
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# DELETE /jobs/{id}
# ---------------------------------------------------------------------------


class TestDeleteJob:
    def test_happy_running_job_returns_200(
        self,
        client: TestClient,
        mock_executor: MagicMock,
        mock_store: JobStore,
    ):
        state = _make_job_state()
        mock_store.add(state)

        with patch("solar_host.routes.jobs.delete_workspace") as mock_del_ws:
            resp = client.delete("/jobs/job-001", headers=_headers())

        assert resp.status_code == 200
        body = resp.json()
        assert body["detail"] == "cancelled"
        assert body["job_id"] == "job-001"

        mock_executor.cancel_job.assert_awaited_once_with("job-001")
        mock_executor.await_job.assert_awaited_once_with("job-001", timeout=20.0)
        mock_del_ws.assert_called_once_with(Path(state.workspace_path))
        assert mock_store.get("job-001") is None

    def test_unknown_id_returns_404(self, client: TestClient):
        resp = client.delete("/jobs/ghost-job", headers=_headers())
        assert resp.status_code == 404

    def test_terminal_job_returns_409(self, client: TestClient, mock_store: JobStore):
        for terminal_status in (
            JobStatus.completed,
            JobStatus.failed,
            JobStatus.cancelled,
        ):
            store = JobStore()
            store.add(_make_job_state(job_id="term-job", status=terminal_status))
            app.state.job_store = store

            resp = client.delete("/jobs/term-job", headers=_headers())
            assert resp.status_code == 409, f"expected 409 for {terminal_status}"
