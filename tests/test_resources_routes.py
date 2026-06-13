"""HTTP route tests for POST/GET/DELETE /resources endpoints (S-034).

Uses TestClient(app) like test_routes_jobs.py. DockerService is patched out;
ResourceManager is injected directly into app.state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from solar_host.jobs.models import JobState, JobStatus, StepState
from solar_host.jobs.store import JobStore
from solar_host.main import app
from solar_host.resources.manager import ResourceManager

API_KEY = "test-resources-key-s034"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEM_VRAM = {
    "used_gb": 8.0,
    "total_gb": 24.0,
    "available_gb": 16.0,
    "percent": 33.33,
    "memory_type": "VRAM",
}
_DISK = {"used_gb": 100.0, "total_gb": 500.0, "available_gb": 400.0}


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _reservation_body(**overrides) -> dict:
    base = {
        "job_id": "job-001",
        "workload_type": "training",
        "vram_gb": 4.0,
        "ram_gb": 2.0,
    }
    base.update(overrides)
    return base


def _make_running_job(job_id: str = "job-001") -> JobState:
    now = datetime.now(UTC)
    return JobState(
        job_id=job_id,
        name="Test Job",
        status=JobStatus.running,
        steps=[StepState(name="step1")],
        current_step_index=0,
        workspace_path=f"/tmp/jobs/{job_id}",
        created_at=now,
        started_at=now,
        retention_hours=24.0,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("solar_host.config.settings.jobs_dir", str(tmp_path / "jobs"))
    monkeypatch.setattr("solar_host.config.settings.solar_control_url", "")
    monkeypatch.setattr("solar_host.config.settings.api_key", API_KEY)


@pytest.fixture(autouse=True)
def _patch_memory(monkeypatch):
    import psutil

    monkeypatch.setattr(
        "solar_host.resources.manager.get_memory_info", lambda: _MEM_VRAM
    )
    monkeypatch.setattr(
        "solar_host.resources.manager.get_disk_info", lambda _path: _DISK
    )
    vm = MagicMock()
    vm.used = int(4.0 * 1024**3)
    vm.total = int(16.0 * 1024**3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)


@pytest.fixture()
def store() -> JobStore:
    return JobStore()


@pytest.fixture()
def manager(store: JobStore, tmp_path: Path) -> ResourceManager:
    return ResourceManager(
        job_store=store,
        docker_service=None,
        job_executor=None,
        jobs_dir=str(tmp_path / "jobs"),
    )


@pytest.fixture()
def client(manager: ResourceManager):
    with patch("solar_host.docker.service.DockerService"):
        with TestClient(app, raise_server_exceptions=True) as c:
            app.state.resource_manager = manager
            yield c


# ---------------------------------------------------------------------------
# Auth smoke test
# ---------------------------------------------------------------------------


class TestAuth:
    def test_post_without_key_returns_401(self, client: TestClient):
        resp = client.post("/resources/reservations", json=_reservation_body())
        assert resp.status_code == 401

    def test_get_without_key_returns_401(self, client: TestClient):
        resp = client.get("/resources/")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /resources/reservations
# ---------------------------------------------------------------------------


class TestPostReservation:
    def test_happy_path_returns_201_with_view(self, client: TestClient):
        resp = client.post(
            "/resources/reservations",
            json=_reservation_body(),
            headers=_headers(),
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["job_id"] == "job-001"
        assert body["status"] == "pending"
        assert body["vram_gb"] == 4.0
        assert "id" in body

    def test_capacity_exceeded_returns_409(self, client: TestClient):
        resp = client.post(
            "/resources/reservations",
            json=_reservation_body(vram_gb=20.0),  # > 16 available
            headers=_headers(),
        )
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"] == "capacity_exceeded"
        assert body["dimension"] == "vram"
        assert "requested_gb" in body
        assert "available_gb" in body

    def test_invalid_body_returns_422(self, client: TestClient):
        resp = client.post(
            "/resources/reservations",
            json={"job_id": "job-001"},  # missing required fields
            headers=_headers(),
        )
        assert resp.status_code == 422

    def test_conflicting_expiry_returns_422(self, client: TestClient):
        body = _reservation_body()
        body["ttl_seconds"] = 60.0
        body["expires_at"] = "2099-01-01T00:00:00Z"
        resp = client.post(
            "/resources/reservations",
            json=body,
            headers=_headers(),
        )
        assert resp.status_code == 422

    def test_503_when_manager_unavailable(self, client: TestClient):
        app.state.resource_manager = None
        try:
            resp = client.post(
                "/resources/reservations",
                json=_reservation_body(),
                headers=_headers(),
            )
            assert resp.status_code == 503
        finally:
            # Restore — next test's fixture will set it again anyway.
            app.state.resource_manager = client.app.state.resource_manager  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# GET /resources
# ---------------------------------------------------------------------------


class TestGetResources:
    def test_returns_200_with_snapshot_shape(self, client: TestClient):
        resp = client.get("/resources/", headers=_headers())
        assert resp.status_code == 200
        body = resp.json()
        assert "memory_type" in body
        assert "reservations" in body
        assert isinstance(body["reservations"], list)

    def test_snapshot_includes_created_reservation(self, client: TestClient):
        client.post(
            "/resources/reservations",
            json=_reservation_body(),
            headers=_headers(),
        )
        resp = client.get("/resources/", headers=_headers())
        body = resp.json()
        assert len(body["reservations"]) == 1
        assert body["reservations"][0]["job_id"] == "job-001"

    def test_reservation_status_reflects_job_store(
        self, client: TestClient, store: JobStore
    ):
        post_resp = client.post(
            "/resources/reservations",
            json=_reservation_body(job_id="job-running"),
            headers=_headers(),
        )
        assert post_resp.status_code == 201

        store.add(_make_running_job("job-running"))
        resp = client.get("/resources/", headers=_headers())
        body = resp.json()
        reservation_view = next(
            v for v in body["reservations"] if v["job_id"] == "job-running"
        )
        assert reservation_view["status"] == "running"


# ---------------------------------------------------------------------------
# DELETE /resources/reservations/{id}
# ---------------------------------------------------------------------------


class TestDeleteReservation:
    def test_happy_path_pending_returns_200(self, client: TestClient):
        post_resp = client.post(
            "/resources/reservations",
            json=_reservation_body(),
            headers=_headers(),
        )
        res_id = post_resp.json()["id"]

        del_resp = client.delete(
            f"/resources/reservations/{res_id}",
            headers=_headers(),
        )
        assert del_resp.status_code == 200
        body = del_resp.json()
        assert body["detail"] == "released"
        assert body["id"] == res_id

    def test_unknown_id_returns_404(self, client: TestClient):
        resp = client.delete(
            "/resources/reservations/res-doesnotexist",
            headers=_headers(),
        )
        assert resp.status_code == 404

    def test_running_job_returns_409(self, client: TestClient, store: JobStore):
        store.add(_make_running_job("job-001"))
        post_resp = client.post(
            "/resources/reservations",
            json=_reservation_body(job_id="job-001"),
            headers=_headers(),
        )
        res_id = post_resp.json()["id"]

        del_resp = client.delete(
            f"/resources/reservations/{res_id}",
            headers=_headers(),
        )
        assert del_resp.status_code == 409


# ---------------------------------------------------------------------------
# Health payload
# ---------------------------------------------------------------------------


class TestHealthPayload:
    """Verify that send_health includes the reservations block when a
    ResourceManager is provided."""

    def test_send_health_includes_reservations_block(self, manager: ResourceManager):
        import asyncio

        from solar_host.ws_client import SolarControlClient

        sio_mock = MagicMock()
        emit_calls: list[dict] = []

        async def fake_emit(event, data, namespace):
            emit_calls.append({"event": event, "data": data})

        sio_mock.connected = True
        sio_mock.emit = fake_emit

        c = SolarControlClient.__new__(SolarControlClient)
        c._sio = sio_mock
        c._connected = True

        asyncio.run(c.send_health(resource_manager=manager))

        assert len(emit_calls) == 1
        payload = emit_calls[0]["data"]["data"]
        assert "reservations" in payload
        assert "active_count" in payload["reservations"]
