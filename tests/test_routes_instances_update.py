"""Tests for PUT /instances/{id} ownership-marker updates (D-017 disown)."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from solar_host.main import app

API_KEY = "test-disown-key"


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch):
    """Fresh config dir, no WS clients, fixed API key."""
    monkeypatch.setattr(
        "solar_host.config.settings.config_file", str(tmp_path / "config.json")
    )
    monkeypatch.setattr("solar_host.config.settings.solar_control_url", "")
    monkeypatch.setattr("solar_host.config.settings.api_key", API_KEY)


@pytest.fixture()
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _create_instance(client: TestClient, *, managed_by: str = "intent") -> str:
    resp = client.post(
        "/instances",
        headers=_headers(),
        json={
            "config": {
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test-disown",
                "model_source": "repo://test-disown:v1",
            },
            "managed_by": managed_by,
            "intent_id": "intent-123",
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["instance"]["id"]


class TestDisownUpdate:
    def test_marker_clearing_update(self, client: TestClient):
        """PUT with managed_by/intent_id null clears ownership markers."""
        instance_id = _create_instance(client)
        inst = client.get(f"/instances/{instance_id}", headers=_headers()).json()
        assert inst["managed_by"] == "intent"
        assert inst["intent_id"] == "intent-123"

        resp = client.put(
            f"/instances/{instance_id}",
            headers=_headers(),
            json={"managed_by": None, "intent_id": None},
        )
        assert resp.status_code == 200, resp.text

        inst = client.get(f"/instances/{instance_id}", headers=_headers()).json()
        assert inst["managed_by"] is None
        assert inst["intent_id"] is None

    def test_config_only_update_preserves_markers(self, client: TestClient):
        """A config-only PUT must not clobber ownership markers."""
        instance_id = _create_instance(client)

        resp = client.put(
            f"/instances/{instance_id}",
            headers=_headers(),
            json={
                "config": {
                    "backend_type": "llamacpp",
                    "model": "/tmp/test.gguf",
                    "alias": "test-disown",
                    "model_source": "repo://test-disown:v1",
                    "threads": 8,
                }
            },
        )
        assert resp.status_code == 200, resp.text

        inst = client.get(f"/instances/{instance_id}", headers=_headers()).json()
        assert inst["managed_by"] == "intent"
        assert inst["intent_id"] == "intent-123"
        assert inst["config"]["threads"] == 8
