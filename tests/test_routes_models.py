"""Tests for GET /models endpoint (solar_host/routes/models.py)."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from solar_host.main import app
from solar_host.models_manager import (
    Manifest,
    ManifestEntry,
    add_manifest_entry,
    ensure_models_dir,
    write_manifest,
)

API_KEY = "test-key-s014"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch):
    """Patch settings so each test gets a fresh tmp models dir and a fixed
    API key. The solar_control_url is cleared so no WebSocket clients are
    started during the lifespan."""
    models = tmp_path / "models"
    monkeypatch.setattr("solar_host.config.settings.models_dir", str(models))
    monkeypatch.setattr("solar_host.config.settings.solar_control_url", "")
    monkeypatch.setattr("solar_host.config.settings.api_key", API_KEY)
    return models


@pytest.fixture()
def client():
    """HTTP test client that runs the full app lifespan."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _make_entry(**overrides) -> ManifestEntry:
    defaults = {
        "slug": "repo--iris-osl--v3",
        "source_uri": "repo://iris-osl:v3",
        "path": "/opt/solar/models/repo--iris-osl--v3",
        "size_bytes": 4815162342,
        "digest": "sha256:abc123",
        "downloaded_at": "2026-03-31T14:22:00Z",
    }
    defaults.update(overrides)
    return ManifestEntry(**defaults)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_api_key_returns_401(self, client: TestClient):
        resp = client.get("/models")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient):
        resp = client.get("/models", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_correct_api_key_returns_200(self, client: TestClient):
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class TestEmptyState:
    def test_no_manifest_returns_empty_list(self, client: TestClient):
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        assert resp.json() == []

    def test_empty_manifest_returns_empty_list(self, client: TestClient):
        ensure_models_dir()
        write_manifest(Manifest(models=[]))
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Manifest-only listing
# ---------------------------------------------------------------------------


class TestManifestModels:
    def test_single_manifest_entry(self, client: TestClient):
        ensure_models_dir()
        add_manifest_entry(_make_entry())
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["name"] == "repo--iris-osl--v3"
        assert entry["source_uri"] == "repo://iris-osl:v3"
        assert entry["size_bytes"] == 4815162342
        assert entry["checksum"] == "sha256:abc123"
        assert entry["downloaded_at"] == "2026-03-31T14:22:00Z"
        assert "in_manifest" not in entry

    def test_multiple_manifest_entries(self, client: TestClient):
        ensure_models_dir()
        add_manifest_entry(
            _make_entry(slug="repo--iris-osl--v3", source_uri="repo://iris-osl:v3")
        )
        add_manifest_entry(
            _make_entry(
                slug="hf--microsoft--phi-3",
                source_uri="huggingface://microsoft/phi-3",
                path="/opt/solar/models/hf--microsoft--phi-3",
                size_bytes=7000000000,
                digest=None,
            )
        )
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = {e["name"] for e in data}
        assert names == {"repo--iris-osl--v3", "hf--microsoft--phi-3"}

    def test_manifest_entry_without_digest(self, client: TestClient):
        ensure_models_dir()
        add_manifest_entry(_make_entry(digest=None))
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert data[0]["checksum"] is None

    def test_path_comes_from_manifest(self, client: TestClient):
        ensure_models_dir()
        add_manifest_entry(_make_entry(path="/custom/absolute/path/to/model"))
        resp = client.get("/models", headers=_headers())
        assert resp.json()[0]["path"] == "/custom/absolute/path/to/model"


class TestManifestOnlyNoDirectoryScan:
    def test_manifest_entry_returned_without_directory_on_disk(
        self, client: TestClient
    ):
        """Manifest rows are returned even when the model directory does not exist."""
        ensure_models_dir()
        add_manifest_entry(_make_entry())
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "repo--iris-osl--v3"

    def test_extra_directories_on_disk_ignored(
        self, client: TestClient, _isolated_env: Path
    ):
        """Directories under MODELS_DIR that are not in the manifest are not listed."""
        ensure_models_dir()
        write_manifest(Manifest(models=[]))
        orphan = _isolated_env / "manually-placed-model"
        orphan.mkdir()
        (orphan / "model.gguf").write_bytes(b"x" * 512)

        resp = client.get("/models", headers=_headers())
        assert resp.json() == []

    def test_only_manifest_entries_when_both_exist(
        self, client: TestClient, _isolated_env: Path
    ):
        ensure_models_dir()
        tracked_dir = _isolated_env / "repo--iris-osl--v3"
        tracked_dir.mkdir()
        add_manifest_entry(_make_entry(path=str(tracked_dir.resolve()), size_bytes=100))
        untracked_dir = _isolated_env / "manually-placed"
        untracked_dir.mkdir()
        (untracked_dir / "weights.bin").write_bytes(b"\x00" * 256)

        resp = client.get("/models", headers=_headers())
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "repo--iris-osl--v3"
        assert data[0]["size_bytes"] == 100
