"""Tests for GET /models endpoint (solar_host/routes/models.py)."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from solar_host.main import app
from solar_host.models_manager import (
    ManifestEntry,
    add_manifest_entry,
    ensure_models_dir,
    write_manifest,
    Manifest,
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
    # settings is a module-level singleton; patching the object attribute
    # propagates to all modules that imported it.
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
        """No models directory and no manifest → empty list, no error."""
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
# Manifest-backed models
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
        assert entry["in_manifest"] is True

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
        """The path in the response is taken from the manifest entry, not
        derived from the current models dir."""
        ensure_models_dir()
        add_manifest_entry(_make_entry(path="/custom/absolute/path/to/model"))
        resp = client.get("/models", headers=_headers())
        assert resp.json()[0]["path"] == "/custom/absolute/path/to/model"


# ---------------------------------------------------------------------------
# Orphaned manifest entries (in manifest but directory missing)
# ---------------------------------------------------------------------------


class TestOrphanedManifestEntries:
    def test_orphaned_entry_still_returned(self, client: TestClient):
        """Manifest entry whose directory does not exist is returned with
        in_manifest=True so callers can detect stale cache entries."""
        ensure_models_dir()
        add_manifest_entry(_make_entry())
        # Do NOT create the directory — simulate orphaned entry.
        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["in_manifest"] is True
        assert data[0]["name"] == "repo--iris-osl--v3"


# ---------------------------------------------------------------------------
# Untracked directories (on disk but not in manifest)
# ---------------------------------------------------------------------------


class TestUntrackedDirectories:
    def test_untracked_dir_is_included(self, client: TestClient, _isolated_env: Path):
        ensure_models_dir()
        # Manually create a model directory with a file.
        model_dir = _isolated_env / "manually-placed-model"
        model_dir.mkdir()
        (model_dir / "model.gguf").write_bytes(b"x" * 512)

        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        entry = data[0]
        assert entry["name"] == "manually-placed-model"
        assert entry["in_manifest"] is False
        assert entry["source_uri"] is None
        assert entry["checksum"] is None
        assert entry["size_bytes"] == 512

    def test_untracked_dir_path_is_absolute(
        self, client: TestClient, _isolated_env: Path
    ):
        ensure_models_dir()
        model_dir = _isolated_env / "some-model"
        model_dir.mkdir()

        resp = client.get("/models", headers=_headers())
        data = resp.json()
        assert Path(data[0]["path"]).is_absolute()

    def test_files_in_models_dir_root_are_ignored(
        self, client: TestClient, _isolated_env: Path
    ):
        """Non-directory entries in MODELS_DIR (e.g. manifest.json) must not
        appear in the response."""
        ensure_models_dir()
        # manifest.json is written by write_manifest; also add a stray file.
        write_manifest(Manifest(models=[]))
        (_isolated_env / "stray-file.txt").write_text("hello")

        resp = client.get("/models", headers=_headers())
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Mixed: manifest + untracked
# ---------------------------------------------------------------------------


class TestMixed:
    def test_manifest_and_untracked_both_returned(
        self, client: TestClient, _isolated_env: Path
    ):
        ensure_models_dir()
        # Tracked model (manifest entry, directory present).
        tracked_dir = _isolated_env / "repo--iris-osl--v3"
        tracked_dir.mkdir()
        add_manifest_entry(
            _make_entry(
                path=str(tracked_dir.resolve()),
                size_bytes=100,
            )
        )
        # Untracked model (directory only, no manifest entry).
        untracked_dir = _isolated_env / "manually-placed"
        untracked_dir.mkdir()
        (untracked_dir / "weights.bin").write_bytes(b"\x00" * 256)

        resp = client.get("/models", headers=_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        by_name = {e["name"]: e for e in data}
        assert "repo--iris-osl--v3" in by_name
        assert "manually-placed" in by_name

        tracked = by_name["repo--iris-osl--v3"]
        assert tracked["in_manifest"] is True
        assert tracked["size_bytes"] == 100  # from manifest

        untracked = by_name["manually-placed"]
        assert untracked["in_manifest"] is False
        assert untracked["size_bytes"] == 256  # computed from disk

    def test_manifest_entry_takes_priority_over_disk_for_size(
        self, client: TestClient, _isolated_env: Path
    ):
        """When a directory exists AND has a manifest entry, the size from
        the manifest is used (not recomputed from disk)."""
        ensure_models_dir()
        model_dir = _isolated_env / "repo--iris-osl--v3"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"\x00" * 64)
        add_manifest_entry(_make_entry(path=str(model_dir.resolve()), size_bytes=9999))

        resp = client.get("/models", headers=_headers())
        data = resp.json()
        assert len(data) == 1
        assert data[0]["size_bytes"] == 9999  # manifest wins
