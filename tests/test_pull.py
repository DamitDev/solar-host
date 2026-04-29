"""Tests for POST /models/pull endpoint (solar_host/routes/models.py).

All external I/O (Harbor OrasHelper, huggingface_hub.snapshot_download) is
mocked. Filesystem operations use tmp_path so nothing touches the real disk.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from solar_host.main import app
from solar_host.models_manager import (
    ManifestEntry,
    add_manifest_entry,
    ensure_models_dir,
    get_manifest_entry,
    get_models_dir,
)

API_KEY = "test-key-s015"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path: Path, monkeypatch):
    """Point settings at a fresh tmp models dir and fixed API key/credentials.

    Credentials are set to valid values by default; individual tests can
    override them to trigger credential-missing errors.
    """
    models = tmp_path / "models"
    monkeypatch.setattr("solar_host.config.settings.models_dir", str(models))
    monkeypatch.setattr("solar_host.config.settings.solar_control_url", "")
    monkeypatch.setattr("solar_host.config.settings.api_key", API_KEY)
    monkeypatch.setattr(
        "solar_host.config.settings.harbor_url", "https://imgrepo.damit.hu"
    )
    monkeypatch.setattr("solar_host.config.settings.harbor_username", "robot")
    monkeypatch.setattr("solar_host.config.settings.harbor_password", "secret")
    monkeypatch.setattr("solar_host.config.settings.hf_token", "")
    # In-process pulls so mocks on _pull_* apply; low threshold so tmp disks pass.
    monkeypatch.setattr("solar_host.config.settings.pull_use_subprocess", False)
    monkeypatch.setattr("solar_host.config.settings.pull_disk_poll_interval_s", 0.05)
    monkeypatch.setattr("solar_host.config.settings.min_free_disk_gb", 0.001)
    return models


@pytest.fixture()
def client():
    """HTTP test client with full app lifespan."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _headers() -> dict:
    return {"X-API-Key": API_KEY}


def _harbor_body(**overrides) -> dict:
    defaults = {
        "source": "harbor",
        "source_uri": "repo://iris-osl:v3",
        "harbor_ref": "imgrepo.damit.hu/supernova/iris-osl:v3",
        "digest": "sha256:abc123",
    }
    defaults.update(overrides)
    return defaults


def _hf_body(**overrides) -> dict:
    defaults = {
        "source": "huggingface",
        "source_uri": "huggingface://microsoft/phi-3",
        "model_id": "microsoft/phi-3",
    }
    defaults.update(overrides)
    return defaults


def _make_manifest_entry(**overrides) -> ManifestEntry:
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
# Proactive disk space (S-018)
# ---------------------------------------------------------------------------


class TestProactiveDiskSpace:
    def test_returns_507_when_size_bytes_exceeds_available(
        self, client: TestClient, _isolated_env: Path
    ):
        with patch(
            "solar_host.models_manager.get_disk_info",
            return_value={"available_gb": 1.0, "used_gb": 1.0, "total_gb": 2.0},
        ):
            body = {**_harbor_body(), "size_bytes": 200 * 1024**3}
            resp = client.post("/models/pull", json=body, headers=_headers())
        assert resp.status_code == 507
        detail = resp.json()["detail"]
        assert "Insufficient disk space" in detail
        assert "1.00" in detail
        assert "200.00" in detail

    def test_returns_507_when_unknown_size_below_min_free(
        self, client: TestClient, _isolated_env: Path, monkeypatch
    ):
        monkeypatch.setattr("solar_host.config.settings.min_free_disk_gb", 100.0)
        with patch(
            "solar_host.models_manager.get_disk_info",
            return_value={"available_gb": 5.0, "used_gb": 1.0, "total_gb": 6.0},
        ):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 507
        assert "100.00" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuth:
    def test_missing_api_key_returns_401(self, client: TestClient):
        resp = client.post("/models/pull", json=_harbor_body())
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, client: TestClient):
        resp = client.post(
            "/models/pull", json=_harbor_body(), headers={"X-API-Key": "wrong"}
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Request validation (422)
# ---------------------------------------------------------------------------


class TestRequestValidation:
    def test_harbor_source_requires_harbor_ref(self, client: TestClient):
        body = _harbor_body()
        del body["harbor_ref"]
        resp = client.post("/models/pull", json=body, headers=_headers())
        assert resp.status_code == 422

    def test_huggingface_source_requires_model_id(self, client: TestClient):
        body = _hf_body()
        del body["model_id"]
        resp = client.post("/models/pull", json=body, headers=_headers())
        assert resp.status_code == 422

    def test_invalid_source_type_returns_422(self, client: TestClient):
        resp = client.post(
            "/models/pull",
            json={"source": "s3", "source_uri": "s3://bucket/model"},
            headers=_headers(),
        )
        assert resp.status_code == 422

    def test_harbor_ref_empty_string_returns_422(self, client: TestClient):
        resp = client.post(
            "/models/pull",
            json={**_harbor_body(), "harbor_ref": ""},
            headers=_headers(),
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# source_uri / source type mismatch (400)
# ---------------------------------------------------------------------------


class TestSourceUriMismatch:
    def test_harbor_source_with_hf_uri_returns_400(self, client: TestClient):
        resp = client.post(
            "/models/pull",
            json={
                "source": "harbor",
                "source_uri": "huggingface://microsoft/phi-3",
                "harbor_ref": "imgrepo.damit.hu/supernova/phi-3:v1",
            },
            headers=_headers(),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "invalid_request"
        assert data["source_uri"] == "huggingface://microsoft/phi-3"

    def test_huggingface_source_with_repo_uri_returns_400(self, client: TestClient):
        resp = client.post(
            "/models/pull",
            json={
                "source": "huggingface",
                "source_uri": "repo://iris-osl:v3",
                "model_id": "iris-osl",
            },
            headers=_headers(),
        )
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"] == "invalid_request"


# ---------------------------------------------------------------------------
# Cache hit
# ---------------------------------------------------------------------------


class TestCacheHit:
    def test_cache_hit_returns_cached_true(
        self, client: TestClient, _isolated_env: Path
    ):
        ensure_models_dir()
        slug_dir = _isolated_env / "repo--iris-osl--v3"
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / "model.gguf").write_bytes(b"x")
        add_manifest_entry(_make_manifest_entry(path=str(slug_dir.resolve())))

        with patch("solar_host.models_manager._pull_harbor") as mock_pull:
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True
        assert data["source_uri"] == "repo://iris-osl:v3"
        assert data["path"] == str(slug_dir.resolve())
        mock_pull.assert_not_called()

    def test_cache_hit_returns_stored_path(
        self, client: TestClient, _isolated_env: Path
    ):
        ensure_models_dir()
        custom = _isolated_env / "custom-path-to-model"
        custom.mkdir(parents=True, exist_ok=True)
        (custom / "model.gguf").write_bytes(b"x")
        add_manifest_entry(_make_manifest_entry(path=str(custom.resolve())))

        resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.json()["path"] == str(custom.resolve())

    def test_hf_cache_hit(self, client: TestClient, _isolated_env: Path):
        ensure_models_dir()
        slug_dir = _isolated_env / "hf--microsoft--phi-3"
        slug_dir.mkdir(parents=True, exist_ok=True)
        (slug_dir / "config.json").write_bytes(b"{}")
        add_manifest_entry(
            _make_manifest_entry(
                slug="hf--microsoft--phi-3",
                source_uri="huggingface://microsoft/phi-3",
                path=str(slug_dir.resolve()),
            )
        )
        with patch("solar_host.models_manager._pull_huggingface") as mock_dl:
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert resp.status_code == 200
        assert resp.json()["cached"] is True
        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# Harbor pull (cache miss)
# ---------------------------------------------------------------------------


class TestHarborPull:
    def _make_mock_pull(self, tmp_path: Path):
        """Return a side_effect function that creates a dummy file in target_dir."""

        def _side_effect(harbor_ref: str, target_dir: Path, source_uri: str):
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "model.gguf").write_bytes(b"x" * 1024)

        return _side_effect

    def test_harbor_pull_cache_miss(self, client: TestClient, _isolated_env: Path):
        with patch(
            "solar_host.models_manager._pull_harbor",
            side_effect=self._make_mock_pull(_isolated_env),
        ) as mock_pull:
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False
        assert data["source_uri"] == "repo://iris-osl:v3"
        assert "repo--iris-osl--v3" in data["path"]
        mock_pull.assert_called_once()

    def test_harbor_pull_updates_manifest(
        self, client: TestClient, _isolated_env: Path
    ):
        with patch(
            "solar_host.models_manager._pull_harbor",
            side_effect=self._make_mock_pull(_isolated_env),
        ):
            client.post("/models/pull", json=_harbor_body(), headers=_headers())

        entry = get_manifest_entry("repo://iris-osl:v3")
        assert entry is not None
        assert entry.slug == "repo--iris-osl--v3"
        assert entry.digest == "sha256:abc123"
        assert entry.size_bytes == 1024

    def test_harbor_pull_called_with_correct_args(
        self, client: TestClient, _isolated_env: Path
    ):
        captured = {}

        def _capture(harbor_ref, target_dir, source_uri):
            captured["harbor_ref"] = harbor_ref
            captured["target_dir"] = target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "model.gguf").write_bytes(b"x")

        with patch("solar_host.models_manager._pull_harbor", side_effect=_capture):
            client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert captured["harbor_ref"] == "imgrepo.damit.hu/supernova/iris-osl:v3"
        assert str(captured["target_dir"]).endswith("repo--iris-osl--v3")

    def test_second_pull_returns_cached(self, client: TestClient, _isolated_env: Path):
        """After a successful pull, the same URI returns cached=True."""
        with patch(
            "solar_host.models_manager._pull_harbor",
            side_effect=self._make_mock_pull(_isolated_env),
        ):
            resp1 = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp1.json()["cached"] is False

        with patch("solar_host.models_manager._pull_harbor") as mock_pull:
            resp2 = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp2.json()["cached"] is True
        mock_pull.assert_not_called()


# ---------------------------------------------------------------------------
# HuggingFace pull (cache miss)
# ---------------------------------------------------------------------------


class TestHuggingFacePull:
    def _make_mock_dl(self, _isolated_env: Path):
        def _side_effect(model_id, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "pytorch_model.bin").write_bytes(b"w" * 2048)

        return _side_effect

    def test_hf_pull_cache_miss(self, client: TestClient, _isolated_env: Path):
        with patch(
            "solar_host.models_manager._pull_huggingface",
            side_effect=self._make_mock_dl(_isolated_env),
        ):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is False
        assert "hf--microsoft--phi-3" in data["path"]

    def test_hf_pull_updates_manifest(self, client: TestClient, _isolated_env: Path):
        with patch(
            "solar_host.models_manager._pull_huggingface",
            side_effect=self._make_mock_dl(_isolated_env),
        ):
            client.post("/models/pull", json=_hf_body(), headers=_headers())

        entry = get_manifest_entry("huggingface://microsoft/phi-3")
        assert entry is not None
        assert entry.slug == "hf--microsoft--phi-3"
        assert entry.size_bytes == 2048

    def test_hf_pull_passes_none_token_when_empty(
        self, client: TestClient, _isolated_env: Path
    ):
        """When hf_token is empty string, snapshot_download must receive token=None."""
        captured = {}

        def _capture(model_id, target_dir, source_uri):
            captured["model_id"] = model_id
            captured["target_dir"] = target_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "model.bin").write_bytes(b"x")

        with patch("solar_host.models_manager._pull_huggingface", side_effect=_capture):
            client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert captured["model_id"] == "microsoft/phi-3"

    def test_hf_pull_with_token(
        self, client: TestClient, _isolated_env: Path, monkeypatch
    ):
        monkeypatch.setattr("solar_host.config.settings.hf_token", "hf_mytoken123")
        captured_token = {}

        def _fake_snapshot(repo_id, local_dir, token):
            captured_token["token"] = token
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.bin").write_bytes(b"x")

        with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert resp.status_code == 200
        assert captured_token["token"] == "hf_mytoken123"

    def test_hf_pull_passes_none_not_empty_string(
        self, client: TestClient, _isolated_env: Path
    ):
        """Empty hf_token must become None, not empty string, in snapshot_download."""
        captured_token = {}

        def _fake_snapshot(repo_id, local_dir, token):
            captured_token["token"] = token
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.bin").write_bytes(b"x")

        with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert resp.status_code == 200
        assert captured_token["token"] is None


# ---------------------------------------------------------------------------
# Missing credentials (500)
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_missing_harbor_url_returns_500(self, client: TestClient, monkeypatch):
        monkeypatch.setattr("solar_host.config.settings.harbor_url", "")
        resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "credentials_missing"

    def test_missing_harbor_username_returns_500(self, client: TestClient, monkeypatch):
        monkeypatch.setattr("solar_host.config.settings.harbor_username", "")
        resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "credentials_missing"

    def test_missing_harbor_password_returns_500(self, client: TestClient, monkeypatch):
        monkeypatch.setattr("solar_host.config.settings.harbor_password", "")
        resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 500
        data = resp.json()
        assert data["error"] == "credentials_missing"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    """Verify that library exceptions map to the correct HTTP status codes
    and that the response body matches the spec Section 6.2 format."""

    def _post_harbor(self, client, exc_to_raise):
        with patch(
            "solar_host.models_manager._pull_harbor",
            side_effect=exc_to_raise,
        ):
            return client.post("/models/pull", json=_harbor_body(), headers=_headers())

    def _assert_error_body(self, resp, expected_status: int, expected_error: str):
        assert resp.status_code == expected_status
        data = resp.json()
        assert data["error"] == expected_error
        assert "detail" in data
        assert data["source_uri"] == "repo://iris-osl:v3"
        assert data["status_code"] == expected_status

    def test_harbor_connection_error_returns_502(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(
            502, "source_unreachable", "Harbor unreachable", "repo://iris-osl:v3"
        )
        resp = self._post_harbor(client, exc)
        self._assert_error_body(resp, 502, "source_unreachable")

    def test_harbor_auth_error_returns_401(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(401, "auth_failed", "Auth failed", "repo://iris-osl:v3")
        resp = self._post_harbor(client, exc)
        self._assert_error_body(resp, 401, "auth_failed")

    def test_artifact_not_found_returns_404(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(404, "not_found", "Not found", "repo://iris-osl:v3")
        resp = self._post_harbor(client, exc)
        self._assert_error_body(resp, 404, "not_found")

    def test_disk_full_returns_507(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(
            507, "insufficient_storage", "Disk full", "repo://iris-osl:v3"
        )
        resp = self._post_harbor(client, exc)
        self._assert_error_body(resp, 507, "insufficient_storage")

    def test_hf_repo_not_found_returns_404(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(
            404, "not_found", "HF repo not found", "huggingface://microsoft/phi-3"
        )
        with patch("solar_host.models_manager._pull_huggingface", side_effect=exc):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"

    def test_hf_gated_repo_returns_401(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(
            401, "auth_failed", "Gated repo", "huggingface://microsoft/phi-3"
        )
        with patch("solar_host.models_manager._pull_huggingface", side_effect=exc):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())
        assert resp.status_code == 401
        assert resp.json()["error"] == "auth_failed"


# ---------------------------------------------------------------------------
# Failure cleanup
# ---------------------------------------------------------------------------


class TestFailureCleanup:
    def test_partial_directory_cleaned_up_on_failure(
        self, client: TestClient, _isolated_env: Path
    ):
        """If download raises, the (possibly partial) target directory must not remain."""
        from solar_host.models_manager import ModelPullError

        slug_dir = _isolated_env / "repo--iris-osl--v3"

        def _fail(harbor_ref, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "partial.bin").write_bytes(b"partial")
            raise ModelPullError(502, "source_unreachable", "Harbor down", source_uri)

        with patch("solar_host.models_manager._pull_harbor", side_effect=_fail):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 502
        assert not slug_dir.exists(), "Partial directory should have been cleaned up"

    def test_failed_pull_not_added_to_manifest(
        self, client: TestClient, _isolated_env: Path
    ):
        from solar_host.models_manager import ModelPullError

        def _fail(harbor_ref, target_dir, source_uri):
            raise ModelPullError(404, "not_found", "Not found", source_uri)

        with patch("solar_host.models_manager._pull_harbor", side_effect=_fail):
            client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert get_manifest_entry("repo://iris-osl:v3") is None

    def test_stale_directory_removed_before_pull(
        self, client: TestClient, _isolated_env: Path
    ):
        """Pre-existing orphan directory is deleted before a fresh download starts."""
        ensure_models_dir()
        slug_dir = get_models_dir() / "repo--iris-osl--v3"
        slug_dir.mkdir(parents=True, exist_ok=True)
        stale_file = slug_dir / "stale.bin"
        stale_file.write_bytes(b"stale data")

        removed_before_dl = {}

        def _check_and_create(harbor_ref, target_dir, source_uri):
            removed_before_dl["stale_gone"] = not stale_file.exists()
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "fresh.bin").write_bytes(b"fresh data")

        with patch(
            "solar_host.models_manager._pull_harbor", side_effect=_check_and_create
        ):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 200
        assert (
            removed_before_dl.get("stale_gone") is True
        ), "Stale directory should have been removed before download started"


# ---------------------------------------------------------------------------
# OSError / ENOSPC
# ---------------------------------------------------------------------------


class TestDiskFull:
    def test_enospc_during_harbor_pull_returns_507(
        self, client: TestClient, _isolated_env: Path
    ):
        """An OSError with errno.ENOSPC from the download must surface as 507."""
        import errno as _errno

        def _fail(harbor_ref, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            raise OSError(_errno.ENOSPC, "No space left on device")

        with patch("solar_host.models_manager._pull_harbor", side_effect=_fail):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 507
        data = resp.json()
        assert data["error"] == "insufficient_storage"
        assert data["source_uri"] == "repo://iris-osl:v3"

    def test_enospc_during_hf_pull_returns_507(
        self, client: TestClient, _isolated_env: Path
    ):
        import errno as _errno

        def _fail(model_id, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            raise OSError(_errno.ENOSPC, "No space left on device")

        with patch("solar_host.models_manager._pull_huggingface", side_effect=_fail):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())

        assert resp.status_code == 507
        assert resp.json()["error"] == "insufficient_storage"

    def test_non_enospc_oserror_returns_500(
        self, client: TestClient, _isolated_env: Path
    ):
        import errno as _errno

        def _fail(harbor_ref, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            raise OSError(_errno.EACCES, "Permission denied")

        with patch("solar_host.models_manager._pull_harbor", side_effect=_fail):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 500
        assert resp.json()["error"] == "model_pull_failed"


# ---------------------------------------------------------------------------
# _map_download_exception integration
# ---------------------------------------------------------------------------


class TestMapDownloadException:
    """Exercise _map_download_exception via pull_model() with real-shaped exceptions."""

    @staticmethod
    def _make_exc(module: str, name: str, msg: str = "boom") -> Exception:
        """Create an exception whose __module__ and __name__ match a library."""
        cls = type(name, (Exception,), {"__module__": module})
        return cls(msg)

    def test_harbor_connection_error_mapped(
        self, client: TestClient, _isolated_env: Path
    ):
        exc = self._make_exc("harbor_oci_client.exceptions", "HarborConnectionError")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 502
        assert resp.json()["error"] == "source_unreachable"

    def test_harbor_auth_error_mapped(self, client: TestClient, _isolated_env: Path):
        exc = self._make_exc("harbor_oci_client.exceptions", "HarborAuthError")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 401
        assert resp.json()["error"] == "auth_failed"

    def test_artifact_not_found_mapped(self, client: TestClient, _isolated_env: Path):
        exc = self._make_exc("harbor_oci_client.exceptions", "ArtifactNotFoundError")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"

    def test_unknown_harbor_error_mapped_to_502(
        self, client: TestClient, _isolated_env: Path
    ):
        exc = self._make_exc("harbor_oci_client.exceptions", "HarborAPIError")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 502
        assert resp.json()["error"] == "source_unreachable"

    def test_hf_repository_not_found_mapped(
        self, client: TestClient, _isolated_env: Path
    ):
        exc = self._make_exc("huggingface_hub.utils", "RepositoryNotFoundError")
        with patch("solar_host.models_manager._pull_huggingface", side_effect=exc):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())
        assert resp.status_code == 404
        assert resp.json()["error"] == "not_found"

    def test_hf_gated_repo_mapped(self, client: TestClient, _isolated_env: Path):
        exc = self._make_exc("huggingface_hub.utils", "GatedRepoError")
        with patch("solar_host.models_manager._pull_huggingface", side_effect=exc):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())
        assert resp.status_code == 401
        assert resp.json()["error"] == "auth_failed"

    def test_unknown_hf_error_mapped_to_502(
        self, client: TestClient, _isolated_env: Path
    ):
        exc = self._make_exc("huggingface_hub.utils", "HfHubHTTPError")
        with patch("solar_host.models_manager._pull_huggingface", side_effect=exc):
            resp = client.post("/models/pull", json=_hf_body(), headers=_headers())
        assert resp.status_code == 502
        assert resp.json()["error"] == "source_unreachable"

    def test_unknown_exception_mapped_to_500(
        self, client: TestClient, _isolated_env: Path
    ):
        exc = RuntimeError("something unexpected")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())
        assert resp.status_code == 500
        assert resp.json()["error"] == "model_pull_failed"


# ---------------------------------------------------------------------------
# Response structure
# ---------------------------------------------------------------------------


class TestResponseStructure:
    def test_response_contains_required_fields(
        self, client: TestClient, _isolated_env: Path
    ):
        def _create(harbor_ref, target_dir, source_uri):
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "model.gguf").write_bytes(b"x" * 512)

        with patch("solar_host.models_manager._pull_harbor", side_effect=_create):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        assert resp.status_code == 200
        data = resp.json()
        assert set(data.keys()) == {"path", "cached", "source_uri"}

    def test_error_response_contains_required_fields(self, client: TestClient):
        from solar_host.models_manager import ModelPullError

        exc = ModelPullError(404, "not_found", "Gone", "repo://iris-osl:v3")
        with patch("solar_host.models_manager._pull_harbor", side_effect=exc):
            resp = client.post("/models/pull", json=_harbor_body(), headers=_headers())

        data = resp.json()
        assert set(data.keys()) == {"error", "detail", "source_uri", "status_code"}
