"""Tests for app.models_manager — slug derivation, manifest CRUD, directory helpers."""

import json
from pathlib import Path

import pytest

from solar_host.models_manager import (
    MANIFEST_FILENAME,
    MANIFEST_TMP_FILENAME,
    Manifest,
    ManifestEntry,
    add_manifest_entry,
    ensure_models_dir,
    get_manifest_entry,
    get_models_dir,
    read_manifest,
    remove_manifest_entry,
    source_uri_to_slug,
    write_manifest,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path: Path, monkeypatch):
    """Point settings.models_dir at a temporary directory for every test."""
    models = tmp_path / "models"
    monkeypatch.setattr("solar_host.models_manager.settings.models_dir", str(models))
    return models


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
# source_uri_to_slug
# ---------------------------------------------------------------------------


class TestSourceUriToSlug:
    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("repo://iris-osl:v3", "repo--iris-osl--v3"),
            ("repo://iris-tickets:2026-03", "repo--iris-tickets--2026-03"),
            ("repo://IRIS-BERT-base:v1", "repo--IRIS-BERT-base--v1"),
        ],
    )
    def test_repo_scheme(self, uri: str, expected: str):
        assert source_uri_to_slug(uri) == expected

    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("huggingface://microsoft/phi-3", "hf--microsoft--phi-3"),
            ("huggingface://meta-llama/Llama-2-7b-hf", "hf--meta-llama--Llama-2-7b-hf"),
            ("huggingface://phi-3", "hf--phi-3"),
        ],
    )
    def test_huggingface_scheme(self, uri: str, expected: str):
        assert source_uri_to_slug(uri) == expected

    def test_local_raises(self):
        with pytest.raises(ValueError, match="local://"):
            source_uri_to_slug("local:///opt/models/foo.gguf")

    def test_unknown_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            source_uri_to_slug("s3://bucket/model")

    def test_malformed_repo_no_version_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            source_uri_to_slug("repo://iris-osl")


# ---------------------------------------------------------------------------
# get_models_dir / ensure_models_dir
# ---------------------------------------------------------------------------


class TestModelsDir:
    def test_get_models_dir_returns_absolute(self):
        result = get_models_dir()
        assert result.is_absolute()

    def test_ensure_creates_directory(self, tmp_path: Path):
        target = tmp_path / "models"
        assert not target.exists()
        ensure_models_dir()
        assert target.is_dir()

    def test_ensure_creates_manifest(self, tmp_path: Path):
        ensure_models_dir()
        manifest_path = get_models_dir() / MANIFEST_FILENAME
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data == {"models": []}

    def test_ensure_does_not_overwrite_existing_manifest(self, tmp_path: Path):
        ensure_models_dir()
        entry = _make_entry()
        add_manifest_entry(entry)
        ensure_models_dir()
        manifest = read_manifest()
        assert len(manifest.models) == 1

    def test_ensure_idempotent(self, tmp_path: Path):
        ensure_models_dir()
        ensure_models_dir()
        assert (tmp_path / "models").is_dir()


# ---------------------------------------------------------------------------
# read_manifest / write_manifest
# ---------------------------------------------------------------------------


class TestManifestIO:
    def test_read_returns_empty_when_file_missing(self, tmp_path: Path):
        (tmp_path / "models").mkdir()
        manifest = read_manifest()
        assert manifest.models == []

    def test_read_returns_empty_after_ensure(self):
        ensure_models_dir()
        manifest = read_manifest()
        assert manifest.models == []

    def test_roundtrip(self):
        ensure_models_dir()
        entry = _make_entry()
        original = Manifest(models=[entry])
        write_manifest(original)
        loaded = read_manifest()
        assert len(loaded.models) == 1
        assert loaded.models[0].source_uri == entry.source_uri
        assert loaded.models[0].size_bytes == entry.size_bytes

    def test_atomic_write_no_tmp_left(self):
        ensure_models_dir()
        write_manifest(Manifest())
        tmp = get_models_dir() / MANIFEST_TMP_FILENAME
        assert not tmp.exists()

    def test_manifest_file_is_valid_json(self):
        ensure_models_dir()
        write_manifest(Manifest(models=[_make_entry()]))
        raw = (get_models_dir() / MANIFEST_FILENAME).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "models" in data
        assert len(data["models"]) == 1

    def test_read_returns_empty_on_corrupt_file(self):
        ensure_models_dir()
        (get_models_dir() / MANIFEST_FILENAME).write_text("NOT JSON", encoding="utf-8")
        manifest = read_manifest()
        assert manifest.models == []


# ---------------------------------------------------------------------------
# get / add / remove manifest entry
# ---------------------------------------------------------------------------


class TestManifestCRUD:
    def test_get_returns_none_when_empty(self):
        ensure_models_dir()
        assert get_manifest_entry("repo://iris-osl:v3") is None

    def test_add_then_get(self):
        ensure_models_dir()
        entry = _make_entry()
        add_manifest_entry(entry)
        found = get_manifest_entry("repo://iris-osl:v3")
        assert found is not None
        assert found.slug == "repo--iris-osl--v3"

    def test_add_upserts_existing(self):
        ensure_models_dir()
        add_manifest_entry(_make_entry(size_bytes=100))
        add_manifest_entry(_make_entry(size_bytes=200))
        manifest = read_manifest()
        assert len(manifest.models) == 1
        assert manifest.models[0].size_bytes == 200

    def test_add_multiple_entries(self):
        ensure_models_dir()
        add_manifest_entry(_make_entry(source_uri="repo://a:v1", slug="repo--a--v1"))
        add_manifest_entry(_make_entry(source_uri="repo://b:v2", slug="repo--b--v2"))
        manifest = read_manifest()
        assert len(manifest.models) == 2

    def test_remove_existing(self):
        ensure_models_dir()
        add_manifest_entry(_make_entry())
        removed = remove_manifest_entry("repo://iris-osl:v3")
        assert removed is True
        assert get_manifest_entry("repo://iris-osl:v3") is None

    def test_remove_nonexistent_returns_false(self):
        ensure_models_dir()
        assert remove_manifest_entry("repo://nope:v1") is False

    def test_remove_preserves_other_entries(self):
        ensure_models_dir()
        add_manifest_entry(_make_entry(source_uri="repo://a:v1", slug="repo--a--v1"))
        add_manifest_entry(_make_entry(source_uri="repo://b:v2", slug="repo--b--v2"))
        remove_manifest_entry("repo://a:v1")
        manifest = read_manifest()
        assert len(manifest.models) == 1
        assert manifest.models[0].source_uri == "repo://b:v2"

    def test_digest_is_optional(self):
        ensure_models_dir()
        entry = _make_entry(digest=None)
        add_manifest_entry(entry)
        found = get_manifest_entry(entry.source_uri)
        assert found is not None
        assert found.digest is None
