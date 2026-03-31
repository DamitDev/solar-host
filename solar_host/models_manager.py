"""Managed models directory and manifest for solar-host.

Provides slug derivation from model source URIs, atomic manifest read/write,
and CRUD helpers for tracking downloaded models. The manifest file
(MODELS_DIR/manifest.json) is the single source of truth for cache detection.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from solar_host.config import settings

logger = logging.getLogger(__name__)

_REPO_PATTERN = re.compile(r"^repo://([A-Za-z0-9\-_]+):([A-Za-z0-9\-_.]+)$")
_HF_PATTERN = re.compile(r"^huggingface://(.+)$")

MANIFEST_FILENAME = "manifest.json"
MANIFEST_TMP_FILENAME = "manifest.json.tmp"


class ManifestEntry(BaseModel):
    """A single downloaded model tracked in the manifest."""

    slug: str
    source_uri: str
    path: str
    size_bytes: int
    digest: Optional[str] = None
    downloaded_at: str


class Manifest(BaseModel):
    """Root manifest object stored at MODELS_DIR/manifest.json."""

    models: list[ManifestEntry] = []


def get_models_dir() -> Path:
    """Return the resolved absolute path to the models directory."""
    return Path(settings.models_dir).resolve()


def ensure_models_dir() -> None:
    """Create the models directory if it does not exist."""
    get_models_dir().mkdir(parents=True, exist_ok=True)


def source_uri_to_slug(uri: str) -> str:
    """Derive a deterministic directory slug from a model source URI.

    Raises ValueError for local:// URIs (not stored in MODELS_DIR)
    and for unrecognised or malformed URIs.
    """
    if uri.startswith("local://"):
        raise ValueError("local:// URIs are not stored in the models directory")

    m = _REPO_PATTERN.match(uri)
    if m:
        name, version = m.group(1), m.group(2)
        return f"repo--{name}--{version}"

    m = _HF_PATTERN.match(uri)
    if m:
        model_id = m.group(1)
        return f"hf--{model_id.replace('/', '--')}"

    raise ValueError(f"Unsupported or malformed model source URI: {uri}")


def _manifest_path() -> Path:
    return get_models_dir() / MANIFEST_FILENAME


def read_manifest() -> Manifest:
    """Read and parse the manifest file.

    Returns an empty Manifest when the file is missing or cannot be parsed.
    """
    path = _manifest_path()
    if not path.exists():
        return Manifest()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(data)
    except Exception:
        logger.warning("Failed to parse manifest at %s, treating as empty", path)
        return Manifest()


def write_manifest(manifest: Manifest) -> None:
    """Atomically write the manifest to disk (write tmp, then rename)."""
    target = _manifest_path()
    tmp = target.parent / MANIFEST_TMP_FILENAME
    tmp.write_text(
        json.dumps(manifest.model_dump(), indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, target)


def get_manifest_entry(source_uri: str) -> Optional[ManifestEntry]:
    """Look up a manifest entry by source_uri. Returns None if not found."""
    manifest = read_manifest()
    for entry in manifest.models:
        if entry.source_uri == source_uri:
            return entry
    return None


def add_manifest_entry(entry: ManifestEntry) -> None:
    """Add or update (upsert) a manifest entry, matched by source_uri."""
    manifest = read_manifest()
    manifest.models = [e for e in manifest.models if e.source_uri != entry.source_uri]
    manifest.models.append(entry)
    write_manifest(manifest)


def remove_manifest_entry(source_uri: str) -> bool:
    """Remove a manifest entry by source_uri. Returns True if an entry was removed."""
    manifest = read_manifest()
    before = len(manifest.models)
    manifest.models = [e for e in manifest.models if e.source_uri != source_uri]
    if len(manifest.models) == before:
        return False
    write_manifest(manifest)
    return True
