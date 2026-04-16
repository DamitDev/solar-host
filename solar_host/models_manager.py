"""Managed models directory and manifest for solar-host.

Provides slug derivation from model source URIs, atomic manifest read/write,
CRUD helpers for tracking downloaded models, and the pull_model() orchestration
function for downloading from Harbor (ORAS) or HuggingFace Hub.

The manifest file (MODELS_DIR/manifest.json) is the single source of truth for
cache detection.
"""

import errno
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

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
    """Create the models directory and initialize an empty manifest if needed."""
    get_models_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path()
    if not manifest_path.exists():
        write_manifest(Manifest())
        logger.info("Initialized manifest at %s", manifest_path)


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


def get_manifest_entry_by_slug(slug: str) -> Optional[ManifestEntry]:
    """Look up a manifest entry by slug. Returns None if not found."""
    manifest = read_manifest()
    for entry in manifest.models:
        if entry.slug == slug:
            return entry
    return None


def delete_model_files(path: str) -> None:
    """Remove model files from disk.

    Handles both directory and single-file models. Silently succeeds if the
    path no longer exists (e.g. already manually removed).
    """
    model_path = Path(path)
    try:
        if model_path.is_dir():
            shutil.rmtree(model_path, ignore_errors=False)
        elif model_path.exists():
            model_path.unlink()
        # If it does not exist at all, nothing to do.
    except FileNotFoundError:
        logger.warning("Model path already gone during deletion: %s", path)
    except OSError as exc:
        logger.error("Failed to delete model files at %s: %s", path, exc)
        raise


# ---------------------------------------------------------------------------
# Pull orchestration
# ---------------------------------------------------------------------------

# Protects manifest read-modify-write from concurrent pulls in different threads.
_manifest_lock = threading.Lock()


def remove_manifest_entry_by_slug(slug: str) -> Optional[ManifestEntry]:
    """Remove a manifest entry by slug under the manifest lock.

    Returns the removed entry (so the caller can obtain the path to delete
    from disk), or None if no entry matched.
    """
    with _manifest_lock:
        manifest = read_manifest()
        removed: Optional[ManifestEntry] = None
        new_models = []
        for entry in manifest.models:
            if entry.slug == slug and removed is None:
                removed = entry
            else:
                new_models.append(entry)
        if removed is None:
            return None
        manifest.models = new_models
        write_manifest(manifest)
        return removed


# Per-URI locks serialise the full pull lifecycle (cache check → download →
# manifest write) so two concurrent requests for the *same* source_uri cannot
# both miss the cache and download the model twice.
_uri_locks: dict[str, threading.Lock] = {}
_uri_locks_guard = threading.Lock()


def _get_uri_lock(source_uri: str) -> threading.Lock:
    """Return a per-URI lock, creating one if it doesn't exist yet."""
    with _uri_locks_guard:
        if source_uri not in _uri_locks:
            _uri_locks[source_uri] = threading.Lock()
        return _uri_locks[source_uri]


_SOURCE_URI_PREFIXES = {
    "harbor": "repo://",
    "huggingface": "huggingface://",
}


class ModelPullError(Exception):
    """Raised by pull_model() for expected failure conditions.

    Carries enough context for the route handler to build a spec-compliant
    error response without leaking internal details.
    """

    def __init__(self, status_code: int, error: str, detail: str, source_uri: str):
        self.status_code = status_code
        self.error = error
        self.detail = detail
        self.source_uri = source_uri
        super().__init__(detail)


def _compute_dir_size(path: Path) -> int:
    """Return total size in bytes of regular files under *path* (no symlinks)."""
    if not path.is_dir():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if not entry.is_symlink() and entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _pull_harbor(
    harbor_ref: str,
    target_dir: Path,
    source_uri: str,
) -> None:
    """Download a Harbor OCI artifact via ORAS into *target_dir*.

    Credentials must have been validated by the caller before this is invoked.
    """
    from harbor_oci_client import OrasHelper  # type: ignore[import-untyped]

    parsed = urlparse(settings.harbor_url)
    hostname = parsed.hostname or settings.harbor_url

    oras = OrasHelper(
        hostname=hostname,
        username=settings.harbor_username,
        password=settings.harbor_password,
    )
    oras.pull(harbor_ref, outdir=str(target_dir))


def _pull_huggingface(
    model_id: str,
    target_dir: Path,
    source_uri: str,
) -> None:
    """Download a HuggingFace Hub model snapshot into *target_dir*."""
    import huggingface_hub  # type: ignore[import-untyped]

    hf_token: Optional[str] = settings.hf_token or None

    huggingface_hub.snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        token=hf_token,
    )


def pull_model(
    *,
    source: str,
    source_uri: str,
    harbor_ref: Optional[str] = None,
    model_id: Optional[str] = None,
    digest: Optional[str] = None,
) -> dict:
    """Download a model from Harbor or HuggingFace Hub and record it in the manifest.

    This function is synchronous and intended to be called via
    ``asyncio.to_thread()`` from the async route handler.

    Returns a dict with keys ``path``, ``cached``, and ``source_uri``.
    Raises ``ModelPullError`` for all expected failure conditions.
    """
    # 1. Validate that source_uri scheme matches the declared source.
    expected_prefix = _SOURCE_URI_PREFIXES.get(source)
    if expected_prefix is None:
        raise ModelPullError(
            400, "invalid_request", f"Unsupported source: {source!r}", source_uri
        )
    if not source_uri.startswith(expected_prefix):
        raise ModelPullError(
            400,
            "invalid_request",
            f"source_uri {source_uri!r} does not match source type {source!r} "
            f"(expected prefix {expected_prefix!r})",
            source_uri,
        )

    # Acquire a per-URI lock so that concurrent pulls for the *same* source_uri
    # are serialised end-to-end.  The second caller will block here, then see
    # the cache hit once the first caller finishes.
    uri_lock = _get_uri_lock(source_uri)
    with uri_lock:
        # 2. Cache check — manifest is the single source of truth.
        #    Verify the files still exist on disk; remove stale entries.
        cached_entry = get_manifest_entry(source_uri)
        if cached_entry is not None:
            if Path(cached_entry.path).exists():
                return {
                    "path": cached_entry.path,
                    "cached": True,
                    "source_uri": source_uri,
                }
            logger.warning(
                "Manifest entry for %s points to missing path %s, re-pulling",
                source_uri,
                cached_entry.path,
            )
            with _manifest_lock:
                remove_manifest_entry(source_uri)

        # 3. Derive slug and target directory.
        try:
            slug = source_uri_to_slug(source_uri)
        except ValueError as exc:
            raise ModelPullError(400, "invalid_request", str(exc), source_uri) from exc

        target_dir = get_models_dir() / slug

        # 4. Validate credentials before touching the filesystem.
        if source == "harbor":
            if not all(
                [
                    settings.harbor_url,
                    settings.harbor_username,
                    settings.harbor_password,
                ]
            ):
                raise ModelPullError(
                    500,
                    "credentials_missing",
                    "Harbor credentials not configured. Set HARBOR_URL, HARBOR_USERNAME, and HARBOR_PASSWORD.",
                    source_uri,
                )

        # 5. Remove any stale/partial directory from a previous failed pull.
        if target_dir.exists():
            logger.warning("Removing stale model directory before pull: %s", target_dir)
            shutil.rmtree(target_dir, ignore_errors=True)

        # 6. Download — wrap in try/except to clean up on failure.
        try:
            if source == "harbor":
                _pull_harbor(harbor_ref or "", target_dir, source_uri)
            else:
                _pull_huggingface(model_id or "", target_dir, source_uri)
        except ModelPullError:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            if exc.errno == errno.ENOSPC:
                raise ModelPullError(
                    507, "insufficient_storage", "Insufficient disk space.", source_uri
                ) from exc
            raise ModelPullError(
                500, "model_pull_failed", str(exc), source_uri
            ) from exc
        except Exception as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            _map_download_exception(exc, source_uri)

        # 7. Compute size of downloaded files.
        size_bytes = _compute_dir_size(target_dir)

        # 8. Update manifest atomically under lock to prevent concurrent write
        #    races between pulls for *different* URIs finishing simultaneously.
        entry = ManifestEntry(
            slug=slug,
            source_uri=source_uri,
            path=str(target_dir.resolve()),
            size_bytes=size_bytes,
            digest=digest,
            downloaded_at=datetime.now(timezone.utc).isoformat(),
        )
        with _manifest_lock:
            add_manifest_entry(entry)

        logger.info("Model pulled successfully: %s -> %s", source_uri, target_dir)
        return {
            "path": str(target_dir.resolve()),
            "cached": False,
            "source_uri": source_uri,
        }


def _map_download_exception(exc: Exception, source_uri: str) -> None:
    """Re-raise a library exception as a ModelPullError with an appropriate HTTP status.

    Always raises — never returns.
    """
    exc_type = type(exc).__name__
    module = type(exc).__module__ or ""

    # harbor-oci-client exceptions
    if module.startswith("harbor_oci_client"):
        if exc_type == "HarborConnectionError":
            raise ModelPullError(
                502,
                "source_unreachable",
                f"Harbor registry unreachable: {exc}",
                source_uri,
            ) from exc
        if exc_type == "HarborAuthError":
            raise ModelPullError(
                401, "auth_failed", f"Harbor authentication failed: {exc}", source_uri
            ) from exc
        if exc_type == "ArtifactNotFoundError":
            raise ModelPullError(
                404, "not_found", f"Artifact not found in Harbor: {exc}", source_uri
            ) from exc
        raise ModelPullError(
            502, "source_unreachable", f"Harbor error: {exc}", source_uri
        ) from exc

    # huggingface_hub exceptions
    if module.startswith("huggingface_hub"):
        if exc_type == "RepositoryNotFoundError":
            raise ModelPullError(
                404, "not_found", f"HuggingFace repository not found: {exc}", source_uri
            ) from exc
        if exc_type == "GatedRepoError":
            raise ModelPullError(
                401,
                "auth_failed",
                f"HuggingFace repository is gated: {exc}",
                source_uri,
            ) from exc
        raise ModelPullError(
            502, "source_unreachable", f"HuggingFace Hub error: {exc}", source_uri
        ) from exc

    # Fallback
    raise ModelPullError(500, "model_pull_failed", str(exc), source_uri) from exc
