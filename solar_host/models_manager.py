"""Managed models directory and manifest for solar-host.

Provides slug derivation from model source URIs, atomic manifest read/write,
CRUD helpers for tracking downloaded models, and the pull_model() orchestration
function for downloading from Harbor (ORAS) or HuggingFace Hub.

The manifest file (MODELS_DIR/manifest.json) is the single source of truth for
cache detection.
"""

import errno
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pebble
from pydantic import BaseModel

from solar_host.config import settings
from solar_host.memory_monitor import get_disk_info

logger = logging.getLogger(__name__)

_REPO_PATTERN = re.compile(r"^repo://([A-Za-z0-9\-_]+):([A-Za-z0-9\-_.]+)(/.*)?$")
_HF_PATTERN = re.compile(r"^huggingface://(.+)$")

MANIFEST_FILENAME = "manifest.json"
MANIFEST_TMP_FILENAME = "manifest.json.tmp"


class ManifestEntry(BaseModel):
    """A single downloaded model tracked in the manifest.

    The ``category``, ``name``, ``version``, ``checksum`` and ``metadata``
    fields are optional and were introduced with D-016 to carry through
    authoritative metadata supplied by Data Repository. They are absent on
    entries created before that change; readers must treat them as optional.
    """

    slug: str
    source_uri: str
    path: str
    size_bytes: int
    digest: str | None = None
    downloaded_at: str
    category: str | None = None
    name: str | None = None
    version: str | None = None
    checksum: str | None = None
    metadata: dict | None = None
    # Per-file sha256 (hex) of the pulled artifact, recorded at pull time
    # and verified on every cache hit (D-017). Absent on entries created
    # before this field existed — readers must treat it as optional.
    file_digests: dict | None = None


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

    For ``repo://name:version/subpath`` the subpath is ignored — the slug
    is always ``repo--name--version`` (the subpath selects a file inside
    the pulled artifact directory).

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


def extract_repo_subpath(uri: str) -> str:
    """Return the subpath of a ``repo://name:version/subpath`` URI.

    Returns the subpath without the leading slash (e.g. ``model.gguf``
    for ``repo://iris-osl:v3/model.gguf``), or ``""`` when the URI has
    no subpath or is not a ``repo://`` URI.
    """
    if not uri.startswith("repo://"):
        return ""
    m = _REPO_PATTERN.match(uri)
    if m and m.group(3):
        return m.group(3).lstrip("/")
    return ""


def repo_base_uri(uri: str) -> str:
    """Strip the subpath from a ``repo://name:version/subpath`` URI.

    Returns ``repo://name:version`` (the artifact identity used as the
    manifest cache key).  Non-repo URIs are returned unchanged.
    """
    if not uri.startswith("repo://"):
        return uri
    m = _REPO_PATTERN.match(uri)
    if m:
        return f"repo://{m.group(1)}:{m.group(2)}"
    return uri


def _select_gguf_path(model_dir: Path) -> Path | None:
    """Return the largest ``*.gguf`` at the root of *model_dir*, or None.

    Used for llama.cpp + ``repo://`` artifacts: llama-server needs a file,
    and when the artifact carries multiple GGUFs (e.g. quantised variants)
    the largest one is the definitive model.  Subdirectories are not
    scanned — ORAS pulls are flat and an explicit subpath already exists
    for nested files.
    """
    candidates = [
        p for p in model_dir.glob("*.gguf") if p.is_file() and not p.is_symlink()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


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
    except Exception:  # noqa: BLE001
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


def get_manifest_entry(source_uri: str) -> ManifestEntry | None:
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


def get_manifest_entry_by_slug(slug: str) -> ManifestEntry | None:
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


def remove_manifest_entry_by_slug(slug: str) -> ManifestEntry | None:
    """Remove a manifest entry by slug under the manifest lock.

    Returns the removed entry (so the caller can obtain the path to delete
    from disk), or None if no entry matched.
    """
    with _manifest_lock:
        manifest = read_manifest()
        removed: ManifestEntry | None = None
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
) -> dict | None:
    """Download a Harbor OCI artifact via ORAS into *target_dir*.

    Verifies every pulled file's sha256 against the OCI manifest layer
    digests (flat layers: layer digest = sha256 of the exact file bytes)
    and returns ``{filename: sha256-hex}`` for the verified files. Raises
    ``ModelPullError`` (502 model_pull_failed) on any mismatch so the
    caller's retry/backoff surfaces it.

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
    return _verify_pulled_digests(oras, harbor_ref, target_dir, source_uri)


def _verify_pulled_digests(
    oras: Any,
    harbor_ref: str,
    target_dir: Path,
    source_uri: str,
) -> dict | None:
    """Verify on-disk pulled files against the OCI manifest layer digests.

    Uses the shared client's authenticated manifest fetch (the public
    ``OrasHelper`` API does not expose manifests yet — follow-up: add a
    ``get_manifest_layers`` method to harbor-oci-client). If the manifest
    cannot be fetched or carries no per-file digests, verification is
    skipped (log warning) rather than failing the pull.

    Returns ``{filename: sha256-hex}`` of verified files, or None when
    verification was not possible.
    """
    try:
        manifest = oras._client.get_manifest(harbor_ref)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Post-pull digest verification skipped for %s: manifest fetch failed: %s",
            harbor_ref,
            exc,
        )
        return None

    expected: dict[str, str] = {}
    for layer in manifest.get("layers", []):
        title = (layer.get("annotations") or {}).get("org.opencontainers.image.title")
        digest = layer.get("digest", "")
        if title and digest.startswith("sha256:"):
            expected[title] = digest[len("sha256:") :]

    if not expected:
        logger.warning(
            "Post-pull digest verification skipped for %s: manifest has no "
            "per-file layer digests",
            harbor_ref,
        )
        return None

    problems: list[str] = []
    actual: dict[str, str] = {}
    for path in target_dir.iterdir():
        if not path.is_file():
            continue
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        actual[path.name] = h
        want = expected.get(path.name)
        if want is None:
            logger.warning(
                "Pulled file %s is not covered by the manifest for %s",
                path.name,
                harbor_ref,
            )
        elif h != want:
            problems.append(f"{path.name}: digest mismatch (got {h}, want {want})")
    for name in expected:
        if name not in actual:
            problems.append(f"{name}: missing on disk after pull")

    if problems:
        raise ModelPullError(
            502,
            "model_pull_failed",
            f"Pulled artifact integrity check failed: {'; '.join(problems)}",
            source_uri,
        )
    return actual


def _verify_cached_digests(entry: "ManifestEntry") -> bool:
    """Verify on-disk artifact files against the manifest entry's digests.

    Entries without recorded digests (pre-D-017) are trusted as-is.
    """
    if not entry.file_digests:
        return True
    base = Path(entry.path)
    for name, want in entry.file_digests.items():
        f = base / name
        if not f.is_file():
            logger.warning(
                "Cached artifact %s corrupt: %s missing", entry.source_uri, name
            )
            return False
        if hashlib.sha256(f.read_bytes()).hexdigest() != want:
            logger.warning(
                "Cached artifact %s corrupt: %s digest mismatch",
                entry.source_uri,
                name,
            )
            return False
    return True


def _pull_huggingface(
    model_id: str,
    target_dir: Path,
    source_uri: str,
) -> None:
    """Download a HuggingFace Hub model snapshot into *target_dir*."""
    import huggingface_hub  # type: ignore[import-untyped]

    hf_token: str | None = settings.hf_token or None

    huggingface_hub.snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        token=hf_token,
    )


def pull_model(
    *,
    source: str,
    source_uri: str,
    harbor_ref: str | None = None,
    model_id: str | None = None,
    digest: str | None = None,
    size_bytes: int | None = None,
    category: str | None = None,
    name: str | None = None,
    version: str | None = None,
    checksum: str | None = None,
    metadata: dict | None = None,
    backend_type: str | None = None,
) -> dict:
    """Download a model from Harbor or HuggingFace Hub and record it in the manifest.

    This function is synchronous and intended to be called via
    ``asyncio.to_thread()`` from the async route handler.

    Returns a dict with keys ``path``, ``cached``, and ``source_uri``.
    Raises ``ModelPullError`` for all expected failure conditions.

    GGUF selection: when the caller declares a llama.cpp backend
    (``backend_type == "llamacpp"``) and the artifact comes from Harbor
    (``repo://``), the returned path resolves to the largest ``*.gguf``
    inside the artifact directory instead of the directory itself — a
    llama-server model must be a file. An explicit subpath in the URI wins
    over selection. ``local://`` and ``huggingface://`` artifacts are never
    selected (they are used as directories).
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

    # Acquire a per-URI lock so that concurrent pulls for the *same* artifact
    # are serialised end-to-end.  The lock key is the base URI (subpath
    # stripped) — different subpaths of one artifact share the directory.
    uri_lock = _get_uri_lock(repo_base_uri(source_uri))
    with uri_lock:
        # 2a. Extract optional subpath (repo://name:version/subpath → file
        #     inside the artifact directory).  The slug stays name:version.
        subpath = extract_repo_subpath(source_uri)
        cache_key = repo_base_uri(source_uri)

        # 2b. GGUF selection for llama.cpp + Harbor artifacts: the returned
        #     path points at the largest *.gguf inside the artifact directory
        #     instead of the directory (llama-server needs a file).  An
        #     explicit subpath always wins over selection.
        select_gguf = source == "harbor" and backend_type == "llamacpp" and not subpath

        # 2. Cache check — manifest is the single source of truth, keyed by
        #    the base URI (the artifact identity).  Verify the resolved path
        #    (dir + optional subpath) still exists on disk AND the recorded
        #    per-file digests still match (a corrupt cached artifact would
        #    otherwise keep crashing every restart-in-place RECREATE).
        cached_entry = get_manifest_entry(cache_key)
        if cached_entry is not None:
            cached_path = Path(cached_entry.path)
            resolved_cache = cached_path / subpath if subpath else cached_path
            if select_gguf:
                resolved_cache = _select_gguf_path(cached_path)
                if resolved_cache is None:
                    raise ModelPullError(
                        404,
                        "not_found",
                        f"No .gguf file found in artifact for llamacpp backend ({source_uri}).",
                        source_uri,
                    )
            if resolved_cache.exists() and _verify_cached_digests(cached_entry):
                return {
                    "path": str(resolved_cache.resolve()),
                    "cached": True,
                    "source_uri": source_uri,
                }
            logger.warning(
                "Manifest entry for %s is missing or corrupt on disk, re-pulling",
                cache_key,
            )
            with _manifest_lock:
                remove_manifest_entry(cache_key)

        # 3. Derive slug and target directory.
        try:
            slug = source_uri_to_slug(source_uri)
        except ValueError as exc:
            raise ModelPullError(400, "invalid_request", str(exc), source_uri) from exc

        target_dir = get_models_dir() / slug

        # 3.5 Proactive disk space validation.
        disk = get_disk_info(str(get_models_dir()))
        if disk:
            available_gb = disk["available_gb"]
            required_gb = (
                (size_bytes / (1024**3)) if size_bytes else settings.min_free_disk_gb
            )

            if available_gb < required_gb:
                raise ModelPullError(
                    507,
                    "insufficient_storage",
                    f"Insufficient disk space. Available: {available_gb:.2f} GB, required: {required_gb:.2f} GB.",
                    source_uri,
                )

        # 4. Validate credentials before touching the filesystem.
        if source == "harbor" and not all(
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

        # 6. Download — subprocess + polling allows aborting on low disk (S-018).
        # In-process mode skips the worker (used by unit tests that mock pull funcs).
        file_digests: dict | None = None
        try:
            if settings.pull_use_subprocess:
                poll_s = max(0.05, settings.pull_disk_poll_interval_s)
                with pebble.ProcessPool(max_workers=1) as pool:
                    if source == "harbor":
                        future = pool.schedule(
                            _pull_harbor,
                            args=(harbor_ref or "", target_dir, source_uri),
                        )
                    else:
                        future = pool.schedule(
                            _pull_huggingface,
                            args=(model_id or "", target_dir, source_uri),
                        )

                    while not future.done():
                        time.sleep(poll_s)
                        disk = get_disk_info(str(get_models_dir()))
                        if disk and disk["available_gb"] < settings.min_free_disk_gb:
                            future.cancel()
                            logger.error(
                                "Aborting pull for %s: disk space dropped below %s GB",
                                source_uri,
                                settings.min_free_disk_gb,
                            )
                            raise ModelPullError(
                                507,
                                "insufficient_storage",
                                "Insufficient disk space during download.",
                                source_uri,
                            )

                    file_digests = future.result()
            else:
                if source == "harbor":
                    file_digests = _pull_harbor(
                        harbor_ref or "", target_dir, source_uri
                    )
                else:
                    _pull_huggingface(model_id or "", target_dir, source_uri)
        except ModelPullError:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise
        except pebble.ProcessExpired as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise ModelPullError(
                500, "model_pull_failed", f"Download process expired: {exc}", source_uri
            ) from exc
        except OSError as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            if exc.errno == errno.ENOSPC:
                raise ModelPullError(
                    507, "insufficient_storage", "Insufficient disk space.", source_uri
                ) from exc
            # huggingface_hub errors (e.g. RepositoryNotFoundError) subclass OSError
            # via httpx.HTTPError; they must not be collapsed to a generic 500 here.
            _map_download_exception(exc, source_uri)
        except Exception as exc:  # noqa: BLE001
            shutil.rmtree(target_dir, ignore_errors=True)
            _map_download_exception(exc, source_uri)

        # 7. Compute size of downloaded files.
        size_bytes = _compute_dir_size(target_dir)

        # 7.5 Resolve the returned path.  When the URI carries a subpath
        #     (repo://name:version/model.gguf), the returned path points at
        #     that file inside the pulled directory.  A subpath that does
        #     not exist in the artifact is a client error (404).
        #     When GGUF selection applies (llama.cpp + Harbor artifact, no
        #     subpath), the path points at the largest *.gguf inside the
        #     pulled directory; an artifact without any .gguf is a client
        #     error (404) — the intent can never run on llama.cpp.
        if subpath:
            resolved_path = target_dir / subpath
            if not resolved_path.exists():
                shutil.rmtree(target_dir, ignore_errors=True)
                raise ModelPullError(
                    404,
                    "not_found",
                    f"Subpath '{subpath}' not found in artifact for {source_uri}.",
                    source_uri,
                )
        elif select_gguf:
            resolved_path = _select_gguf_path(target_dir)
            if resolved_path is None:
                shutil.rmtree(target_dir, ignore_errors=True)
                raise ModelPullError(
                    404,
                    "not_found",
                    f"No .gguf file found in artifact for llamacpp backend ({source_uri}).",
                    source_uri,
                )
        else:
            resolved_path = target_dir

        # 8. Update manifest atomically under lock to prevent concurrent write
        #    races between pulls for *different* URIs finishing simultaneously.
        #    The entry is keyed by the base URI (artifact identity), so any
        #    subpath of the same artifact resolves against the same entry.
        entry = ManifestEntry(
            slug=slug,
            source_uri=cache_key,
            path=str(target_dir.resolve()),
            size_bytes=size_bytes,
            digest=digest,
            downloaded_at=datetime.now(UTC).isoformat(),
            category=category,
            name=name,
            version=version,
            checksum=checksum,
            metadata=metadata,
            file_digests=file_digests,
        )
        with _manifest_lock:
            add_manifest_entry(entry)

        logger.info("Model pulled successfully: %s -> %s", source_uri, resolved_path)
        return {
            "path": str(resolved_path.resolve()),
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
