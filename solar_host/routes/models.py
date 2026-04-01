"""GET /models endpoint — lists models from the managed models directory.

Source data is the manifest (single source of truth for tracked models).
Subdirectories in MODELS_DIR that are not in the manifest are also included
as untracked entries so that manually-placed models are visible.
Orphaned manifest entries (directory deleted) are included as well so that
consumers such as the HuggingFace resolver (S-012) can distinguish "known but
missing" from "never seen".
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from solar_host.models_manager import get_models_dir, read_manifest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


class ModelEntry(BaseModel):
    """A single model entry returned by GET /models."""

    name: str
    path: str
    size_bytes: int
    source_uri: Optional[str] = None
    checksum: Optional[str] = None
    downloaded_at: Optional[str] = None
    in_manifest: bool


def _dir_size(directory: Path) -> int:
    """Return the total size in bytes of all files under *directory*."""
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def _build_model_list(models_dir: Path) -> List[ModelEntry]:
    """Build the list of ModelEntry objects synchronously (for thread offload)."""
    manifest = read_manifest()

    # Index manifest entries by slug for O(1) lookup.
    by_slug: dict[str, object] = {entry.slug: entry for entry in manifest.models}

    results: List[ModelEntry] = []
    seen_slugs: set[str] = set()

    # Walk subdirectories in MODELS_DIR.
    if models_dir.is_dir():
        for child in sorted(models_dir.iterdir()):
            if not child.is_dir():
                continue

            slug = child.name
            seen_slugs.add(slug)

            if slug in by_slug:
                entry = by_slug[slug]  # type: ignore[assignment]
                results.append(
                    ModelEntry(
                        name=entry.slug,  # type: ignore[attr-defined]
                        path=entry.path,  # type: ignore[attr-defined]
                        size_bytes=entry.size_bytes,  # type: ignore[attr-defined]
                        source_uri=entry.source_uri,  # type: ignore[attr-defined]
                        checksum=entry.digest,  # type: ignore[attr-defined]
                        downloaded_at=entry.downloaded_at,  # type: ignore[attr-defined]
                        in_manifest=True,
                    )
                )
            else:
                # Untracked model — compute size from disk.
                try:
                    size = _dir_size(child)
                except OSError:
                    size = 0
                results.append(
                    ModelEntry(
                        name=slug,
                        path=str(child.resolve()),
                        size_bytes=size,
                        source_uri=None,
                        checksum=None,
                        downloaded_at=None,
                        in_manifest=False,
                    )
                )

    # Include orphaned manifest entries (in manifest but directory absent).
    for entry in manifest.models:
        if entry.slug not in seen_slugs:
            results.append(
                ModelEntry(
                    name=entry.slug,
                    path=entry.path,
                    size_bytes=entry.size_bytes,
                    source_uri=entry.source_uri,
                    checksum=entry.digest,
                    downloaded_at=entry.downloaded_at,
                    in_manifest=True,
                )
            )

    return results


@router.get("", response_model=List[ModelEntry], summary="List managed models")
async def list_models() -> List[ModelEntry]:
    """Return all models known to this host.

    Combines manifest-tracked models with any subdirectories in MODELS_DIR
    that are not yet in the manifest (e.g. manually placed files).
    Manifest entries whose directories have been deleted are also returned so
    that callers can detect orphaned cache entries.

    Returns an empty list if the models directory or manifest does not exist.
    """
    models_dir = get_models_dir()
    return await asyncio.to_thread(_build_model_list, models_dir)
