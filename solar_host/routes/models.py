"""GET /models — lists models recorded in MODELS_DIR/manifest.json.

Per S-009, the manifest is the single source of truth. This endpoint does not
scan the filesystem for models; only entries present in the manifest are
returned. Missing or invalid manifest yields an empty list (see read_manifest).
"""

import asyncio
from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from solar_host.models_manager import read_manifest

router = APIRouter(prefix="/models", tags=["models"])


class ModelEntry(BaseModel):
    """A single model entry returned by GET /models."""

    name: str
    path: str
    size_bytes: int
    source_uri: Optional[str] = None
    checksum: Optional[str] = None
    downloaded_at: Optional[str] = None


def _manifest_to_entries() -> List[ModelEntry]:
    manifest = read_manifest()
    return [
        ModelEntry(
            name=e.slug,
            path=e.path,
            size_bytes=e.size_bytes,
            source_uri=e.source_uri,
            checksum=e.digest,
            downloaded_at=e.downloaded_at,
        )
        for e in manifest.models
    ]


@router.get("", response_model=List[ModelEntry], summary="List managed models")
async def list_models() -> List[ModelEntry]:
    """Return all models listed in the managed models manifest.

    Data comes only from ``manifest.json`` under ``MODELS_DIR`` (see
    ``read_manifest``). No directory scanning is performed.

    Returns an empty list when the manifest is missing, empty, or unreadable.
    """
    return await asyncio.to_thread(_manifest_to_entries)
