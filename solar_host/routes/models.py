"""GET /models and POST /models/pull routes.

GET /models — lists models recorded in MODELS_DIR/manifest.json.
Per S-009, the manifest is the single source of truth. This endpoint does not
scan the filesystem for models; only entries present in the manifest are
returned. Missing or invalid manifest yields an empty list (see read_manifest).

POST /models/pull — pulls a model from Harbor (ORAS) or HuggingFace Hub and
records it in the manifest. Returns the local path and whether it was a cache
hit. Per S-015 / spec Section 3.6.
"""

import asyncio
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from solar_host import models_manager
from solar_host.models_manager import ModelPullError, read_manifest

router = APIRouter(prefix="/models", tags=["models"])


# ---------------------------------------------------------------------------
# GET /models — list
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# POST /models/pull
# ---------------------------------------------------------------------------


class PullRequest(BaseModel):
    """Request body for POST /models/pull (spec Section 3.6)."""

    model_config = {"protected_namespaces": ()}

    source: Literal["harbor", "huggingface"]
    source_uri: str
    harbor_ref: Optional[str] = None
    model_id: Optional[str] = None
    digest: Optional[str] = None


class PullResponse(BaseModel):
    """Response body for POST /models/pull (spec Section 3.6)."""

    path: str
    cached: bool
    source_uri: str


@router.post(
    "/pull",
    response_model=PullResponse,
    summary="Pull a model from source",
    responses={
        200: {
            "description": "Model available at returned path (cached or freshly downloaded)"
        },
        400: {
            "description": "Invalid request (bad source_uri scheme or malformed URI)"
        },
        401: {"description": "Source authentication failed"},
        404: {"description": "Model not found at source"},
        422: {"description": "Missing required field for the chosen source"},
        500: {"description": "Missing credentials or unexpected server error"},
        502: {"description": "Source registry/hub unreachable"},
        507: {"description": "Insufficient disk space"},
    },
)
async def pull_model(req: PullRequest) -> PullResponse:
    """Pull a model from Harbor or HuggingFace Hub.

    Checks the manifest cache first. On a cache hit the stored path is returned
    immediately without re-downloading. On a cache miss the model is downloaded,
    the manifest is updated atomically, and the new path is returned.

    The caller blocks until the model is fully available.
    """
    # Validate conditional required fields before doing any I/O.
    if req.source == "harbor" and not req.harbor_ref:
        raise HTTPException(
            status_code=422, detail="harbor_ref is required for harbor source"
        )
    if req.source == "huggingface" and not req.model_id:
        raise HTTPException(
            status_code=422, detail="model_id is required for huggingface source"
        )

    try:
        result = await asyncio.to_thread(
            models_manager.pull_model,
            source=req.source,
            source_uri=req.source_uri,
            harbor_ref=req.harbor_ref,
            model_id=req.model_id,
            digest=req.digest,
        )
    except ModelPullError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error,
                "detail": exc.detail,
                "source_uri": exc.source_uri,
                "status_code": exc.status_code,
            },
        )

    return PullResponse(**result)
