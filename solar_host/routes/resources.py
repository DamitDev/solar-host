"""Routes for the resource reservation API (S-034).

Endpoints:
    POST   /resources/reservations          → 201 ReservationView
    GET    /resources                       → 200 ResourceSnapshot
    DELETE /resources/reservations/{id}     → 200 {detail, id}

Error mapping:
    409  capacity_exceeded / running_release
    422  pydantic validation (automatic) / duplicate expiry fields
    404  unknown reservation id
    503  resource_manager not initialised (Docker unavailable at startup)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse

from solar_host.resources.manager import (
    CapacityExceededError,
    ReservationRunningError,
    ResourceManager,
)
from solar_host.resources.models import (
    ReservationRequest,
    ReservationView,
    ResourceSnapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/resources", tags=["resources"])


def _get_manager(request: Request) -> ResourceManager:
    manager: ResourceManager | None = getattr(
        request.app.state, "resource_manager", None
    )
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Resource manager unavailable",
        )
    return manager


@router.post(
    "/reservations",
    response_model=ReservationView,
    status_code=status.HTTP_201_CREATED,
    summary="Create a resource reservation",
)
async def create_reservation(req: ReservationRequest, request: Request) -> Any:
    """Reserve VRAM/RAM/disk capacity before placing a training job.

    Returns **201 Created** with the new ``ReservationView`` on success.

    Error responses:
    - **409** when the request would exceed available capacity
      (body: ``{error, dimension, requested_gb, available_gb}``).
    - **422** for invalid request body (Pydantic) or conflicting expiry fields.
    - **503** when the resource manager is unavailable.
    """
    manager = _get_manager(request)
    try:
        reservation = manager.create(req)
    except CapacityExceededError as exc:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": "capacity_exceeded",
                "dimension": exc.dimension,
                "requested_gb": exc.requested_gb,
                "available_gb": exc.available_gb,
            },
        )

    # Build the view from the snapshot (picks up status from JobStore).
    snap = manager.snapshot()
    for view in snap.reservations:
        if view.id == reservation.id:
            return view

    # Fallback (reservation created but not yet visible in snapshot — should not happen).
    return ReservationView(
        id=reservation.id,
        job_id=reservation.job_id,
        workload_type=reservation.workload_type,
        status="pending",
        vram_gb=reservation.vram_gb,
        ram_gb=reservation.ram_gb,
        disk_gb=reservation.disk_gb,
        expires_at=reservation.expires_at,
    )


@router.get(
    "",
    response_model=ResourceSnapshot,
    summary="Get resource availability and reservations",
)
async def get_resources(request: Request) -> ResourceSnapshot:
    """Return per-dimension capacity snapshot and the full reservation list.

    The ``reservations`` array includes per-job actual usage for running jobs
    and ``null`` actuals for pending ones.
    """
    manager = _get_manager(request)
    return manager.snapshot()


@router.delete(
    "/reservations/{reservation_id}",
    summary="Release a resource reservation",
)
async def delete_reservation(reservation_id: str, request: Request) -> dict[str, str]:
    """Release a reservation by ID.

    Returns **200** ``{detail: "released", id}`` on success.

    Error responses:
    - **404** when the reservation ID is unknown.
    - **409** when the linked job is currently running (running reservations
      must not be released while the job holds actual capacity).
    """
    manager = _get_manager(request)
    try:
        manager.release(reservation_id)
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reservation {reservation_id!r} not found",
        )
    except ReservationRunningError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return {"detail": "released", "id": reservation_id}
