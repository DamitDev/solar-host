"""REST endpoints for job submission, inspection, and cancellation (S-027)."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status

if TYPE_CHECKING:
    from solar_host.jobs.executor import JobExecutor

from solar_host.jobs.errors import (
    GpuValidationError,
    InsufficientDiskError,
    WorkspaceError,
)
from solar_host.jobs.models import JobState
from solar_host.jobs.step_log_buffer import step_log_buffer
from solar_host.jobs.workspace import delete_workspace
from solar_host.routes.job_schemas import (
    JobStateResponse,
    JobSubmitRequest,
    JobSubmitResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Terminal job statuses — DELETE on these is a 409.
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# Error → HTTP mapping
# ---------------------------------------------------------------------------


def _map_error(exc: Exception) -> HTTPException:
    """Convert a jobs-layer exception to the appropriate HTTPException."""
    if isinstance(exc, InsufficientDiskError):
        return HTTPException(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            detail={
                "error": "insufficient_storage",
                "required_gb": exc.required_gb,
                "available_gb": exc.available_gb,
            },
        )
    if isinstance(exc, GpuValidationError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "gpu_unavailable",
                "requested": exc.requested,
                "available_count": exc.available_count,
            },
        )
    if isinstance(exc, WorkspaceError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "workspace_create_failed", "reason": exc.reason},
        )
    if isinstance(exc, ValueError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    if isinstance(exc, KeyError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=str(exc),
    )


def _get_executor(request: Request) -> JobExecutor:
    """Return the job executor from app state, raising 503 when unavailable."""
    executor = getattr(request.app.state, "job_executor", None)
    if executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "executor_unavailable"},
        )
    return executor


def _get_job_or_404(request: Request, job_id: str) -> JobState:
    """Return the JobState for *job_id* or raise 404."""
    store = request.app.state.job_store
    job = store.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id!r} not found",
        )
    return job


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=JobSubmitResponse)
async def submit_job(body: JobSubmitRequest, request: Request) -> JobSubmitResponse:
    """Submit a new job for background execution."""
    executor = _get_executor(request)
    try:
        job_state = await executor.submit_job(body)
    except (
        ValueError,
        KeyError,
        InsufficientDiskError,
        WorkspaceError,
        GpuValidationError,
    ) as exc:
        raise _map_error(exc) from exc
    return JobSubmitResponse(
        job_id=job_state.job_id,
        status=job_state.status.value,
        workspace_path=job_state.workspace_path,
        submission_id=job_state.submission_id,
        correlation_id=job_state.correlation_id,
    )


@router.get("/{job_id}", response_model=JobStateResponse)
async def get_job(job_id: str, request: Request) -> JobStateResponse:
    """Return the current state of a job, including per-step log snippets."""
    job = _get_job_or_404(request, job_id)
    return JobStateResponse.from_job_state(job)


@router.delete("/{job_id}")
async def delete_job(job_id: str, request: Request) -> dict:
    """Cancel a running job, delete its workspace, and remove it from the store."""
    executor = _get_executor(request)
    job = _get_job_or_404(request, job_id)

    if job.status.value in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id!r} is already in terminal state {job.status.value!r}",
        )

    await executor.cancel_job(job_id)

    # Wait for the cancelled task to finish.  The Docker stop timeout is 30 s
    # (10 s SIGTERM grace + kill); use 20 s here so the API doesn't time out
    # before Docker even sends SIGKILL.
    try:
        await executor.await_job(job_id, timeout=20.0)
    except asyncio.TimeoutError:
        logger.warning("Timed out waiting for job %r to finish after cancel", job_id)
    except Exception:
        logger.exception(
            "Error awaiting job %r after cancel — cleaning up anyway", job_id
        )

    if job.workspace_path:
        await asyncio.to_thread(delete_workspace, Path(job.workspace_path))

    step_log_buffer.remove(job_id)
    request.app.state.job_store.remove(job_id)

    return {"detail": "cancelled", "job_id": job_id}
