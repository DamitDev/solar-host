"""Pydantic models for resource reservations (S-034)."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class WorkloadType(str, Enum):
    training = "training"
    inference = "inference"
    other = "other"


class ReservationRequest(BaseModel):
    """Input model for POST /resources/reservations."""

    job_id: str
    requester: Optional[str] = None
    workload_type: WorkloadType = WorkloadType.training
    vram_gb: float
    ram_gb: float
    disk_gb: Optional[float] = None
    ttl_seconds: Optional[float] = None
    expires_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _validate_expiry(self) -> "ReservationRequest":
        if self.ttl_seconds is not None and self.expires_at is not None:
            raise ValueError(
                "Specify at most one of ttl_seconds or expires_at, not both"
            )
        return self


class Reservation(BaseModel):
    """Runtime state for an active reservation."""

    id: str
    job_id: str
    requester: Optional[str] = None
    workload_type: WorkloadType
    vram_gb: float
    ram_gb: float
    disk_gb: Optional[float] = None
    created_at: datetime
    expires_at: Optional[datetime] = None

    # Cached actual usage (updated by the poll loop for running reservations).
    actual_vram_gb: Optional[float] = None
    actual_ram_gb: Optional[float] = None
    actual_disk_gb: Optional[float] = None
    usage_polled_at: Optional[datetime] = None

    @classmethod
    def from_request(cls, req: ReservationRequest, now: datetime) -> "Reservation":
        expires_at: Optional[datetime] = None
        if req.expires_at is not None:
            expires_at = req.expires_at
        elif req.ttl_seconds is not None:
            expires_at = now + timedelta(seconds=req.ttl_seconds)
        return cls(
            id=f"res-{uuid.uuid4().hex}",
            job_id=req.job_id,
            requester=req.requester,
            workload_type=req.workload_type,
            vram_gb=req.vram_gb,
            ram_gb=req.ram_gb,
            disk_gb=req.disk_gb,
            created_at=now,
            expires_at=expires_at,
        )


class ResourceDimensionSnapshot(BaseModel):
    """Capacity snapshot for a single resource dimension (VRAM/RAM/disk)."""

    total_gb: float
    system_used_gb: float
    reserved_headroom_gb: float
    reported_used_gb: float
    available_gb: float


class ReservationView(BaseModel):
    """Public view of a reservation (returned in API responses)."""

    id: str
    job_id: str
    workload_type: WorkloadType
    status: str  # "pending" | "running"
    vram_gb: float
    ram_gb: float
    disk_gb: Optional[float] = None
    actual_vram_gb: Optional[float] = None
    actual_ram_gb: Optional[float] = None
    actual_disk_gb: Optional[float] = None
    expires_at: Optional[datetime] = None


class ResourceSnapshot(BaseModel):
    """Full resource snapshot returned by GET /resources."""

    memory_type: str  # "VRAM" or "RAM" — primary dimension reported by get_memory_info
    vram: Optional[ResourceDimensionSnapshot] = None
    ram: Optional[ResourceDimensionSnapshot] = None
    disk: Optional[ResourceDimensionSnapshot] = None
    reservations: list[ReservationView] = []
