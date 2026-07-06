"""ResourceManager: thread-safe in-memory reservation ledger (S-034).

Accounting formula (per dimension d):
    reported_usage_d = system.usage_d + Σ_i max(reservation_i.reserved_d − reservation_i.actual_d ?? 0, 0)
    available_d      = total_d − reported_usage_d

For a *pending* reservation actual is None → treated as 0, so the full
reserved amount is added as headroom.  For a *running* reservation only the
unconsumed headroom max(reserved − actual, 0) is added, so real consumption
that is already captured in system.usage is never double-counted.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Optional

import psutil

from solar_host.memory_monitor import get_disk_info, get_memory_info
from solar_host.resources.models import (
    Reservation,
    ReservationRequest,
    ReservationView,
    ResourceDimensionSnapshot,
    ResourceSnapshot,
)

if TYPE_CHECKING:
    from solar_host.docker.service import DockerService
    from solar_host.jobs.executor import JobExecutor
    from solar_host.jobs.store import JobStore

logger = logging.getLogger(__name__)


class CapacityExceededError(Exception):
    """Raised when a new reservation would exceed available capacity."""

    def __init__(
        self, dimension: str, requested_gb: float, available_gb: float
    ) -> None:
        super().__init__(
            f"Capacity exceeded for {dimension}: requested {requested_gb:.3f} GB, "
            f"available {available_gb:.3f} GB"
        )
        self.dimension = dimension
        self.requested_gb = requested_gb
        self.available_gb = available_gb


class ReservationRunningError(Exception):
    """Raised when trying to release a reservation whose job is running."""

    def __init__(self, reservation_id: str, job_id: str) -> None:
        super().__init__(
            f"Cannot release reservation {reservation_id!r}: "
            f"job {job_id!r} is currently running"
        )
        self.reservation_id = reservation_id
        self.job_id = job_id


class ResourceManager:
    """Thread-safe in-memory resource reservation ledger.

    Mirrors the ``threading.Lock`` pattern used by ``JobStore`` and
    ``ConfigManager``.  All public methods acquire ``_lock`` before reading or
    mutating ``_reservations``.
    """

    def __init__(
        self,
        job_store: "JobStore",
        docker_service: Optional["DockerService"],
        job_executor: Optional["JobExecutor"],
        jobs_dir: str = "./jobs",
        default_ttl_seconds: Optional[float] = 86400.0,
    ) -> None:
        self._job_store = job_store
        self._docker_service = docker_service
        self._job_executor = job_executor
        self._jobs_dir = jobs_dir
        self._default_ttl_seconds = default_ttl_seconds
        self._reservations: dict[str, Reservation] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create(self, req: ReservationRequest) -> Reservation:
        """Create and register a new reservation after a capacity check.

        Raises:
            CapacityExceededError: when any requested dimension would push
                ``reported_used + requested > total``.
        """
        with self._lock:
            now = datetime.now(UTC)
            snap = self._snapshot_unlocked(now)

            if snap.vram is not None and req.vram_gb > snap.vram.available_gb:
                raise CapacityExceededError("vram", req.vram_gb, snap.vram.available_gb)
            if snap.ram is not None and req.ram_gb > snap.ram.available_gb:
                raise CapacityExceededError("ram", req.ram_gb, snap.ram.available_gb)
            if (
                req.disk_gb is not None
                and snap.disk is not None
                and req.disk_gb > snap.disk.available_gb
            ):
                raise CapacityExceededError("disk", req.disk_gb, snap.disk.available_gb)

            reservation = Reservation.from_request(
                req, now, default_ttl_seconds=self._default_ttl_seconds
            )
            self._reservations[reservation.id] = reservation
            logger.info("Created reservation %s for job %s", reservation.id, req.job_id)
            return reservation

    def release(self, reservation_id: str) -> None:
        """Remove a reservation by ID.

        Raises:
            KeyError: when the reservation ID is unknown.
            ReservationRunningError: when the linked job is currently running
                (per decision O6 — running reservations cannot be released).
        """
        with self._lock:
            res = self._reservations.get(reservation_id)
            if res is None:
                raise KeyError(f"Reservation {reservation_id!r} not found")
            if self._status_unlocked(res) == "running":
                raise ReservationRunningError(reservation_id, res.job_id)
            del self._reservations[reservation_id]
        logger.info("Released reservation %s", reservation_id)

    def snapshot(self) -> ResourceSnapshot:
        """Return a current ResourceSnapshot (reads live system usage)."""
        with self._lock:
            return self._snapshot_unlocked(datetime.now(UTC))

    async def refresh_usage_async(self) -> None:
        """Update actual usage for all *running* reservations.

        Intended to be awaited directly from the ``resource_usage_poll_loop``
        background task in ``main.py``.
        """
        with self._lock:
            running = [
                res
                for res in self._reservations.values()
                if self._status_unlocked(res) == "running"
            ]

        if not running or self._docker_service is None or self._job_executor is None:
            return

        await self._refresh_actuals(running)

    def cleanup_expired(self, now: datetime) -> int:
        """Remove expired non-running reservations and return the count removed.

        Running reservations are never expired — they release when the job
        completes or is cancelled (per decision 5 of the supervisor Q&A).
        """
        with self._lock:
            expired_ids = [
                res_id
                for res_id, res in self._reservations.items()
                if (
                    res.expires_at is not None
                    and res.expires_at <= now
                    and self._status_unlocked(res) != "running"
                )
            ]
            for res_id in expired_ids:
                del self._reservations[res_id]

        if expired_ids:
            logger.info("Expired %d reservation(s)", len(expired_ids))
        return len(expired_ids)

    # ------------------------------------------------------------------
    # Internal helpers  (caller must hold _lock unless noted)
    # ------------------------------------------------------------------

    def _status_unlocked(self, res: Reservation) -> str:
        """Return ``'running'`` iff the linked job is in *running* state."""
        from solar_host.jobs.models import JobStatus

        state = self._job_store.get(res.job_id)
        if state is not None and state.status == JobStatus.running:
            return "running"
        return "pending"

    def _headroom_unlocked(self, dimension: str) -> float:
        """Compute Σ max(reserved_d − actual_d, 0) across all reservations."""
        total = 0.0
        for res in self._reservations.values():
            if dimension == "vram":
                reserved = res.vram_gb
                actual = res.actual_vram_gb if res.actual_vram_gb is not None else 0.0
            elif dimension == "ram":
                reserved = res.ram_gb
                actual = res.actual_ram_gb if res.actual_ram_gb is not None else 0.0
            elif dimension == "disk":
                reserved = res.disk_gb if res.disk_gb is not None else 0.0
                actual = res.actual_disk_gb if res.actual_disk_gb is not None else 0.0
            else:
                continue
            total += max(reserved - actual, 0.0)
        return total

    def _snapshot_unlocked(self, _now: datetime) -> ResourceSnapshot:
        """Compute a ResourceSnapshot without acquiring the lock (caller holds it).

        Reads live system usage via memory_monitor (CACHE_DURATION = 5 s) and
        get_disk_info, then applies the accounting formula using each
        reservation's cached actual usage.
        """
        from solar_host.config import settings

        mem = get_memory_info()
        memory_type = "RAM"
        vram_dim: Optional[ResourceDimensionSnapshot] = None
        ram_dim: Optional[ResourceDimensionSnapshot] = None

        if mem is not None:
            memory_type = str(mem["memory_type"])
            sys_used = float(mem["used_gb"])
            total = float(mem["total_gb"])

            if memory_type == "VRAM":
                headroom = self._headroom_unlocked("vram")
                reported = sys_used + headroom
                vram_dim = ResourceDimensionSnapshot(
                    total_gb=total,
                    system_used_gb=sys_used,
                    reserved_headroom_gb=headroom,
                    reported_used_gb=reported,
                    available_gb=max(0.0, total - reported),
                )
                # Also compute RAM independently via psutil.
                try:
                    vm = psutil.virtual_memory()
                    ram_sys = vm.used / (1024**3)
                    ram_total = vm.total / (1024**3)
                    ram_headroom = self._headroom_unlocked("ram")
                    ram_reported = ram_sys + ram_headroom
                    ram_dim = ResourceDimensionSnapshot(
                        total_gb=round(ram_total, 2),
                        system_used_gb=round(ram_sys, 2),
                        reserved_headroom_gb=ram_headroom,
                        reported_used_gb=ram_reported,
                        available_gb=max(0.0, ram_total - ram_reported),
                    )
                except Exception:
                    pass
            else:
                headroom = self._headroom_unlocked("ram")
                reported = sys_used + headroom
                ram_dim = ResourceDimensionSnapshot(
                    total_gb=total,
                    system_used_gb=sys_used,
                    reserved_headroom_gb=headroom,
                    reported_used_gb=reported,
                    available_gb=max(0.0, total - reported),
                )

        disk_dim: Optional[ResourceDimensionSnapshot] = None
        disk_info = get_disk_info(settings.jobs_dir)
        if disk_info is not None:
            sys_disk = float(disk_info["used_gb"])
            total_disk = float(disk_info["total_gb"])
            headroom_disk = self._headroom_unlocked("disk")
            reported_disk = sys_disk + headroom_disk
            disk_dim = ResourceDimensionSnapshot(
                total_gb=total_disk,
                system_used_gb=sys_disk,
                reserved_headroom_gb=headroom_disk,
                reported_used_gb=reported_disk,
                available_gb=max(0.0, total_disk - reported_disk),
            )

        views = [self._to_view_unlocked(res) for res in self._reservations.values()]
        return ResourceSnapshot(
            memory_type=memory_type,
            vram=vram_dim,
            ram=ram_dim,
            disk=disk_dim,
            reservations=views,
        )

    def _to_view_unlocked(self, res: Reservation) -> ReservationView:
        return ReservationView(
            id=res.id,
            job_id=res.job_id,
            workload_type=res.workload_type,
            status=self._status_unlocked(res),
            vram_gb=res.vram_gb,
            ram_gb=res.ram_gb,
            disk_gb=res.disk_gb,
            actual_vram_gb=res.actual_vram_gb,
            actual_ram_gb=res.actual_ram_gb,
            actual_disk_gb=res.actual_disk_gb,
            expires_at=res.expires_at,
        )

    async def _refresh_actuals(self, running: list[Reservation]) -> None:
        """Collect actual usage for each running reservation and cache it."""
        from solar_host.resources.usage import (
            collect_container_ram_gb,
            collect_container_vram_gb,
            collect_workspace_disk_gb,
        )

        assert self._docker_service is not None
        assert self._job_executor is not None
        now = datetime.now(UTC)

        for res in running:
            container_id = self._job_executor.get_active_container(res.job_id)
            if container_id is None:
                continue

            actual_ram = await collect_container_ram_gb(
                self._docker_service, container_id
            )
            actual_vram = await collect_container_vram_gb(
                self._docker_service, container_id
            )
            actual_disk = await collect_workspace_disk_gb(res.job_id, self._jobs_dir)

            with self._lock:
                stored = self._reservations.get(res.id)
                if stored is not None:
                    self._reservations[res.id] = stored.model_copy(
                        update={
                            "actual_ram_gb": actual_ram,
                            "actual_vram_gb": actual_vram,
                            "actual_disk_gb": actual_disk,
                            "usage_polled_at": now,
                        }
                    )
