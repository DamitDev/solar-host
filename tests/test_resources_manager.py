"""Unit tests for ResourceManager math and logic (S-034).

No HTTP — exercises the in-memory ledger directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from solar_host.jobs.models import JobState, JobStatus, StepState
from solar_host.jobs.store import JobStore
from solar_host.resources.manager import (
    CapacityExceededError,
    ReservationRunningError,
    ResourceManager,
)
from solar_host.resources.models import ReservationRequest, WorkloadType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MEM_VRAM = {
    "used_gb": 8.0,
    "total_gb": 24.0,
    "available_gb": 16.0,
    "percent": 33.33,
    "memory_type": "VRAM",
}
_MEM_RAM = {
    "used_gb": 4.0,
    "total_gb": 16.0,
    "available_gb": 12.0,
    "percent": 25.0,
    "memory_type": "RAM",
}
_DISK = {"used_gb": 100.0, "total_gb": 500.0, "available_gb": 400.0}


def _make_request(
    job_id: str = "job-001",
    vram_gb: float = 4.0,
    ram_gb: float = 2.0,
    disk_gb: float | None = None,
    ttl_seconds: float | None = None,
) -> ReservationRequest:
    return ReservationRequest(
        job_id=job_id,
        workload_type=WorkloadType.training,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        disk_gb=disk_gb,
        ttl_seconds=ttl_seconds,
    )


def _make_running_job(job_id: str = "job-001") -> JobState:
    now = datetime.now(UTC)
    return JobState(
        job_id=job_id,
        name="Test Job",
        status=JobStatus.running,
        steps=[StepState(name="step1")],
        current_step_index=0,
        workspace_path=f"/tmp/jobs/{job_id}",
        created_at=now,
        started_at=now,
        retention_hours=24.0,
    )


def _make_manager(store: JobStore | None = None) -> ResourceManager:
    return ResourceManager(
        job_store=store or JobStore(),
        docker_service=None,
        job_executor=None,
        jobs_dir="/tmp/test-jobs",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_memory(monkeypatch):
    """Stub memory_monitor so tests run without NVIDIA hardware."""
    import psutil

    monkeypatch.setattr(
        "solar_host.resources.manager.get_memory_info", lambda: _MEM_VRAM
    )
    monkeypatch.setattr(
        "solar_host.resources.manager.get_disk_info", lambda _path: _DISK
    )
    # Patch psutil.virtual_memory used for the RAM dimension on VRAM hosts.
    vm = MagicMock()
    vm.used = int(4.0 * 1024**3)
    vm.total = int(16.0 * 1024**3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: vm)


# ---------------------------------------------------------------------------
# Snapshot baseline
# ---------------------------------------------------------------------------


class TestSnapshotBaseline:
    def test_empty_ledger_available_equals_total_minus_system(self):
        mgr = _make_manager()
        snap = mgr.snapshot()
        assert snap.vram is not None
        assert snap.vram.available_gb == pytest.approx(16.0)  # 24 - 8
        assert snap.vram.reserved_headroom_gb == 0.0

    def test_ram_dimension_also_populated_on_vram_host(self):
        mgr = _make_manager()
        snap = mgr.snapshot()
        assert snap.ram is not None
        assert snap.ram.available_gb == pytest.approx(12.0)  # 16 - 4

    def test_disk_dimension_populated(self):
        mgr = _make_manager()
        snap = mgr.snapshot()
        assert snap.disk is not None
        assert snap.disk.total_gb == pytest.approx(500.0)
        assert snap.disk.available_gb == pytest.approx(400.0)


# ---------------------------------------------------------------------------
# Pending reservation math
# ---------------------------------------------------------------------------


class TestPendingReservationMath:
    def test_pending_reduces_available_by_full_reserved(self):
        mgr = _make_manager()
        mgr.create(_make_request(vram_gb=4.0, ram_gb=2.0))
        snap = mgr.snapshot()
        assert snap.vram is not None
        # system 8 + headroom 4 = 12 reported; 24 - 12 = 12 available
        assert snap.vram.reserved_headroom_gb == pytest.approx(4.0)
        assert snap.vram.available_gb == pytest.approx(12.0)

    def test_multiple_pending_reservations_sum_headroom(self):
        mgr = _make_manager()
        mgr.create(_make_request(job_id="job-001", vram_gb=3.0, ram_gb=1.0))
        mgr.create(_make_request(job_id="job-002", vram_gb=2.0, ram_gb=1.0))
        snap = mgr.snapshot()
        assert snap.vram is not None
        assert snap.vram.reserved_headroom_gb == pytest.approx(5.0)
        assert snap.vram.available_gb == pytest.approx(11.0)  # 24 - 8 - 5

    def test_disk_headroom_counts_for_pending(self):
        mgr = _make_manager()
        mgr.create(_make_request(vram_gb=1.0, ram_gb=0.5, disk_gb=50.0))
        snap = mgr.snapshot()
        assert snap.disk is not None
        assert snap.disk.reserved_headroom_gb == pytest.approx(50.0)
        assert snap.disk.available_gb == pytest.approx(350.0)


# ---------------------------------------------------------------------------
# Running reservation math (no double-count)
# ---------------------------------------------------------------------------


class TestRunningReservationMath:
    def test_running_with_actual_below_reserved_uses_headroom(self):
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001", vram_gb=8.0, ram_gb=4.0))
        # Inject actual usage: 5 GB actual, 8 GB reserved → headroom = 3
        with mgr._lock:
            mgr._reservations[res.id] = mgr._reservations[res.id].model_copy(
                update={"actual_vram_gb": 5.0}
            )
        snap = mgr.snapshot()
        assert snap.vram is not None
        assert snap.vram.reserved_headroom_gb == pytest.approx(3.0)
        # system 8 + headroom 3 = 11; 24 - 11 = 13
        assert snap.vram.available_gb == pytest.approx(13.0)

    def test_running_with_actual_ge_reserved_headroom_is_zero(self):
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001", vram_gb=4.0, ram_gb=2.0))
        with mgr._lock:
            mgr._reservations[res.id] = mgr._reservations[res.id].model_copy(
                update={"actual_vram_gb": 6.0}
            )
        snap = mgr.snapshot()
        assert snap.vram is not None
        # actual >= reserved → headroom = 0, no double-count
        assert snap.vram.reserved_headroom_gb == pytest.approx(0.0)
        assert snap.vram.available_gb == pytest.approx(16.0)

    def test_per_dimension_independence(self):
        """VRAM headroom=0 should not affect RAM headroom."""
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001", vram_gb=4.0, ram_gb=3.0))
        with mgr._lock:
            mgr._reservations[res.id] = mgr._reservations[res.id].model_copy(
                update={"actual_vram_gb": 10.0, "actual_ram_gb": 1.0}
            )
        snap = mgr.snapshot()
        assert snap.vram is not None
        assert snap.ram is not None
        assert snap.vram.reserved_headroom_gb == pytest.approx(0.0)
        assert snap.ram.reserved_headroom_gb == pytest.approx(2.0)  # 3 - 1


# ---------------------------------------------------------------------------
# Status derivation from JobStore
# ---------------------------------------------------------------------------


class TestStatusCoupling:
    def test_pending_when_no_job_in_store(self):
        mgr = _make_manager()
        res = mgr.create(_make_request())
        snap = mgr.snapshot()
        view = next(v for v in snap.reservations if v.id == res.id)
        assert view.status == "pending"

    def test_running_when_job_status_running(self):
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001"))
        snap = mgr.snapshot()
        view = next(v for v in snap.reservations if v.id == res.id)
        assert view.status == "running"

    def test_flips_to_running_when_store_updated(self):
        store = JobStore()
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001"))

        snap = mgr.snapshot()
        view = next(v for v in snap.reservations if v.id == res.id)
        assert view.status == "pending"

        store.add(_make_running_job("job-001"))
        snap2 = mgr.snapshot()
        view2 = next(v for v in snap2.reservations if v.id == res.id)
        assert view2.status == "running"


# ---------------------------------------------------------------------------
# Create: capacity enforcement
# ---------------------------------------------------------------------------


class TestCapacityEnforcement:
    def test_rejects_when_vram_would_exceed_available(self):
        mgr = _make_manager()
        # available VRAM = 16 GB; request 20 GB → should fail
        with pytest.raises(CapacityExceededError) as exc_info:
            mgr.create(_make_request(vram_gb=20.0, ram_gb=1.0))
        assert exc_info.value.dimension == "vram"
        assert exc_info.value.requested_gb == pytest.approx(20.0)

    def test_rejects_when_ram_would_exceed_available(self):
        mgr = _make_manager()
        # available RAM = 12 GB; request 15 GB → fail
        with pytest.raises(CapacityExceededError) as exc_info:
            mgr.create(_make_request(vram_gb=1.0, ram_gb=15.0))
        assert exc_info.value.dimension == "ram"

    def test_second_reservation_may_exceed_after_first(self):
        mgr = _make_manager()
        mgr.create(_make_request(job_id="job-001", vram_gb=10.0, ram_gb=1.0))
        # 16 - 10 = 6 available; request 8 should fail
        with pytest.raises(CapacityExceededError):
            mgr.create(_make_request(job_id="job-002", vram_gb=8.0, ram_gb=1.0))

    def test_disk_capacity_check(self):
        mgr = _make_manager()
        # available disk = 400 GB; request 500 GB → fail
        with pytest.raises(CapacityExceededError) as exc_info:
            mgr.create(_make_request(vram_gb=1.0, ram_gb=0.5, disk_gb=500.0))
        assert exc_info.value.dimension == "disk"


# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------


class TestRelease:
    def test_release_pending_ok(self):
        mgr = _make_manager()
        res = mgr.create(_make_request())
        mgr.release(res.id)
        snap = mgr.snapshot()
        assert not any(v.id == res.id for v in snap.reservations)

    def test_release_unknown_raises_key_error(self):
        mgr = _make_manager()
        with pytest.raises(KeyError):
            mgr.release("res-nonexistent")

    def test_release_running_raises_409_error(self):
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(_make_request(job_id="job-001"))
        with pytest.raises(ReservationRunningError):
            mgr.release(res.id)


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


class TestExpiryCleanup:
    def test_expired_non_running_reservation_removed(self):
        mgr = _make_manager()
        res = mgr.create(_make_request(ttl_seconds=60.0))
        future = datetime.now(UTC) + timedelta(seconds=120)
        removed = mgr.cleanup_expired(future)
        assert removed == 1
        snap = mgr.snapshot()
        assert not any(v.id == res.id for v in snap.reservations)

    def test_non_expired_reservation_kept(self):
        mgr = _make_manager()
        res = mgr.create(_make_request(ttl_seconds=3600.0))
        removed = mgr.cleanup_expired(datetime.now(UTC))
        assert removed == 0
        snap = mgr.snapshot()
        assert any(v.id == res.id for v in snap.reservations)

    def test_running_reservation_never_expired(self):
        store = JobStore()
        store.add(_make_running_job("job-001"))
        mgr = _make_manager(store)
        res = mgr.create(
            ReservationRequest(
                job_id="job-001",
                workload_type=WorkloadType.training,
                vram_gb=1.0,
                ram_gb=0.5,
                ttl_seconds=1.0,
            )
        )
        future = datetime.now(UTC) + timedelta(seconds=120)
        removed = mgr.cleanup_expired(future)
        assert removed == 0
        snap = mgr.snapshot()
        assert any(v.id == res.id for v in snap.reservations)

    def test_no_expiry_reservation_never_cleaned(self):
        mgr = _make_manager()
        res = mgr.create(_make_request())  # no ttl_seconds
        far_future = datetime.now(UTC) + timedelta(days=365)
        removed = mgr.cleanup_expired(far_future)
        assert removed == 0
        snap = mgr.snapshot()
        assert any(v.id == res.id for v in snap.reservations)
