"""Unit tests for solar_host.jobs.store.JobStore."""

from __future__ import annotations

import threading
from datetime import UTC, datetime

import pytest

from solar_host.jobs.models import JobState, JobStatus, StepState, StepStatus
from solar_host.jobs.store import JobStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(job_id: str = "job-1", name: str = "Test Job") -> JobState:
    return JobState(job_id=job_id, name=name)


def _make_job_with_steps(job_id: str = "job-2") -> JobState:
    steps = [
        StepState(name="prepare"),
        StepState(name="train"),
        StepState(name="export"),
    ]
    return JobState(job_id=job_id, name="Multi-step", steps=steps)


# ---------------------------------------------------------------------------
# add / get
# ---------------------------------------------------------------------------


def test_add_and_get():
    store = JobStore()
    job = _make_job()
    store.add(job)
    retrieved = store.get("job-1")
    assert retrieved is not None
    assert retrieved.job_id == "job-1"


def test_get_returns_none_for_missing():
    store = JobStore()
    assert store.get("nonexistent") is None


def test_add_duplicate_raises():
    store = JobStore()
    store.add(_make_job("dup"))
    with pytest.raises(KeyError, match="dup"):
        store.add(_make_job("dup"))


# ---------------------------------------------------------------------------
# get_all
# ---------------------------------------------------------------------------


def test_get_all_empty():
    store = JobStore()
    assert store.get_all() == []


def test_get_all_returns_all():
    store = JobStore()
    store.add(_make_job("a"))
    store.add(_make_job("b"))
    store.add(_make_job("c"))
    ids = {j.job_id for j in store.get_all()}
    assert ids == {"a", "b", "c"}


def test_get_all_returns_snapshot():
    """Mutating the returned list must not affect the store."""
    store = JobStore()
    store.add(_make_job("snap"))
    snapshot = store.get_all()
    snapshot.clear()
    assert len(store.get_all()) == 1


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


def test_update_status():
    store = JobStore()
    store.add(_make_job("upd"))
    store.update("upd", status=JobStatus.running)
    assert store.get("upd").status == JobStatus.running  # type: ignore[union-attr]


def test_update_multiple_fields():
    store = JobStore()
    store.add(_make_job("multi"))
    now = datetime.now(UTC)
    store.update("multi", status=JobStatus.completed, finished_at=now)
    job = store.get("multi")
    assert job is not None
    assert job.status == JobStatus.completed
    assert job.finished_at == now


def test_update_missing_job_raises():
    store = JobStore()
    with pytest.raises(KeyError, match="ghost"):
        store.update("ghost", status=JobStatus.failed)


def test_update_does_not_mutate_original():
    """model_copy must produce a new object; the original should be gone."""
    store = JobStore()
    store.add(_make_job("copy-test"))
    store.update("copy-test", status=JobStatus.running)
    job = store.get("copy-test")
    assert job is not None
    assert job.status == JobStatus.running


# ---------------------------------------------------------------------------
# update_step
# ---------------------------------------------------------------------------


def test_update_step_status():
    store = JobStore()
    store.add(_make_job_with_steps("steps-1"))
    store.update_step("steps-1", 1, status=StepStatus.running)
    job = store.get("steps-1")
    assert job is not None
    assert job.steps[1].status == StepStatus.running
    # Other steps untouched
    assert job.steps[0].status == StepStatus.pending
    assert job.steps[2].status == StepStatus.pending


def test_update_step_multiple_fields():
    store = JobStore()
    store.add(_make_job_with_steps("steps-2"))
    now = datetime.now(UTC)
    store.update_step(
        "steps-2", 0, status=StepStatus.completed, started_at=now, exit_code=0
    )
    step = store.get("steps-2").steps[0]  # type: ignore[union-attr]
    assert step.status == StepStatus.completed
    assert step.started_at == now
    assert step.exit_code == 0


def test_update_step_missing_job_raises():
    store = JobStore()
    with pytest.raises(KeyError, match="no-job"):
        store.update_step("no-job", 0, status=StepStatus.running)


def test_update_step_out_of_range_raises():
    store = JobStore()
    store.add(_make_job_with_steps("oob"))
    with pytest.raises(IndexError):
        store.update_step("oob", 99, status=StepStatus.running)


def test_update_step_negative_index_raises():
    store = JobStore()
    store.add(_make_job_with_steps("neg"))
    with pytest.raises(IndexError):
        store.update_step("neg", -1, status=StepStatus.running)


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_existing():
    store = JobStore()
    store.add(_make_job("rm"))
    store.remove("rm")
    assert store.get("rm") is None


def test_remove_missing_is_noop():
    store = JobStore()
    store.remove("not-there")  # Must not raise


def test_remove_leaves_others_intact():
    store = JobStore()
    store.add(_make_job("keep"))
    store.add(_make_job("drop"))
    store.remove("drop")
    assert store.get("keep") is not None
    assert store.get("drop") is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_add_is_safe():
    """Multiple threads adding distinct jobs must all succeed."""
    store = JobStore()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            store.add(_make_job(f"concurrent-{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(store.get_all()) == 50


def test_concurrent_update_is_safe():
    """Multiple threads updating the same job must not raise or corrupt state."""
    store = JobStore()
    store.add(_make_job("shared"))
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            store.update("shared", error_message=f"msg-{i}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    job = store.get("shared")
    assert job is not None
    assert job.error_message is not None


def test_concurrent_mixed_operations():
    """Interleaved add/update/remove/get_all must not deadlock or raise."""
    store = JobStore()
    errors: list[Exception] = []

    def adder(i: int) -> None:
        try:
            store.add(_make_job(f"mix-{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader() -> None:
        try:
            store.get_all()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=adder, args=(i,)) for i in range(20)] + [
        threading.Thread(target=reader) for _ in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_singleton_is_job_store_instance():
    from solar_host.jobs.store import job_store

    assert isinstance(job_store, JobStore)
