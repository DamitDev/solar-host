"""Thread-safe in-memory job store."""

from __future__ import annotations

import threading
from typing import Any

from solar_host.jobs.models import JobState


class JobStore:
    """Thread-safe in-memory store for active and recently finished jobs.

    All mutations acquire a single re-entrant lock so callers never need to
    manage locking themselves.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, job_state: JobState) -> None:
        """Register a new job.  Raises KeyError if job_id already exists."""
        with self._lock:
            if job_state.job_id in self._jobs:
                raise KeyError(f"Job {job_state.job_id!r} already exists in store")
            self._jobs[job_state.job_id] = job_state

    def update(self, job_id: str, **kwargs: Any) -> None:
        """Apply partial field updates to an existing JobState.

        Only top-level scalar fields are updated (not nested step list).
        Use update_step for step-level mutations.
        Raises KeyError if the job is not found.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id!r} not found in store")
            updated = job.model_copy(update=kwargs)
            self._jobs[job_id] = updated

    def update_step(self, job_id: str, step_index: int, **kwargs: Any) -> None:
        """Apply partial field updates to a single StepState by index.

        Raises KeyError if job not found; IndexError if step_index is out of range.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"Job {job_id!r} not found in store")
            if step_index < 0 or step_index >= len(job.steps):
                raise IndexError(
                    f"Step index {step_index} out of range for job {job_id!r}"
                )
            updated_step = job.steps[step_index].model_copy(update=kwargs)
            new_steps = list(job.steps)
            new_steps[step_index] = updated_step
            self._jobs[job_id] = job.model_copy(update={"steps": new_steps})

    def remove(self, job_id: str) -> None:
        """Remove a job from the store.  No-op if not found."""
        with self._lock:
            self._jobs.pop(job_id, None)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> JobState | None:
        """Return the JobState for *job_id*, or None if not present."""
        with self._lock:
            return self._jobs.get(job_id)

    def get_all(self) -> list[JobState]:
        """Return a snapshot list of all stored JobState objects."""
        with self._lock:
            return list(self._jobs.values())


# Module-level singleton — same pattern as config_manager in config.py.
job_store = JobStore()
