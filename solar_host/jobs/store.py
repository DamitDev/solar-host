"""Thread-safe in-memory job store with JSON file persistence.

Design
------
Every mutation (add / update / update_step / remove) is serialised to a JSON
file (*jobs_store.json*) under :attr:`store_dir` so the set of terminal jobs
survives a host restart.  The write is done under the existing thread lock
using an atomic ``tmp + os.replace`` pattern (same as
:class:`solar_host.config.ConfigManager`).

Persistence is **opt-in**: when a ``JobStore`` is created without a
*store_dir* (e.g. in unit tests) all load/save operations are no-ops.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from solar_host.jobs.models import JobState, JobStatus

logger = logging.getLogger(__name__)

_STORE_FILENAME = "jobs_store.json"


class JobStore:
    """Thread-safe in-memory store for active and recently finished jobs.

    When *store_dir* is provided all mutations are persisted to
    ``<store_dir>/jobs_store.json`` so the store survives host restarts.
    """

    def __init__(self, store_dir: str | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self.store_dir: str | None = store_dir

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @property
    def _store_path(self) -> Path | None:
        """The on-disk path for the store file, or ``None`` when disabled."""
        if self.store_dir is None:
            return None
        return Path(self.store_dir) / _STORE_FILENAME

    def load(self) -> None:
        """Load and rehydrate job states from the persisted file on disk.

        Jobs whose status was ``running`` or ``pending`` at the time of the
        last save are stale (Docker containers are gone after a host restart)
        and are automatically marked as ``failed`` so the retention cleanup
        loop can eventually reclaim their workspace directories.
        """
        path = self._store_path
        if path is None or not path.exists():
            logger.debug("No persisted job store at %s — starting fresh", path)
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            logger.exception("Failed to read persisted job store from %s", path)
            return

        now = datetime.now(UTC)
        loaded = 0
        for item in data:
            try:
                state = JobState(**item)
            except Exception:
                logger.warning("Skipping malformed job entry in %s: %s", path, item)
                continue

            # Stale non-terminal jobs — host restarted while they were running.
            if state.status in (JobStatus.pending, JobStatus.running):
                logger.info(
                    "Marking stale job %r as failed (host restarted while %s)",
                    state.job_id,
                    state.status.value,
                )
                state = state.model_copy(
                    update={
                        "status": JobStatus.failed,
                        "error_message": (
                            "Host restarted — job was in state "
                            f"{state.status.value!r} at the time"
                        ),
                        "finished_at": now,
                    }
                )

            self._jobs[state.job_id] = state
            loaded += 1

        logger.info("Loaded %d job(s) from %s", loaded, path)

    def _save_unlocked(self) -> None:
        """Atomically write the current store to a JSON file.

        Caller must hold ``_lock``.  When ``store_dir`` is not configured the
        method is a no-op (e.g. in unit tests).
        """
        path = self._store_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            data = [j.model_dump(mode="json") for j in self._jobs.values()]
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, path)
        except Exception:
            logger.exception("Failed to persist job store to %s", path)

    # ------------------------------------------------------------------
    # Write operations (all persist after mutation)
    # ------------------------------------------------------------------

    def add(self, job_state: JobState) -> None:
        """Register a new job.  Raises KeyError if job_id already exists."""
        with self._lock:
            if job_state.job_id in self._jobs:
                raise KeyError(f"Job {job_state.job_id!r} already exists in store")
            self._jobs[job_state.job_id] = job_state
            self._save_unlocked()

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
            self._save_unlocked()

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
            self._save_unlocked()

    def remove(self, job_id: str) -> None:
        """Remove a job from the store.  No-op if not found."""
        with self._lock:
            self._jobs.pop(job_id, None)
            self._save_unlocked()

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


# Module-level singleton — imported by executor, routes, and main.py.
# Persistence is enabled at startup by main.py after the jobs directory has
# been resolved (see lifespan in main.py).
job_store = JobStore()
