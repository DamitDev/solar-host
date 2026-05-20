"""JobExecutor: orchestrates sequential Docker step execution across a full job."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.models import JobState, JobStatus, StepState, StepStatus
from solar_host.jobs.step_executor import JobStepExecutor
from solar_host.jobs.workspace import (
    check_disk_space,
    create_workspace,
    validate_job_id,
)

if TYPE_CHECKING:
    from solar_host.config import Settings
    from solar_host.docker.service import DockerService
    from solar_host.jobs.models import JobDefinition, StepDefinition
    from solar_host.jobs.store import JobStore

logger = logging.getLogger(__name__)


class JobExecutor:
    """Orchestrates sequential Docker step execution across a full job.

    Delegates per-step container management to :class:`~solar_host.jobs.step_executor.JobStepExecutor`
    and focuses on job-level concerns: workspace setup, sequential step
    iteration, fail-fast propagation, and cancellation signalling.
    """

    def __init__(
        self,
        docker_service: DockerService,
        store: JobStore,
        settings: Settings,
    ) -> None:
        self._docker = docker_service
        self._store = store
        self._settings = settings

        # Per-job cancellation flags set by cancel_job().
        self._cancel_events: dict[str, asyncio.Event] = {}
        # Currently active container per job (job_id → container_id).
        # Shared by reference with _step_executor so cancel_job can stop it.
        self._active_containers: dict[str, str] = {}
        self._lock = threading.Lock()

        self._step_executor = JobStepExecutor(
            docker_service=docker_service,
            store=store,
            settings=settings,
            active_containers=self._active_containers,
            lock=self._lock,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_job(self, job_def: JobDefinition) -> JobState:
        """Execute all steps of *job_def* sequentially and return the final state.

        Raises:
            ValueError: when the job ID is invalid.
            InsufficientDiskError: when disk space is below threshold before
                workspace creation or before any individual step.
        """
        validate_job_id(job_def.job_id)

        min_gb = (
            job_def.min_free_disk_gb
            if job_def.min_free_disk_gb is not None
            else self._settings.min_free_disk_gb
        )
        check_disk_space(Path(self._settings.jobs_dir), min_gb)

        workspace_path = await asyncio.to_thread(create_workspace, job_def, self._settings)

        now = datetime.now(UTC)
        self._store.add(
            JobState(
                job_id=job_def.job_id,
                name=job_def.name,
                status=JobStatus.running,
                steps=[StepState(name=s.name) for s in job_def.steps],
                current_step_index=-1,
                workspace_path=str(workspace_path),
                created_at=now,
                started_at=now,
                retention_hours=job_def.retention_hours,
            )
        )

        cancel_event = asyncio.Event()
        with self._lock:
            self._cancel_events[job_def.job_id] = cancel_event

        failed = False
        try:
            failed = await self._run_steps(job_def, workspace_path, min_gb, cancel_event)
        except InsufficientDiskError:
            raise
        except Exception as exc:
            logger.exception("Unexpected error running job %r", job_def.job_id)
            self._store.update(
                job_def.job_id,
                status=JobStatus.failed,
                error_message=str(exc),
                finished_at=datetime.now(UTC),
            )
            raise
        finally:
            with self._lock:
                self._cancel_events.pop(job_def.job_id, None)
                self._active_containers.pop(job_def.job_id, None)

        self._finalise_job(job_def.job_id, failed, cancel_event)

        result = self._store.get(job_def.job_id)
        assert result is not None
        return result

    async def cancel_job(self, job_id: str) -> None:
        """Signal cancellation for *job_id* and stop any active container.

        Best-effort; does not wait for the executor coroutine to finish.
        """
        with self._lock:
            event = self._cancel_events.get(job_id)
            active_container = self._active_containers.get(job_id)

        if event is not None:
            event.set()

        if active_container is not None:
            logger.info(
                "Cancelling job %r — stopping container %s", job_id, active_container
            )
            try:
                await asyncio.to_thread(self._docker.stop_container, active_container)
            except Exception:
                logger.warning(
                    "Error stopping container %s during cancel of job %r",
                    active_container,
                    job_id,
                )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _run_steps(
        self,
        job_def: JobDefinition,
        workspace_path: Path,
        min_gb: float,
        cancel_event: asyncio.Event,
    ) -> bool:
        """Iterate over steps, stopping early on failure or cancellation.

        Returns ``True`` when any step failed.
        """
        for idx, step_def in enumerate(job_def.steps):
            if cancel_event.is_set():
                self._cancel_remaining_steps(job_def.job_id, idx)
                return False  # no step failed; job is being cancelled

            try:
                check_disk_space(Path(self._settings.jobs_dir), min_gb)
            except InsufficientDiskError as exc:
                error_msg = str(exc)
                self._store.update_step(
                    job_def.job_id,
                    idx,
                    status=StepStatus.failed,
                    error_message=error_msg,
                    finished_at=datetime.now(UTC),
                )
                self._cancel_remaining_steps(job_def.job_id, idx + 1)
                self._store.update(
                    job_def.job_id,
                    status=JobStatus.failed,
                    current_step_index=idx,
                    error_message=error_msg,
                    finished_at=datetime.now(UTC),
                )
                raise

            self._store.update(job_def.job_id, current_step_index=idx)
            step_failed = await self._step_executor.run(
                job_def.job_id, idx, step_def, workspace_path
            )
            if step_failed:
                self._cancel_remaining_steps(job_def.job_id, idx + 1)
                return True

        return False

    def _finalise_job(
        self, job_id: str, failed: bool, cancel_event: asyncio.Event
    ) -> None:
        """Write the terminal JobStatus when the step loop exits cleanly."""
        job = self._store.get(job_id)
        if job is None or job.status != JobStatus.running:
            return

        if cancel_event.is_set():
            terminal = JobStatus.cancelled
        elif failed:
            terminal = JobStatus.failed
        else:
            terminal = JobStatus.completed

        self._store.update(job_id, status=terminal, finished_at=datetime.now(UTC))

    def _cancel_remaining_steps(self, job_id: str, from_index: int) -> None:
        """Mark all pending steps from *from_index* onwards as cancelled."""
        job = self._store.get(job_id)
        if job is None:
            return
        for idx in range(from_index, len(job.steps)):
            if job.steps[idx].status == StepStatus.pending:
                self._store.update_step(job_id, idx, status=StepStatus.cancelled)
