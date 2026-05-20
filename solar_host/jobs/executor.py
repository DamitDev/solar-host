"""Async JobExecutor: sequential Docker step execution with fail-fast and cancellation."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from solar_host.docker.errors import ContainerNonZeroExitError, ContainerStartError
from solar_host.jobs.errors import InsufficientDiskError
from solar_host.jobs.models import JobState, JobStatus, StepState, StepStatus
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
    """Orchestrates sequential Docker step execution for a job definition.

    Designed to be called via ``asyncio`` and uses ``asyncio.to_thread`` for
    all blocking Docker calls so the event loop is never stalled.
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
        self._active_containers: dict[str, str] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_job(self, job_def: JobDefinition) -> JobState:
        """Execute all steps of *job_def* sequentially and return final state.

        Raises:
            ValueError: when the job ID is invalid.
            InsufficientDiskError: when disk space is below threshold before
                workspace creation.
        """
        # 1. Validate job ID.
        validate_job_id(job_def.job_id)

        # 2. Check disk space before allocating workspace.
        min_gb = (
            job_def.min_free_disk_gb
            if job_def.min_free_disk_gb is not None
            else self._settings.min_free_disk_gb
        )
        check_disk_space(Path(self._settings.jobs_dir), min_gb)

        # 3. Create workspace.
        workspace_path = await asyncio.to_thread(create_workspace, job_def, self._settings)

        # 4. Register initial JobState in store.
        now = datetime.now(UTC)
        job_state = JobState(
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
        self._store.add(job_state)

        # Prepare cancellation event for this job.
        cancel_event = asyncio.Event()
        with self._lock:
            self._cancel_events[job_def.job_id] = cancel_event

        # 5. Loop over steps sequentially.
        failed = False
        try:
            for idx, step_def in enumerate(job_def.steps):
                # Check cancellation before starting each step.
                if cancel_event.is_set():
                    self._cancel_remaining_steps(job_def.job_id, idx)
                    break

                # Pre-step disk check.
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
                step_failed = await self._run_step(
                    job_def.job_id, idx, step_def, workspace_path, cancel_event
                )
                if step_failed:
                    failed = True
                    # Remaining steps are already marked cancelled/skipped inside _run_step.
                    self._cancel_remaining_steps(job_def.job_id, idx + 1)
                    break

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

        # Determine terminal status if not already set.
        final = self._store.get(job_def.job_id)
        if final is not None and final.status == JobStatus.running:
            if cancel_event.is_set():
                terminal_status = JobStatus.cancelled
            elif failed:
                terminal_status = JobStatus.failed
            else:
                terminal_status = JobStatus.completed
            self._store.update(
                job_def.job_id,
                status=terminal_status,
                finished_at=datetime.now(UTC),
            )

        result = self._store.get(job_def.job_id)
        assert result is not None  # always registered above
        return result

    async def cancel_job(self, job_id: str) -> None:
        """Signal cancellation for *job_id* and stop any active container.

        This is a best-effort operation; it does not wait for the executor
        coroutine to finish.
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

    async def _run_step(
        self,
        job_id: str,
        step_index: int,
        step_def: StepDefinition,
        workspace_path: Path,
        cancel_event: asyncio.Event,
    ) -> bool:
        """Execute a single step.  Returns True if the step failed (fail-fast)."""
        step_start = datetime.now(UTC)
        self._store.update_step(
            job_id,
            step_index,
            status=StepStatus.running,
            started_at=step_start,
        )

        environment = self._build_step_environment(
            job_id, step_index, step_def, workspace_path
        )

        container_id: str | None = None
        try:
            # create_container may raise ContainerStartError on API error.
            container_id = await asyncio.to_thread(
                self._docker.create_container,
                step_def.image,
                job_id,
                step_def.name,
                environment,
                step_def.gpu,
                step_def.is_preparation_step,
            )
            self._store.update_step(job_id, step_index, container_id=container_id)

            with self._lock:
                self._active_containers[job_id] = container_id

            await asyncio.to_thread(self._docker.start_container, container_id)

            # Stream logs to file concurrently while wait_container blocks.
            log_path = workspace_path / "logs" / f"{step_def.name}.log"
            log_future = asyncio.get_event_loop().run_in_executor(
                None, self._stream_logs_to_file, container_id, log_path
            )

            try:
                await asyncio.to_thread(self._docker.wait_container, container_id)
            except ContainerNonZeroExitError as exc:
                step_end = datetime.now(UTC)
                duration = (step_end - step_start).total_seconds()
                error_msg = "\n".join(exc.last_stderr_lines) if exc.last_stderr_lines else str(exc)
                self._store.update_step(
                    job_id,
                    step_index,
                    status=StepStatus.failed,
                    finished_at=step_end,
                    duration_s=duration,
                    exit_code=exc.exit_code,
                    error_message=error_msg,
                )
                self._store.update(
                    job_id,
                    status=JobStatus.failed,
                    error_message=f"Step {step_def.name!r} failed with exit code {exc.exit_code}",
                )
                # Wait for log streaming to finish (best effort).
                try:
                    await asyncio.wait_for(log_future, timeout=5.0)
                except Exception:
                    pass
                return True  # fail-fast

            step_end = datetime.now(UTC)
            duration = (step_end - step_start).total_seconds()
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.completed,
                finished_at=step_end,
                duration_s=duration,
                exit_code=0,
            )
            try:
                await asyncio.wait_for(log_future, timeout=5.0)
            except Exception:
                pass
            return False  # success

        except ContainerStartError as exc:
            step_end = datetime.now(UTC)
            duration = (step_end - step_start).total_seconds()
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.failed,
                finished_at=step_end,
                duration_s=duration,
                error_message=str(exc),
            )
            self._store.update(
                job_id,
                status=JobStatus.failed,
                error_message=f"Step {step_def.name!r} container start failed: {exc}",
            )
            return True  # fail-fast

        finally:
            with self._lock:
                if self._active_containers.get(job_id) == container_id:
                    self._active_containers.pop(job_id, None)
            if container_id is not None:
                try:
                    await asyncio.to_thread(
                        self._docker.remove_container, container_id, True
                    )
                except Exception:
                    logger.warning(
                        "Failed to remove container %s for step %r in job %r",
                        container_id,
                        step_def.name,
                        job_id,
                    )

    def _stream_logs_to_file(self, container_id: str, log_path: Path) -> None:
        """Synchronous log streaming from a container, written to *log_path*."""
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                for line in self._docker.stream_logs(container_id, follow=True, tail=0):
                    fh.write(line)
                    if not line.endswith("\n"):
                        fh.write("\n")
        except Exception:
            logger.warning(
                "Log streaming failed for container %s → %s", container_id, log_path
            )

    def _cancel_remaining_steps(self, job_id: str, from_index: int) -> None:
        """Mark steps from *from_index* onwards as cancelled in the store."""
        job = self._store.get(job_id)
        if job is None:
            return
        for idx in range(from_index, len(job.steps)):
            step = job.steps[idx]
            if step.status == StepStatus.pending:
                self._store.update_step(job_id, idx, status=StepStatus.cancelled)

    def _build_step_environment(
        self,
        job_id: str,
        step_index: int,
        step_def: StepDefinition,
        workspace_path: Path,
    ) -> dict[str, str]:
        """Build the full environment dict for a step container (S-021 Sections 4.1-4.3)."""
        s = self._settings

        # Section 4.1: Workspace path variables (container-side paths).
        env: dict[str, str] = {
            "JOB_ID": job_id,
            "WORKSPACE_MODELS": "/workspace/models",
            "WORKSPACE_DATA": "/workspace/data",
            "WORKSPACE_OUTPUT": "/workspace/output",
            "WORKSPACE_CONFIG": "/workspace/config",
            "JOB_CONFIG": "/workspace/config/job.json",
            "STEP_NAME": step_def.name,
            "STEP_INDEX": str(step_index),
        }

        # Section 4.2: Infrastructure credentials from Settings.
        env["HARBOR_URL"] = s.harbor_url
        env["HARBOR_USERNAME"] = s.harbor_username
        env["HARBOR_PASSWORD"] = s.harbor_password
        env["HF_TOKEN"] = s.hf_token
        env["HF_HOME"] = "/workspace/.cache/huggingface"

        # Section 4.3: Step-specific variables (override infra vars if clashing).
        env.update(step_def.environment)

        return env
