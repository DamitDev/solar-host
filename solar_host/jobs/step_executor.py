"""JobStepExecutor: runs a single Docker step (create → start → stream logs → wait → cleanup)."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from solar_host.docker.errors import ContainerNonZeroExitError, ContainerStartError
from solar_host.jobs.models import JobStatus, StepStatus

if TYPE_CHECKING:
    from solar_host.config import Settings
    from solar_host.docker.service import DockerService
    from solar_host.jobs.models import StepDefinition
    from solar_host.jobs.store import JobStore

logger = logging.getLogger(__name__)


class JobStepExecutor:
    """Runs a single Docker step: create → start → stream logs → wait → cleanup.

    Instances are created by :class:`~solar_host.jobs.executor.JobExecutor` and
    share its ``active_containers`` dict and ``lock`` so that
    :meth:`~solar_host.jobs.executor.JobExecutor.cancel_job` can stop the
    currently running container.
    """

    def __init__(
        self,
        docker_service: DockerService,
        store: JobStore,
        settings: Settings,
        active_containers: dict[str, str],
        lock: threading.Lock,
    ) -> None:
        self._docker = docker_service
        self._store = store
        self._settings = settings
        self._active_containers = active_containers
        self._lock = lock

    async def run(
        self,
        job_id: str,
        step_index: int,
        step_def: StepDefinition,
        workspace_path: Path,
    ) -> bool:
        """Execute the step and return ``True`` if it failed (triggers fail-fast)."""
        step_start = datetime.now(UTC)
        self._store.update_step(
            job_id,
            step_index,
            status=StepStatus.running,
            started_at=step_start,
        )

        environment = self.build_environment(job_id, step_index, step_def, workspace_path)

        container_id: str | None = None
        try:
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

            log_path = workspace_path / "logs" / f"{step_def.name}.log"
            log_future = asyncio.get_event_loop().run_in_executor(
                None, self._stream_logs_to_file, container_id, log_path
            )

            return await self._wait_and_record(
                job_id, step_index, step_def, step_start, container_id, log_future
            )

        except ContainerStartError as exc:
            step_end = datetime.now(UTC)
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.failed,
                finished_at=step_end,
                duration_s=(step_end - step_start).total_seconds(),
                error_message=str(exc),
            )
            self._store.update(
                job_id,
                status=JobStatus.failed,
                error_message=f"Step {step_def.name!r} container start failed: {exc}",
            )
            return True

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

    async def _wait_and_record(
        self,
        job_id: str,
        step_index: int,
        step_def: StepDefinition,
        step_start: datetime,
        container_id: str,
        log_future: asyncio.Future[None],
    ) -> bool:
        """Block until the container exits and update step state. Returns True on failure."""
        try:
            await asyncio.to_thread(self._docker.wait_container, container_id)
        except ContainerNonZeroExitError as exc:
            step_end = datetime.now(UTC)
            error_msg = (
                "\n".join(exc.last_stderr_lines) if exc.last_stderr_lines else str(exc)
            )
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.failed,
                finished_at=step_end,
                duration_s=(step_end - step_start).total_seconds(),
                exit_code=exc.exit_code,
                error_message=error_msg,
            )
            self._store.update(
                job_id,
                status=JobStatus.failed,
                error_message=f"Step {step_def.name!r} failed with exit code {exc.exit_code}",
            )
            await self._drain_log_future(log_future)
            return True

        step_end = datetime.now(UTC)
        self._store.update_step(
            job_id,
            step_index,
            status=StepStatus.completed,
            finished_at=step_end,
            duration_s=(step_end - step_start).total_seconds(),
            exit_code=0,
        )
        await self._drain_log_future(log_future)
        return False

    def build_environment(
        self,
        job_id: str,
        step_index: int,
        step_def: StepDefinition,
        workspace_path: Path,
    ) -> dict[str, str]:
        """Build the full env dict for a step container (S-021 Sections 4.1–4.3)."""
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

    @staticmethod
    async def _drain_log_future(log_future: asyncio.Future[None]) -> None:
        """Wait for the log-streaming future with a short timeout (best effort)."""
        try:
            await asyncio.wait_for(log_future, timeout=5.0)
        except Exception:
            pass
