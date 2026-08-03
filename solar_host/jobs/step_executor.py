"""JobStepExecutor: runs a single Docker step (create → start → stream logs → wait → cleanup)."""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from solar_host.docker.errors import (
    ContainerNonZeroExitError,
    ContainerStartError,
    GpuUnavailableError,
)
from solar_host.jobs.errors import GpuValidationError
from solar_host.jobs.events import (
    emit_step_completed,
    emit_step_failed,
    emit_step_started,
)
from solar_host.jobs.models import GpuOptions, JobStatus, StepStatus
from solar_host.jobs.step_log_buffer import step_log_buffer
from solar_host.memory_monitor import get_gpu_devices

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
        await emit_step_started(job_id, step_def.name, step_index, step_start)

        environment = self.build_environment(
            job_id, step_index, step_def, workspace_path
        )

        container_id: str | None = None
        try:
            if step_def.gpu is not None:
                self._validate_gpu_options(step_def.gpu)

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
                None,
                self._stream_logs,
                job_id,
                step_index,
                step_def.name,
                container_id,
                log_path,
            )

            return await self._wait_and_record(
                job_id, step_index, step_def, step_start, container_id, log_future
            )

        except (ContainerStartError, GpuUnavailableError, GpuValidationError) as exc:
            step_end = datetime.now(UTC)
            duration_s = (step_end - step_start).total_seconds()
            error_msg = str(exc)
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.failed,
                finished_at=step_end,
                duration_s=duration_s,
                error_message=error_msg,
            )
            self._store.update(
                job_id,
                status=JobStatus.failed,
                error_message=f"Step {step_def.name!r} container start failed: {exc}",
            )
            await emit_step_failed(
                job_id, step_def.name, step_index, step_end, duration_s, None, error_msg
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
                except Exception:  # noqa: BLE001
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
            duration_s = (step_end - step_start).total_seconds()
            error_msg = (
                "\n".join(exc.last_stderr_lines) if exc.last_stderr_lines else str(exc)
            )
            self._store.update_step(
                job_id,
                step_index,
                status=StepStatus.failed,
                finished_at=step_end,
                duration_s=duration_s,
                exit_code=exc.exit_code,
                error_message=error_msg,
            )
            self._store.update(
                job_id,
                status=JobStatus.failed,
                error_message=f"Step {step_def.name!r} failed with exit code {exc.exit_code}",
            )
            await self._drain_log_future(log_future)
            step_log_buffer.mark_completed(
                job_id, step_def.name, step_index, exit_code=exc.exit_code
            )
            await emit_step_failed(
                job_id,
                step_def.name,
                step_index,
                step_end,
                duration_s,
                exc.exit_code,
                error_msg,
            )
            return True

        step_end = datetime.now(UTC)
        duration_s = (step_end - step_start).total_seconds()
        self._store.update_step(
            job_id,
            step_index,
            status=StepStatus.completed,
            finished_at=step_end,
            duration_s=duration_s,
            exit_code=0,
        )
        await self._drain_log_future(log_future)
        step_log_buffer.mark_completed(job_id, step_def.name, step_index, exit_code=0)
        await emit_step_completed(
            job_id, step_def.name, step_index, step_end, duration_s, 0
        )
        return False

    def _validate_gpu_options(self, gpu: GpuOptions) -> None:
        """Validate GPU options against host inventory before container creation.

        Raises GpuValidationError if requested devices exceed the available inventory.
        Skips validation silently when pynvml inventory is unavailable.
        """
        devices = get_gpu_devices()
        if not devices:
            return
        available_count = len(devices)
        if gpu.device_ids is not None:
            available_uuids = {d["uuid"] for d in devices}
            available_indices = {str(d["index"]) for d in devices}
            for dev_id in gpu.device_ids:
                if dev_id not in available_uuids and dev_id not in available_indices:
                    raise GpuValidationError(dev_id, available_count)
        elif gpu.count is not None and gpu.count != -1:
            if gpu.count > available_count:
                raise GpuValidationError(f"count={gpu.count}", available_count)

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

        # Section 4.2.5: NVIDIA env vars when GPU is requested (callers may override).
        if step_def.gpu is not None:
            env.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
            env.setdefault("NVIDIA_DRIVER_CAPABILITIES", "compute,utility")

        # Section 4.3: Step-specific variables (override infra vars if clashing).
        env.update(step_def.environment)

        return env

    def _stream_logs(
        self,
        job_id: str,
        step_index: int,
        step_name: str,
        container_id: str,
        log_path: Path,
    ) -> None:
        """Stream container logs to disk and the step log buffer (thread-safe).

        Dual-writes each chunk to the combined host log file and enqueues it
        in the per-step buffer for real-time Socket.IO emission.  Errors are
        swallowed so that a streaming failure never fails the step.
        """
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                for stream_tag, chunk in self._docker.stream_logs(
                    container_id, follow=True, tail=0, demux=True
                ):
                    fh.write(chunk)
                    if not chunk.endswith("\n"):
                        fh.write("\n")
                    step_log_buffer.append(
                        job_id, step_name, step_index, stream_tag, chunk
                    )
        except Exception:  # noqa: BLE001
            logger.debug(
                "Log streaming finished for container %s → %s", container_id, log_path
            )

    @staticmethod
    async def _drain_log_future(log_future: asyncio.Future[None]) -> None:
        """Wait for the log-streaming future with a short timeout (best effort)."""
        try:
            await asyncio.wait_for(log_future, timeout=5.0)
        except Exception:  # noqa: S110, BLE001
            pass
