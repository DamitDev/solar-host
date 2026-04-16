"""Process manager for solar-host with multi-backend support."""

import logging
import subprocess
import socket
import time
import uuid
import queue
import shutil
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import deque
import asyncio
import threading

from solar_host.models import (
    Instance,
    InstanceStatus,
    LogMessage,
    InstanceRuntimeState,
    InstanceStateEvent,
    GenerationMetrics,
    BackendType,
)
from solar_host.config import settings, config_manager, parse_instance_config
from solar_host.backends.base import BackendRunner
from solar_host.backends.llamacpp import LlamaCppRunner
from solar_host.backends.huggingface import HuggingFaceRunner
from solar_host.ws_client import (
    get_clients,
    broadcast_log_batch,
    broadcast_instance_state_batch,
    broadcast_instances_update,
)

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_S = 0.1
WATCHDOG_INTERVAL_S = 15.0
MAX_QUEUE_SIZE = 10_000
_HAS_STDBUF = shutil.which("stdbuf") is not None


def get_runner_for_config(config) -> BackendRunner:
    """Get the appropriate backend runner for a config type."""
    backend_type = getattr(config, "backend_type", "llamacpp")

    if backend_type == BackendType.LLAMACPP or backend_type == "llamacpp":
        return LlamaCppRunner()
    elif backend_type in (
        BackendType.HUGGINGFACE_CAUSAL,
        BackendType.HUGGINGFACE_CLASSIFICATION,
        BackendType.HUGGINGFACE_EMBEDDING,
        "huggingface_causal",
        "huggingface_classification",
        "huggingface_embedding",
    ):
        return HuggingFaceRunner()
    else:
        # Default to llama.cpp for backward compatibility
        return LlamaCppRunner()


class ProcessManager:
    """Manages model server processes across multiple backends."""

    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.log_buffers: Dict[str, deque] = {}
        self.log_sequences: Dict[str, int] = {}
        self.log_threads: Dict[str, threading.Thread] = {}
        self.log_dir = Path(settings.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Runtime state streaming (ephemeral)
        self.state_buffers: Dict[str, deque] = {}
        self.state_sequences: Dict[str, int] = {}

        # Per-instance parsing context (managed by backend runners)
        self.instance_contexts: Dict[str, Dict[str, Any]] = {}

        # Per-instance runner reference
        self.instance_runners: Dict[str, BackendRunner] = {}

        # Batched emission queues (thread-safe, drained by _flush_loop).
        # Bounded to prevent OOM if the flush loop or WS broadcast stalls.
        self._log_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._state_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._flush_task: Optional[asyncio.Task] = None
        self._child_exit_lock = threading.Lock()

    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available (not bound by any process)."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    def _get_assigned_ports(self) -> set:
        """Get ports assigned to currently running instances only."""
        assigned = set()
        for instance in config_manager.get_all_instances():
            if instance.port is not None and instance.status == InstanceStatus.RUNNING:
                assigned.add(instance.port)
        return assigned

    def _get_available_port(self) -> int:
        """Get the lowest available port starting from settings.start_port.

        Finds the first port (starting from start_port) that is:
        1. Not assigned to a currently running instance
        2. Not currently bound by any process
        """
        assigned_ports = self._get_assigned_ports()
        port = settings.start_port

        while port in assigned_ports or not self._is_port_available(port):
            port += 1

        return port

    def _purge_instance_resources(
        self, instance_id: str, *, call_runner_on_stop: bool = True
    ) -> None:
        """Remove in-memory runners, buffers, and threads for an instance."""
        runner = self.instance_runners.get(instance_id)
        if runner and call_runner_on_stop:
            context = self.instance_contexts.get(instance_id, {})
            runner.on_process_stopped(instance_id, context)
        self.instance_runners.pop(instance_id, None)
        self.instance_contexts.pop(instance_id, None)
        self.log_buffers.pop(instance_id, None)
        self.log_sequences.pop(instance_id, None)
        self.log_threads.pop(instance_id, None)
        self.state_buffers.pop(instance_id, None)
        self.state_sequences.pop(instance_id, None)

    def _handle_child_exit(self, instance_id: str, process: subprocess.Popen) -> None:
        """Mark instance FAILED when the child process exits unexpectedly.

        Idempotent: safe from log thread EOF and watchdog; skips intentional stops.
        """
        with self._child_exit_lock:
            instance = config_manager.get_instance(instance_id)
            if not instance:
                self.processes.pop(instance_id, None)
                return

            if instance.status not in (
                InstanceStatus.RUNNING,
                InstanceStatus.STARTING,
            ):
                self.processes.pop(instance_id, None)
                return

            tracked = self.processes.get(instance_id)
            if tracked is not process:
                return

            exit_code = process.poll()
            if exit_code is None:
                return

            del self.processes[instance_id]

            instance.status = InstanceStatus.FAILED
            instance.error_message = (
                f"Process exited unexpectedly (exit code: {exit_code})"
            )
            instance.pid = None
            instance.started_at = None
            config_manager.update_instance(instance_id, instance)

            self._purge_instance_resources(instance_id, call_runner_on_stop=True)
            self._push_instances_update()

    def _read_logs(
        self,
        instance_id: str,
        process: subprocess.Popen,
        log_file: Path,
        runner: BackendRunner,
    ):
        """Read logs from process and store in buffer."""
        if not process.stdout:
            return

        try:
            with open(log_file, "a") as f:
                for line in iter(process.stdout.readline, b""):
                    if not line:
                        break

                    decoded_line = line.decode("utf-8", errors="replace").rstrip()

                    # Write to file
                    f.write(decoded_line + "\n")
                    f.flush()

                    # Store in buffer
                    if instance_id not in self.log_buffers:
                        self.log_buffers[instance_id] = deque(
                            maxlen=settings.log_buffer_size
                        )
                        self.log_sequences[instance_id] = 0

                    seq = self.log_sequences[instance_id]
                    self.log_sequences[instance_id] += 1

                    timestamp = datetime.now(timezone.utc).isoformat()
                    log_msg = LogMessage(
                        seq=seq, timestamp=timestamp, line=decoded_line
                    )
                    self.log_buffers[instance_id].append(log_msg)

                    # Push log to solar-control via WebSocket
                    self._push_log_event(instance_id, seq, decoded_line, timestamp)

                    # Parse log line using backend runner
                    try:
                        context = self.instance_contexts.get(instance_id, {})
                        state_update = runner.parse_log_line(
                            instance_id, decoded_line, context
                        )
                        if state_update:
                            self._emit_state_event(instance_id, state_update)
                    except Exception:
                        # Parsing errors should not break logging
                        pass
        except Exception as e:
            logger.warning("Error reading logs for %s: %s", instance_id, e)
        finally:
            # stdout closed or reader error: child likely exited — reconcile state
            self._handle_child_exit(instance_id, process)

    def ensure_flush_loop(self, loop: asyncio.AbstractEventLoop):
        """Start the batched emission flush loop on the given event loop.

        Called once from main.py after the event loop is running.
        """
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.run_coroutine_threadsafe(
                self._flush_loop(), loop
            )

    async def _flush_loop(self):
        """Periodically drain queued log/state events and emit as batches."""
        while True:
            try:
                await asyncio.sleep(FLUSH_INTERVAL_S)
                await self._flush_pending()
            except asyncio.CancelledError:
                await self._flush_pending()
                break
            except Exception:
                logger.exception("Error in flush loop")
                await asyncio.sleep(1)

    async def _flush_pending(self):
        """Drain both queues and emit batched events."""
        log_entries: List[dict] = []
        while True:
            try:
                log_entries.append(self._log_queue.get_nowait())
            except queue.Empty:
                break

        latest_states: Dict[str, dict] = {}
        while True:
            try:
                entry = self._state_queue.get_nowait()
                latest_states[entry["instance_id"]] = entry
            except queue.Empty:
                break

        if log_entries:
            await broadcast_log_batch(log_entries)

        if latest_states:
            await broadcast_instance_state_batch(list(latest_states.values()))

    async def watchdog_loop(self):
        """Periodically poll child processes; mark FAILED if any exited unexpectedly."""
        while True:
            try:
                await asyncio.sleep(WATCHDOG_INTERVAL_S)
                for instance_id, process in list(self.processes.items()):
                    if process.poll() is not None:
                        await asyncio.to_thread(
                            self._handle_child_exit, instance_id, process
                        )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in watchdog loop")
                await asyncio.sleep(1)

    def _push_log_event(self, instance_id: str, seq: int, line: str, timestamp: str):
        """Queue a log event for batched emission (thread-safe, non-blocking).

        Silently discards events when the queue is full to prevent OOM.
        """
        try:
            self._log_queue.put_nowait(
                {
                    "instance_id": instance_id,
                    "seq": seq,
                    "line": line,
                    "timestamp": timestamp,
                }
            )
        except queue.Full:
            pass

    def _emit_state_event(self, instance_id: str, update):
        """Emit a state event from a RuntimeStateUpdate."""
        # Update in-memory instance runtime fields
        config_manager.update_instance_runtime(
            instance_id,
            busy=update.busy,
            prefill_progress=update.prefill_progress,
            active_slots=update.active_slots,
        )

        # Initialize state buffer/seq lazily
        if instance_id not in self.state_buffers:
            self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
            self.state_sequences[instance_id] = 0

        seq = self.state_sequences[instance_id]
        self.state_sequences[instance_id] += 1

        now_ts = datetime.now(timezone.utc).isoformat()
        state = InstanceRuntimeState(
            instance_id=instance_id,
            busy=update.busy,
            phase=update.phase,
            prefill_progress=update.prefill_progress,
            active_slots=update.active_slots,
            slot_id=update.slot_id,
            task_id=update.task_id,
            prefill_prompt_tokens=update.prefill_prompt_tokens,
            generated_tokens=update.generated_tokens,
            decode_tps=update.decode_tps,
            decode_ms_per_token=update.decode_ms_per_token,
            checkpoint_index=update.checkpoint_index,
            checkpoint_total=update.checkpoint_total,
            timestamp=now_ts,
        )
        event = InstanceStateEvent(
            seq=seq,
            timestamp=now_ts,
            data=state,
        )
        self.state_buffers[instance_id].append(event)

        # Push state to solar-control via WebSocket
        self._push_state_event(instance_id, state)

    def _push_state_event(self, instance_id: str, state: InstanceRuntimeState):
        """Queue a state event for batched emission (thread-safe, non-blocking).

        Silently discards events when the queue is full to prevent OOM.
        """
        try:
            self._state_queue.put_nowait(
                {
                    "instance_id": instance_id,
                    "timestamp": state.timestamp,
                    "data": {
                        "busy": state.busy,
                        "phase": state.phase.value if state.phase else None,
                        "prefill_progress": state.prefill_progress,
                        "active_slots": state.active_slots,
                        "slot_id": state.slot_id,
                        "task_id": state.task_id,
                        "prefill_prompt_tokens": state.prefill_prompt_tokens,
                        "generated_tokens": state.generated_tokens,
                        "decode_tps": state.decode_tps,
                        "decode_ms_per_token": state.decode_ms_per_token,
                        "checkpoint_index": state.checkpoint_index,
                        "checkpoint_total": state.checkpoint_total,
                    },
                }
            )
        except queue.Full:
            pass

    def get_last_generation(self, instance_id: str) -> Optional[GenerationMetrics]:
        """Get the last generation metrics for an instance."""
        runner = self.instance_runners.get(instance_id)
        context = self.instance_contexts.get(instance_id, {})

        if runner and hasattr(runner, "get_last_generation"):
            return runner.get_last_generation(context)
        return None

    async def start_instance(self, instance_id: str) -> bool:
        """Start a model server instance with iterative retry."""
        for attempt in range(1 + settings.max_retries):
            result = await self._try_start_instance(instance_id, attempt)
            if result is not None:
                return result
            # result is None means "retry"
            await asyncio.sleep(1)
        return False

    async def _try_start_instance(self, instance_id: str, attempt: int) -> Optional[bool]:
        """Single start attempt. Returns True/False for final result, None to retry."""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False

        # Check if already running (verify subprocess is still alive)
        if instance.status == InstanceStatus.RUNNING:
            proc = self.processes.get(instance_id)
            if proc and proc.poll() is None:
                return True
            # Stale RUNNING: process missing or dead -- reset and continue start
            if proc is not None:
                self.processes.pop(instance_id, None)
            self._purge_instance_resources(instance_id, call_runner_on_stop=True)
            instance.status = InstanceStatus.STOPPED
            instance.pid = None
            instance.started_at = None
            instance.error_message = None
            config_manager.update_instance(instance_id, instance)

        instance.port = self._get_available_port()

        runner = get_runner_for_config(instance.config)
        self.instance_runners[instance_id] = runner
        self.instance_contexts[instance_id] = runner.initialize_context()

        instance.status = InstanceStatus.STARTING
        instance.error_message = None

        if hasattr(runner, "get_supported_endpoints_for_model_type"):
            model_type = getattr(instance.config, "model_type", "llm")
            instance.supported_endpoints = (
                runner.get_supported_endpoints_for_model_type(model_type)
            )
        elif hasattr(runner, "get_supported_endpoints_for_type"):
            backend_type = getattr(instance.config, "backend_type", "llamacpp")
            instance.supported_endpoints = runner.get_supported_endpoints_for_type(
                backend_type
            )
        else:
            instance.supported_endpoints = runner.get_supported_endpoints()

        config_manager.update_instance(instance_id, instance)

        try:
            cmd = runner.build_command(instance)

            alias_safe = instance.config.alias.replace(":", "-").replace("/", "-")
            log_file = self.log_dir / f"{alias_safe}_{int(time.time())}.log"

            run_env = os.environ.copy()
            run_env["PYTHONUNBUFFERED"] = "1"
            run_cmd = ["stdbuf", "-oL"] + cmd if _HAS_STDBUF else cmd

            process = subprocess.Popen(
                run_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                env=run_env,
            )

            self.processes[instance_id] = process

            log_thread = threading.Thread(
                target=self._read_logs,
                args=(instance_id, process, log_file, runner),
                daemon=True,
            )
            log_thread.start()
            self.log_threads[instance_id] = log_thread

            await asyncio.sleep(2)

            instance = config_manager.get_instance(instance_id)
            if not instance or instance.status not in (
                InstanceStatus.STARTING,
                InstanceStatus.RUNNING,
            ):
                return (
                    instance is not None and instance.status == InstanceStatus.RUNNING
                )

            if process.poll() is None:
                instance.status = InstanceStatus.RUNNING
                instance.pid = process.pid
                instance.started_at = datetime.now(timezone.utc)
                instance.retry_count = 0
                config_manager.update_instance(instance_id, instance)

                self.state_buffers[instance_id] = deque(maxlen=settings.log_buffer_size)
                self.state_sequences[instance_id] = 0
                config_manager.update_instance_runtime(
                    instance_id, busy=False, prefill_progress=None, active_slots=0
                )

                runner.on_process_started(
                    instance_id, self.instance_contexts[instance_id]
                )

                self._push_instances_update()

                return True
            else:
                instance.status = InstanceStatus.FAILED
                instance.error_message = "Process exited immediately"
                instance.retry_count = attempt + 1
                config_manager.update_instance(instance_id, instance)

                if instance.retry_count < settings.max_retries:
                    return None  # signal retry
                return False

        except Exception as e:
            instance = config_manager.get_instance(instance_id)
            if instance:
                instance.status = InstanceStatus.FAILED
                instance.error_message = str(e)
                instance.retry_count = attempt + 1
                config_manager.update_instance(instance_id, instance)

                if instance.retry_count < settings.max_retries:
                    return None  # signal retry
            return False

    async def stop_instance(self, instance_id: str) -> bool:
        """Stop a model server instance."""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False

        if instance.status == InstanceStatus.STOPPED:
            return True

        # For failed instances, just transition to stopped cleanly
        if instance.status == InstanceStatus.FAILED:
            instance.status = InstanceStatus.STOPPED
            instance.pid = None
            instance.started_at = None
            instance.error_message = None
            config_manager.update_instance(instance_id, instance)
            self._push_instances_update()
            return True

        instance.status = InstanceStatus.STOPPING
        config_manager.update_instance(instance_id, instance)

        try:
            if instance_id in self.processes:
                process = self.processes[instance_id]
                process.terminate()

                try:
                    await asyncio.to_thread(process.wait, 10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    await asyncio.to_thread(process.wait)

                self.processes.pop(instance_id, None)

            # Wait for log thread to finish before purging resources
            log_thread = self.log_threads.get(instance_id)
            if log_thread and log_thread.is_alive():
                log_thread.join(timeout=5)

            self._purge_instance_resources(instance_id, call_runner_on_stop=True)

            instance.status = InstanceStatus.STOPPED
            instance.pid = None
            instance.started_at = None
            config_manager.update_instance(instance_id, instance)

            await self._cleanup_old_logs(instance.config.alias)

            self._push_instances_update()

            return True

        except Exception as e:
            instance.status = InstanceStatus.FAILED
            instance.error_message = f"Failed to stop: {str(e)}"
            config_manager.update_instance(instance_id, instance)
            return False

    async def _cleanup_old_logs(self, alias: str):
        """Clean up old log files for stopped instances."""
        try:
            alias_safe = alias.replace(":", "-").replace("/", "-")
            pattern = f"{alias_safe}_*.log"
            for log_file in self.log_dir.glob(pattern):
                # Keep only the most recent log
                if log_file.stat().st_mtime < time.time() - 300:  # 5 minutes old
                    log_file.unlink()
        except Exception as e:
            logger.warning("Error cleaning up logs: %s", e)

    async def restart_instance(self, instance_id: str) -> bool:
        """Restart a model server instance."""
        stopped = await self.stop_instance(instance_id)
        if not stopped:
            instance = config_manager.get_instance(instance_id)
            if instance and instance.status in (
                InstanceStatus.RUNNING,
                InstanceStatus.STARTING,
            ):
                logger.error("Cannot restart %s: stop failed and instance is still active", instance_id)
                return False
        await asyncio.sleep(1)
        return await self.start_instance(instance_id)

    def create_instance(self, config) -> Instance:
        """Create a new instance."""
        # Parse config if it's a dict (from FastAPI request body)
        if isinstance(config, dict):
            config = parse_instance_config(config)

        instance_id = str(uuid.uuid4())

        runner = get_runner_for_config(config)

        if hasattr(runner, "get_supported_endpoints_for_model_type"):
            model_type = getattr(config, "model_type", "llm")
            supported_endpoints = runner.get_supported_endpoints_for_model_type(
                model_type
            )
        elif hasattr(runner, "get_supported_endpoints_for_type"):
            backend_type = getattr(config, "backend_type", "llamacpp")
            supported_endpoints = runner.get_supported_endpoints_for_type(backend_type)
        else:
            supported_endpoints = runner.get_supported_endpoints()

        instance = Instance(
            id=instance_id,
            config=config,
            status=InstanceStatus.STOPPED,
            supported_endpoints=supported_endpoints,
        )
        config_manager.add_instance(instance)

        # Notify solar-control of instance update
        self._push_instances_update()

        return instance

    def get_log_buffer(self, instance_id: str) -> List[LogMessage]:
        """Get log buffer for an instance."""
        if instance_id in self.log_buffers:
            return list(self.log_buffers[instance_id])
        return []

    def get_next_sequence(self, instance_id: str) -> int:
        """Get next sequence number for an instance."""
        return self.log_sequences.get(instance_id, 0)

    def get_state_buffer(self, instance_id: str) -> List[InstanceStateEvent]:
        """Get state buffer for an instance."""
        if instance_id in self.state_buffers:
            return list(self.state_buffers[instance_id])
        return []

    def get_state_next_sequence(self, instance_id: str) -> int:
        """Get next state sequence number for an instance."""
        return self.state_sequences.get(instance_id, 0)

    def _push_instances_update(self):
        """Push instance list update to all connected solar-controls (thread-safe)."""
        try:
            clients = get_clients()
            for client in clients:
                if client.is_connected:
                    loop = getattr(client, "_main_loop", None)
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            broadcast_instances_update(), loop
                        )
                        break
        except Exception:
            logger.debug("Failed to push instances update to solar-control", exc_info=True)

    def delete_instance(self, instance_id: str) -> bool:
        """Delete an instance and notify solar-control."""
        instance = config_manager.get_instance(instance_id)
        if not instance:
            return False

        # Defensively terminate any lingering process
        process = self.processes.pop(instance_id, None)
        if process and process.poll() is None:
            logger.warning("Terminating orphan process for deleted instance %s", instance_id)
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()
                process.wait()

        config_manager.remove_instance(instance_id)

        # Wait for log thread to finish before removing buffers
        log_thread = self.log_threads.pop(instance_id, None)
        if log_thread and log_thread.is_alive():
            log_thread.join(timeout=3)

        self._purge_instance_resources(instance_id, call_runner_on_stop=False)

        self._push_instances_update()

        return True

    async def auto_restart_running_instances(self):
        """Auto-restart instances that were running before shutdown.
        Also resolves intermediate states (starting/stopping) left over
        from an interrupted shutdown.
        """
        for instance in config_manager.get_all_instances():
            if instance.status in (InstanceStatus.RUNNING, InstanceStatus.STARTING):
                logger.info(
                    "Auto-restarting instance: %s (%s)", instance.id, instance.config.alias
                )
                self._kill_stale_pid(instance.pid)
                instance.status = InstanceStatus.STOPPED
                instance.pid = None
                config_manager.update_instance(instance.id, instance)
                await self.start_instance(instance.id)
            elif instance.status == InstanceStatus.STOPPING:
                logger.info(
                    "Resolving interrupted stop for instance: %s (%s)",
                    instance.id,
                    instance.config.alias,
                )
                self._kill_stale_pid(instance.pid)
                instance.status = InstanceStatus.STOPPED
                instance.pid = None
                config_manager.update_instance(instance.id, instance)

    @staticmethod
    def _kill_stale_pid(pid: Optional[int]) -> None:
        """Best-effort kill of a stale PID left from a previous run."""
        if pid is None:
            return
        try:
            import signal
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to stale PID %d", pid)
        except ProcessLookupError:
            pass
        except OSError as e:
            logger.debug("Could not kill stale PID %d: %s", pid, e)


# Global process manager instance
process_manager = ProcessManager()
