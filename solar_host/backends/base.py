"""Abstract base class for backend runners."""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict
from dataclasses import dataclass

from solar_host.models.base import InstancePhase


@dataclass
class RuntimeStateUpdate:
    """Represents a runtime state update parsed from log output."""

    busy: bool
    phase: InstancePhase
    prefill_progress: Optional[float] = None
    active_slots: int = 0
    slot_id: Optional[int] = None
    task_id: Optional[int] = None
    prefill_prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None
    checkpoint_index: Optional[int] = None
    checkpoint_total: Optional[int] = None


class BackendRunner(ABC):
    """Abstract base class for backend-specific runners.

    Each backend type (llama.cpp, HuggingFace, etc.) implements this interface
    to handle process spawning, log parsing, and health checking.
    """

    @abstractmethod
    def build_command(self, instance: Any) -> List[str]:
        """Build the command to start the backend process.

        Args:
            instance: The Instance object containing config and runtime info.

        Returns:
            List of command arguments to spawn the process.
        """
        pass

    @abstractmethod
    def parse_log_line(
        self, instance_id: str, line: str, context: Dict[str, Any]
    ) -> Optional[RuntimeStateUpdate]:
        """Parse a log line and optionally return a runtime state update.

        Args:
            instance_id: The instance ID this log belongs to.
            line: The log line to parse.
            context: Mutable context dict for tracking state across log lines
                     (e.g., active slots, pending generations).

        Returns:
            RuntimeStateUpdate if the log line indicates a state change, None otherwise.
        """
        pass

    @abstractmethod
    def get_health_endpoint(self) -> str:
        """Get the health check endpoint path for this backend.

        Returns:
            The health endpoint path (e.g., "/health").
        """
        pass

    @abstractmethod
    def get_supported_endpoints(self) -> List[str]:
        """Get the list of API endpoints this backend supports.

        Returns:
            List of endpoint paths (e.g., ["/v1/chat/completions", "/v1/completions"]).
        """
        pass

    @abstractmethod
    def get_backend_type(self) -> str:
        """Get the backend type identifier.

        Returns:
            The backend type string (e.g., "llamacpp", "huggingface_causal").
        """
        pass

    def initialize_context(self) -> Dict[str, Any]:
        """Initialize the parsing context for a new instance.

        Override this method to provide backend-specific context initialization.

        Returns:
            Initial context dictionary.
        """
        return {}

    def on_process_started(self, instance_id: str, context: Dict[str, Any]) -> None:
        """Called when the backend process has started.

        Override this method to perform post-start initialization.

        Args:
            instance_id: The instance ID.
            context: The instance's parsing context.
        """
        pass

    def on_process_stopped(self, instance_id: str, context: Dict[str, Any]) -> None:
        """Called when the backend process has stopped.

        Override this method to perform cleanup.

        Args:
            instance_id: The instance ID.
            context: The instance's parsing context.
        """
        pass

    def get_supported_endpoints_for_type(self, backend_type: str) -> List[str]:
        """Get supported endpoints based on specific backend type.

        Override this method if the backend supports multiple model types
        with different endpoints (e.g., HuggingFace causal vs classification).

        Args:
            backend_type: The specific backend type string.

        Returns:
            List of endpoint paths for that backend type.
        """
        return self.get_supported_endpoints()

    def get_last_generation(self, context: Dict[str, Any]) -> Optional[Any]:
        """Get the last generation metrics from context.

        Override this method to provide generation metrics tracking.

        Args:
            context: The instance's parsing context.

        Returns:
            GenerationMetrics if available, None otherwise.
        """
        return None
