"""HuggingFace backend runner implementation."""

import re
import sys
from typing import List, Optional, Any, Dict

from solar_host.backends.base import BackendRunner, RuntimeStateUpdate
from solar_host.models.base import InstancePhase, GenerationMetrics, BackendType
from solar_host.config import settings


class HuggingFaceRunner(BackendRunner):
    """Backend runner for HuggingFace model instances.

    Supports AutoModelForCausalLM, AutoModelForSequenceClassification, and AutoModel (embeddings).
    Spawns the hf_server.py process which provides OpenAI-compatible endpoints.
    """

    def __init__(self):
        # Regex patterns for HF server log parsing
        self._re_ready = re.compile(r"Uvicorn running on|Application startup complete")
        self._re_request_start = re.compile(r"\[REQUEST\] model=(\S+) endpoint=(\S+)")
        self._re_request_complete = re.compile(
            r"\[COMPLETE\] model=(\S+) tokens=(\d+) time_ms=([0-9.]+)"
        )
        self._re_error = re.compile(r"\[ERROR\] (.+)")

    def get_backend_type(self) -> str:
        # This will be overridden based on the actual config type
        return "huggingface"

    @staticmethod
    def check_dependencies() -> None:
        """Verify that HuggingFace backend dependencies are installed."""
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "HuggingFace backend requires the 'huggingface' extra. "
                "Install with: pip install solar-host[huggingface]"
            )

    def build_command(self, instance: Any) -> List[str]:
        """Build command to start HuggingFace server process."""
        self.check_dependencies()
        config = instance.config
        backend_type = config.backend_type

        # Determine model type for the server
        if (
            backend_type == BackendType.HUGGINGFACE_CAUSAL
            or backend_type == "huggingface_causal"
        ):
            model_type = "causal"
        elif (
            backend_type == BackendType.HUGGINGFACE_CLASSIFICATION
            or backend_type == "huggingface_classification"
        ):
            model_type = "classification"
        elif (
            backend_type == BackendType.HUGGINGFACE_EMBEDDING
            or backend_type == "huggingface_embedding"
        ):
            model_type = "embedding"
        else:
            raise RuntimeError(f"Unknown HuggingFace backend_type: {backend_type!r}")

        cmd = [
            sys.executable,
            "-m",
            "solar_host.servers.hf_server",
            "--model-id",
            config.model_id,
            "--model-type",
            model_type,
            "--alias",
            config.alias,
            "--host",
            config.host,
            "--port",
            str(instance.port),
            "--api-key",
            settings.api_key,
            "--device",
            config.device,
            "--dtype",
            config.dtype,
            "--max-length",
            str(config.max_length),
        ]

        if getattr(config, "trust_remote_code", False):
            cmd.append("--trust-remote-code")

        if getattr(config, "use_flash_attention", True):
            cmd.append("--use-flash-attention")

        # For classification models, pass labels if specified
        if model_type == "classification" and getattr(config, "labels", None):
            cmd.extend(["--labels", ",".join(config.labels)])

        # For embedding models, pass normalize flag if specified
        if model_type == "embedding" and getattr(config, "normalize_embeddings", True):
            cmd.append("--normalize-embeddings")

        return cmd

    def get_health_endpoint(self) -> str:
        return "/health"

    def get_supported_endpoints(self) -> List[str]:
        """Get supported endpoints.

        Note: This returns all possible endpoints. The actual endpoints
        depend on the model type (causal vs classification).
        The server itself will report which endpoints it supports.
        """
        # Will be refined based on actual model type
        return [
            "/v1/models",
            "/health",
        ]

    def get_supported_endpoints_for_type(self, backend_type: str) -> List[str]:
        """Get supported endpoints based on backend type."""
        if (
            backend_type == BackendType.HUGGINGFACE_CAUSAL
            or backend_type == "huggingface_causal"
        ):
            return [
                "/v1/chat/completions",
                "/v1/completions",
                "/v1/models",
                "/health",
            ]
        elif (
            backend_type == BackendType.HUGGINGFACE_CLASSIFICATION
            or backend_type == "huggingface_classification"
        ):
            return [
                "/v1/classify",
                "/v1/models",
                "/health",
            ]
        elif (
            backend_type == BackendType.HUGGINGFACE_EMBEDDING
            or backend_type == "huggingface_embedding"
        ):
            return [
                "/v1/embeddings",
                "/v1/models",
                "/health",
            ]
        return ["/v1/models", "/health"]

    def initialize_context(self) -> Dict[str, Any]:
        """Initialize parsing context for HuggingFace server log parsing."""
        return {
            "ready": False,
            "busy": False,
            "current_request": None,
            "recent_generations": [],
            "last_state": {
                "busy": False,
                "phase": InstancePhase.IDLE.value,
            },
        }

    def parse_log_line(
        self, instance_id: str, line: str, context: Dict[str, Any]
    ) -> Optional[RuntimeStateUpdate]:
        """Parse HuggingFace server log line and return state update if changed."""
        last_state = context.get("last_state", {})

        # Check for server ready
        if self._re_ready.search(line):
            context["ready"] = True
            return self._create_update(
                busy=False,
                phase=InstancePhase.IDLE,
                last_state=last_state,
                context=context,
            )

        # Check for request start
        m = self._re_request_start.search(line)
        if m:
            context["busy"] = True
            context["current_request"] = {
                "model": m.group(1),
                "endpoint": m.group(2),
            }
            return self._create_update(
                busy=True,
                phase=InstancePhase.GENERATING,
                last_state=last_state,
                context=context,
            )

        # Check for request complete
        m = self._re_request_complete.search(line)
        if m:
            context["busy"] = False
            try:
                tokens = int(m.group(2))
                time_ms = float(m.group(3))
            except (ValueError, TypeError):
                tokens = 0
                time_ms = 0.0

            tps = (tokens / time_ms * 1000) if time_ms > 0 else None

            if context.get("current_request"):
                metrics = GenerationMetrics(
                    instance_id=instance_id,
                    generated_tokens=tokens,
                    decode_tps=tps,
                    decode_ms_per_token=(time_ms / tokens) if tokens > 0 else None,
                )
                recent = context.get("recent_generations", [])
                recent.append(metrics)
                if len(recent) > 100:
                    recent = recent[-100:]
                context["recent_generations"] = recent

            context["current_request"] = None

            return self._create_update(
                busy=False,
                phase=InstancePhase.IDLE,
                generated_tokens=tokens,
                decode_tps=tps,
                last_state=last_state,
                context=context,
            )

        # Check for errors
        m = self._re_error.search(line)
        if m:
            context["busy"] = False
            context["current_request"] = None
            return self._create_update(
                busy=False,
                phase=InstancePhase.IDLE,
                last_state=last_state,
                context=context,
            )

        return None

    def _create_update(
        self,
        busy: bool,
        phase: InstancePhase,
        last_state: Dict[str, Any],
        context: Dict[str, Any],
        generated_tokens: Optional[int] = None,
        decode_tps: Optional[float] = None,
    ) -> Optional[RuntimeStateUpdate]:
        """Create a RuntimeStateUpdate if state has changed."""
        changed = (
            last_state.get("busy") != busy
            or last_state.get("phase") != phase.value
            or last_state.get("generated_tokens") != generated_tokens
            or last_state.get("decode_tps") != decode_tps
        )

        if not changed:
            return None

        context["last_state"] = {
            "busy": busy,
            "phase": phase.value,
            "generated_tokens": generated_tokens,
            "decode_tps": decode_tps,
        }

        return RuntimeStateUpdate(
            busy=busy,
            phase=phase,
            active_slots=1 if busy else 0,
            generated_tokens=generated_tokens,
            decode_tps=decode_tps,
        )

    def get_last_generation(
        self, context: Dict[str, Any]
    ) -> Optional[GenerationMetrics]:
        """Get the last generation metrics from context."""
        recent = context.get("recent_generations", [])
        if not recent:
            return None
        return recent[-1]
