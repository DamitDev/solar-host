"""Base models shared across all backend types."""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BackendType(str, Enum):
    """Supported backend types for model inference."""

    LLAMACPP = "llamacpp"
    HUGGINGFACE_CAUSAL = "huggingface_causal"
    HUGGINGFACE_CLASSIFICATION = "huggingface_classification"
    HUGGINGFACE_EMBEDDING = "huggingface_embedding"
    HUGGINGFACE_VISION = "huggingface_vision"


class InstanceStatus(str, Enum):
    """Status of a model instance."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"


class InstancePhase(str, Enum):
    """Fine-grained runtime phase of an active request."""

    IDLE = "idle"
    PREFILL = "prefill"
    GENERATING = "generating"


class InstancePriority(str, Enum):
    """Priority level for inference instances (S-036)."""

    PRODUCTION = "production"
    STAGING = "staging"
    EPHEMERAL = "ephemeral"


class LogMessage(BaseModel):
    """Log message with sequence number."""

    seq: int
    timestamp: str
    line: str


class InstanceRuntimeState(BaseModel):
    """Ephemeral runtime state for an instance."""

    instance_id: str
    busy: bool
    phase: InstancePhase = InstancePhase.IDLE
    prefill_progress: float | None = None
    active_slots: int = 0
    # Optional contextual metrics
    slot_id: int | None = None
    task_id: int | None = None
    prefill_prompt_tokens: int | None = None
    generated_tokens: int | None = None
    decode_tps: float | None = None
    decode_ms_per_token: float | None = None
    checkpoint_index: int | None = None
    checkpoint_total: int | None = None
    timestamp: str


class InstanceStateEvent(BaseModel):
    """State change event used for WebSocket streaming of runtime state."""

    seq: int
    timestamp: str
    type: str = "instance_state"
    data: InstanceRuntimeState


class MemoryInfo(BaseModel):
    """Memory usage information."""

    used_gb: float = Field(..., description="Used memory in GB")
    total_gb: float = Field(..., description="Total memory in GB")
    available_gb: float = Field(
        ..., description="Memory available for new workloads (total - used)"
    )
    percent: float = Field(..., description="Usage percentage")
    memory_type: str = Field(..., description="Type of memory (VRAM or RAM)")


class GenerationMetrics(BaseModel):
    """Per-generation token usage and timing metrics."""

    instance_id: str
    slot_id: int | None = None
    task_id: int | None = None

    # Token usage
    prompt_tokens: int | None = None
    generated_tokens: int | None = None

    # Decode performance
    decode_tps: float | None = None
    decode_ms_per_token: float | None = None

    # Timestamps
    started_at: str | None = None
    finished_at: str | None = None


class Instance(BaseModel):
    """Runtime instance information.

    Note: config field uses Any type here to avoid circular imports.
    The actual type is InstanceConfig (discriminated union) defined in __init__.py.
    """

    id: str
    config: Any  # InstanceConfig - discriminated union
    status: InstanceStatus = InstanceStatus.STOPPED
    port: int | None = None
    pid: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0

    # Priority and ownership (S-036)
    priority: InstancePriority = Field(
        default=InstancePriority.PRODUCTION,
        description="Instance priority for placement and eviction decisions",
    )
    managed_by: str | None = Field(
        default=None,
        description="Owner subsystem (e.g. 'intent' for reconciler-managed instances)",
    )
    intent_id: str | None = Field(
        default=None,
        description="Owning intent ID (set iff managed_by == 'intent')",
    )

    # Supported API endpoints for this instance (populated by backend runner)
    supported_endpoints: list[str] = Field(default_factory=list)

    # Ephemeral runtime fields (not persisted to disk)
    busy: bool = Field(default=False, exclude=True)
    prefill_progress: float | None = Field(default=None, exclude=True)
    active_slots: int = Field(default=0, exclude=True)


class InstanceCreate(BaseModel):
    """Request to create a new instance.

    Note: config field uses Any type here to avoid circular imports.
    """

    config: Any  # InstanceConfig
    priority: str | None = None
    managed_by: str | None = None
    intent_id: str | None = None


class InstanceUpdate(BaseModel):
    """Request to update an instance config or ownership markers.

    Note: config field uses Any type here to avoid circular imports.
    All fields are optional; only explicitly-provided fields are applied
    (``managed_by``/``intent_id`` may be set to null to clear ownership).
    """

    config: Any | None = None  # InstanceConfig
    managed_by: str | None = None
    intent_id: str | None = None


class InstanceResponse(BaseModel):
    """Response model for instance operations."""

    instance: Instance
    message: str
