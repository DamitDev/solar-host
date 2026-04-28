"""Base models shared across all backend types."""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, timezone
from enum import Enum


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
    prefill_progress: Optional[float] = None
    active_slots: int = 0
    # Optional contextual metrics
    slot_id: Optional[int] = None
    task_id: Optional[int] = None
    prefill_prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None
    checkpoint_index: Optional[int] = None
    checkpoint_total: Optional[int] = None
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
    slot_id: Optional[int] = None
    task_id: Optional[int] = None

    # Token usage
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None

    # Decode performance
    decode_tps: Optional[float] = None
    decode_ms_per_token: Optional[float] = None

    # Timestamps
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class Instance(BaseModel):
    """Runtime instance information.

    Note: config field uses Any type here to avoid circular imports.
    The actual type is InstanceConfig (discriminated union) defined in __init__.py.
    """

    id: str
    config: Any  # InstanceConfig - discriminated union
    status: InstanceStatus = InstanceStatus.STOPPED
    port: Optional[int] = None
    pid: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # Supported API endpoints for this instance (populated by backend runner)
    supported_endpoints: List[str] = Field(default_factory=list)

    # Ephemeral runtime fields (not persisted to disk)
    busy: bool = Field(default=False, exclude=True)
    prefill_progress: Optional[float] = Field(default=None, exclude=True)
    active_slots: int = Field(default=0, exclude=True)


class InstanceCreate(BaseModel):
    """Request to create a new instance.

    Note: config field uses Any type here to avoid circular imports.
    """

    config: Any  # InstanceConfig


class InstanceUpdate(BaseModel):
    """Request to update an instance config.

    Note: config field uses Any type here to avoid circular imports.
    """

    config: Any  # InstanceConfig


class InstanceResponse(BaseModel):
    """Response model for instance operations."""

    instance: Instance
    message: str
