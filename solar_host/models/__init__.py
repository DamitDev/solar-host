"""Models package for solar-host with multi-backend support."""

from typing import Union, Annotated
from pydantic import Field

# Import base models first (no dependencies on config types)
from solar_host.models.base import (
    BackendType,
    InstanceStatus,
    InstancePhase,
    Instance,
    InstanceCreate,
    InstanceUpdate,
    LogMessage,
    InstanceRuntimeState,
    InstanceStateEvent,
    InstanceResponse,
    MemoryInfo,
    GenerationMetrics,
)

# Import config models
from solar_host.models.llamacpp import LlamaCppConfig
from solar_host.models.huggingface import (
    HuggingFaceCausalConfig,
    HuggingFaceClassificationConfig,
    HuggingFaceEmbeddingConfig,
    HuggingFaceVisionConfig,
)

# Create the discriminated union type for InstanceConfig
InstanceConfig = Annotated[
    Union[
        LlamaCppConfig,
        HuggingFaceCausalConfig,
        HuggingFaceClassificationConfig,
        HuggingFaceEmbeddingConfig,
        HuggingFaceVisionConfig,
    ],
    Field(discriminator="backend_type"),
]

__all__ = [
    # Enums
    "BackendType",
    "InstanceStatus",
    "InstancePhase",
    # Config types
    "InstanceConfig",
    "LlamaCppConfig",
    "HuggingFaceCausalConfig",
    "HuggingFaceClassificationConfig",
    "HuggingFaceEmbeddingConfig",
    "HuggingFaceVisionConfig",
    # Instance models
    "Instance",
    "InstanceCreate",
    "InstanceUpdate",
    "InstanceResponse",
    # Runtime models
    "LogMessage",
    "InstanceRuntimeState",
    "InstanceStateEvent",
    "GenerationMetrics",
    # Other
    "MemoryInfo",
]
