"""HuggingFace backend configuration models."""

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, List, Literal, Any


def _strip_api_key(data: Any) -> Any:
    """Remove api_key from old configs - instances use host API key."""
    if isinstance(data, dict):
        data.pop("api_key", None)
    return data


class HuggingFaceCausalConfig(BaseModel):
    """Configuration for a HuggingFace AutoModelForCausalLM instance.

    Note: api_key is NOT a config parameter - instances always use the host's API key.
    """

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def strip_api_key(cls, data: Any) -> Any:
        return _strip_api_key(data)

    backend_type: Literal["huggingface_causal"] = Field(
        default="huggingface_causal", description="Backend type identifier"
    )
    model_source: Optional[str] = Field(
        default=None, description="Model source URI (e.g. local://model_dir)"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path (e.g., 'meta-llama/Llama-2-7b-hf')",
    )

    @model_validator(mode="after")
    def check_model_id_or_source(self) -> "HuggingFaceCausalConfig":
        if not self.model_id and not self.model_source:
            raise ValueError("Either 'model_id' or 'model_source' must be provided")
        return self

    alias: str = Field(..., description="Model alias (e.g., llama2:7b)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(default=4096, description="Maximum sequence length")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )


class HuggingFaceClassificationConfig(BaseModel):
    """Configuration for a HuggingFace AutoModelForSequenceClassification instance.

    Note: api_key is NOT a config parameter - instances always use the host's API key.
    """

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def strip_api_key(cls, data: Any) -> Any:
        return _strip_api_key(data)

    backend_type: Literal["huggingface_classification"] = Field(
        default="huggingface_classification", description="Backend type identifier"
    )
    model_source: Optional[str] = Field(
        default=None, description="Model source URI (e.g. local://model_dir)"
    )
    model_id: Optional[str] = Field(
        default=None, description="HuggingFace model ID or local path"
    )

    @model_validator(mode="after")
    def check_model_id_or_source(self) -> "HuggingFaceClassificationConfig":
        if not self.model_id and not self.model_source:
            raise ValueError("Either 'model_id' or 'model_source' must be provided")
        return self

    alias: str = Field(..., description="Model alias (e.g., classifier:deberta)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(
        default=512, description="Maximum sequence length for classification"
    )
    labels: Optional[List[str]] = Field(
        default=None, description="Optional label names mapping (index -> label name)"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )


class HuggingFaceEmbeddingConfig(BaseModel):
    """Configuration for a HuggingFace embedding model instance using AutoModel.

    Note: api_key is NOT a config parameter - instances always use the host's API key.
    """

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def strip_api_key(cls, data: Any) -> Any:
        return _strip_api_key(data)

    backend_type: Literal["huggingface_embedding"] = Field(
        default="huggingface_embedding", description="Backend type identifier"
    )
    model_source: Optional[str] = Field(
        default=None, description="Model source URI (e.g. local://model_dir)"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path (e.g., 'sentence-transformers/all-MiniLM-L6-v2')",
    )

    @model_validator(mode="after")
    def check_model_id_or_source(self) -> "HuggingFaceEmbeddingConfig":
        if not self.model_id and not self.model_source:
            raise ValueError("Either 'model_id' or 'model_source' must be provided")
        return self

    alias: str = Field(..., description="Model alias (e.g., embed:minilm)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(
        default=512, description="Maximum sequence length for embeddings"
    )
    normalize_embeddings: bool = Field(
        default=True, description="L2 normalize output embedding vectors"
    )
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )


class HuggingFaceVisionConfig(BaseModel):
    """Configuration for a HuggingFace vision/multimodal (image-text-to-text) instance.

    Loaded via AutoModelForImageTextToText (or AutoModelForVision2Seq fallback)
    plus AutoProcessor for image preprocessing.

    Note: api_key is NOT a config parameter - instances always use the host's API key.
    """

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def strip_api_key(cls, data: Any) -> Any:
        return _strip_api_key(data)

    backend_type: Literal["huggingface_vision"] = Field(
        default="huggingface_vision", description="Backend type identifier"
    )
    model_source: Optional[str] = Field(
        default=None, description="Model source URI (e.g. local://model_dir)"
    )
    model_id: Optional[str] = Field(
        default=None,
        description="HuggingFace model ID or local path (e.g., 'Qwen/Qwen2.5-VL-7B-Instruct')",
    )

    @model_validator(mode="after")
    def check_model_id_or_source(self) -> "HuggingFaceVisionConfig":
        if not self.model_id and not self.model_source:
            raise ValueError("Either 'model_id' or 'model_source' must be provided")
        return self

    alias: str = Field(..., description="Model alias (e.g., qwen-vl:7b)")
    device: str = Field(
        default="auto", description="Device to run on: auto, cuda, mps (Mac), cpu"
    )
    dtype: str = Field(
        default="auto", description="Data type: auto, float16, bfloat16, float32"
    )
    max_length: int = Field(default=4096, description="Maximum sequence length")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust remote code from HuggingFace"
    )
    use_flash_attention: bool = Field(
        default=True, description="Use Flash Attention 2 if available"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
