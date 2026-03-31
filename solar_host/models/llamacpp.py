"""LlamaCpp backend configuration models."""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Any


class LlamaCppConfig(BaseModel):
    """Configuration for a llama.cpp server instance.

    Note: api_key is NOT a config parameter - instances always use the host's API key.
    """

    backend_type: Literal["llamacpp"] = Field(
        default="llamacpp", description="Backend type identifier"
    )

    @model_validator(mode="before")
    @classmethod
    def strip_api_key(cls, data: Any) -> Any:
        """Remove api_key from old configs - instances use host API key."""
        if isinstance(data, dict):
            data.pop("api_key", None)
        return data

    model: str = Field(..., description="Path to the GGUF model file")
    alias: str = Field(..., description="Model alias (e.g., gpt-oss:120b)")
    threads: int = Field(default=1, description="Number of threads")
    n_gpu_layers: int = Field(default=999, description="Number of GPU layers")
    temp: float = Field(default=1.0, description="Temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")
    top_k: int = Field(default=0, description="Top-k sampling")
    min_p: float = Field(default=0.0, description="Min-p sampling")
    ctx_size: int = Field(default=131072, description="Context size")
    chat_template_file: Optional[str] = Field(
        default=None, description="Path to Jinja chat template"
    )
    chat_template_kwargs: Optional[str] = Field(
        default=None,
        description="JSON string of chat template kwargs (e.g. '{\"enable_thinking\":true}')",
    )
    reasoning: Optional[Literal["on", "off", "auto"]] = Field(
        default=None,
        description="Reasoning/thinking mode: 'on', 'off', or 'auto' (passed as --reasoning to llama-server)",
    )
    reasoning_budget: Optional[int] = Field(
        default=None,
        description="Reasoning budget token limit (passed as --reasoning-budget to llama-server)",
    )
    cache_type_k: Optional[
        Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
    ] = Field(default=None, description="KV cache quantization type for keys (-ctk)")
    cache_type_v: Optional[
        Literal["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
    ] = Field(default=None, description="KV cache quantization type for values (-ctv)")
    rope_scaling: Optional[Literal["none", "linear", "yarn"]] = Field(
        default=None, description="RoPE scaling method (--rope-scaling)"
    )
    rope_scale: Optional[float] = Field(
        default=None, description="RoPE context scaling factor (--rope-scale)"
    )
    yarn_orig_ctx: Optional[int] = Field(
        default=None, description="YaRN original context size (--yarn-orig-ctx)"
    )
    host: str = Field(default="0.0.0.0", description="Host to bind to")
    port: Optional[int] = Field(
        default=None, description="Port (auto-assigned if not specified)"
    )
    special: bool = Field(
        default=False, description="Enable llama-server --special flag"
    )
    ot: Optional[str] = Field(
        default=None,
        description="Override tensor string (passed as -ot flag to llama-server)",
    )
    model_type: Optional[Literal["llm", "embedding", "reranker"]] = Field(
        default="llm", description="Model type: llm (default), embedding, or reranker"
    )
    pooling: Optional[Literal["none", "mean", "cls", "last", "rank"]] = Field(
        default=None,
        description="Pooling strategy for embedding models (only valid when model_type is embedding)",
    )
