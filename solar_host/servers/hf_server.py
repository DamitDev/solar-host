#!/usr/bin/env python3
"""
Standalone HuggingFace Server

A FastAPI application that loads HuggingFace models and provides
OpenAI-compatible API endpoints.

Supports:
- AutoModelForCausalLM: text generation (/v1/chat/completions, /v1/completions)
- AutoModelForSequenceClassification: classification (/v1/classify)
- AutoModel: embeddings (/v1/embeddings)

Usage:
    python -m solar_host.servers.hf_server \
        --model-id "meta-llama/Llama-2-7b-hf" \
        --model-type causal \
        --alias "llama2:7b" \
        --port 3501 \
        --api-key "secret"
"""

import argparse
import base64
import io
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Any, Optional, List, Dict, Literal, Union, TYPE_CHECKING
from contextlib import asynccontextmanager
from urllib.parse import urlparse

import torch
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import json

if TYPE_CHECKING:
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

# Capabilities exposed via /v1/models, used by downstream apps
# (e.g. Orchestrator) to differentiate text-only vs multimodal models.
MODEL_TYPE_CAPABILITIES: Dict[str, List[str]] = {
    "causal": ["completion"],
    "classification": ["classification"],
    "embedding": ["embedding"],
    "vision": ["completion", "multimodal"],
}

# Configure logging to output structured messages for the runner to parse
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class ToolFunctionCall(BaseModel):
    """OpenAI-compatible function call payload inside a tool_call."""

    name: str
    # ``arguments`` is a JSON-encoded string per OpenAI's spec.
    arguments: str = ""


class ToolCall(BaseModel):
    """OpenAI-compatible tool call (function-only flavor)."""

    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class ChatMessage(BaseModel):
    role: str
    # Either a plain string (text-only) or an OpenAI-style list of content parts
    # like [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}}].
    # ``None`` is permitted for assistant messages that contain only tool_calls.
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    # OpenAI standard fields. ``reasoning_content`` is what downstream apps expect
    # for thinking-mode models (DeepSeek-V4, etc.); ``tool_calls`` and
    # ``tool_call_id`` carry tool-use state across turns.
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None


class FunctionDefinition(BaseModel):
    """OpenAI-style function tool definition (the ``function`` field of a tool)."""

    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    """OpenAI-style tool entry submitted on the request."""

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    # OpenAI tool-calling.
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    # DeepSeek-V4 specific knobs (ignored by other models). Defaults are filled
    # from the server's CLI flags when the request omits them.
    thinking_mode: Optional[Literal["thinking", "chat"]] = None
    reasoning_effort: Optional[Literal["low", "medium", "high", "max"]] = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ClassifyRequest(BaseModel):
    model: str
    input: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to classify"
    )
    return_all_scores: bool = Field(
        default=False,
        description="Return scores for all classes, not just top prediction",
    )


class ClassifyScoreItem(BaseModel):
    """Individual class score."""

    label: str
    score: float


class ClassifyChoice(BaseModel):
    index: int
    label: str
    score: float
    all_scores: Optional[List[ClassifyScoreItem]] = Field(
        default=None, description="Scores for all classes (when return_all_scores=True)"
    )


class ClassifyResponse(BaseModel):
    id: str
    object: str = "classification"
    model: str
    choices: List[ClassifyChoice]
    usage: Dict[str, int]


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "huggingface"


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]] = Field(
        ..., description="Text or list of texts to embed"
    )
    encoding_format: Optional[str] = Field(
        default="float", description="Encoding format: 'float' or 'base64'"
    )
    dimensions: Optional[int] = Field(
        default=None, description="Optional dimension truncation"
    )


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


# ============================================================================
# transformers patches
# ============================================================================


_caching_allocator_warmup_patched: bool = False
_fp8_quantizer_patched: bool = False
_mps_fp8_dtypes_loaded: Optional[bool] = None


def _ensure_mps_fp8_dtypes() -> bool:
    """Import ``fp4_fp8_for_torch_mps`` on macOS so torch's MPS backend gains
    ``float8_e4m3fn`` / ``float8_e5m2`` / ``float4_e2m1fn_x2`` support plus a
    Metal-shader ``torch._scaled_mm``.

    The package self-registers on import, so we only need to import it (it's
    optional and only listed as a dep on ``sys_platform == 'darwin'``).
    Returns True on success, False if the package isn't installed.
    """
    global _mps_fp8_dtypes_loaded
    if _mps_fp8_dtypes_loaded is not None:
        return _mps_fp8_dtypes_loaded

    if not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        _mps_fp8_dtypes_loaded = False
        return False

    try:
        import fp4_fp8_for_torch_mps  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dep
        logger.warning(
            "fp4-fp8-for-torch-mps is not installed; transformers' FP8 "
            "quantizer will dequantize FP8 weights to bf16 on MPS, which "
            f"may exceed unified memory. ({exc})"
        )
        _mps_fp8_dtypes_loaded = False
        return False

    logger.info(
        "Loaded fp4-fp8-for-torch-mps: float8_e4m3fn / float8_e5m2 / "
        "float4_e2m1fn_x2 are now available on MPS via Metal shaders."
    )
    _mps_fp8_dtypes_loaded = True
    return True


def _patch_fp8_quantizer_for_mps() -> None:
    """Stop transformers' FP8 quantizer from forcing bf16 dequant on MPS.

    Upstream's ``FineGrainedFP8HfQuantizer.validate_environment`` flips
    ``quantization_config.dequantize = True`` whenever neither CUDA nor XPU is
    available, which is the wrong call when ``fp4-fp8-for-torch-mps`` is
    installed (it provides FP8 dtypes + ``torch._scaled_mm`` on MPS via
    Metal). We replace ``validate_environment`` with a wrapper that returns
    early on MPS so dequant stays disabled.

    Idempotent. No-op when MPS isn't available or transformers' module path
    has shifted (we log and skip rather than break the load).
    """
    global _fp8_quantizer_patched
    if _fp8_quantizer_patched:
        return
    if not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return
    if not _ensure_mps_fp8_dtypes():
        return

    quantizer_cls = None
    try:
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer as _Cls,
        )

        quantizer_cls = _Cls
    except Exception:
        try:
            from transformers.quantizers.quantizer_fp8 import (  # type: ignore[import-not-found]
                FP8HfQuantizer as _Cls,
            )

            quantizer_cls = _Cls
        except Exception as exc:
            logger.warning(
                "Could not locate transformers FP8 quantizer to patch for MPS "
                f"({exc}); FP8 weights will be dequantized to bf16."
            )
            return

    _orig = quantizer_cls.validate_environment

    def _patched(self, *args, **kwargs):
        # Run the upstream accelerate-availability check, then short-circuit
        # before the cuda/xpu gate flips ``dequantize=True``.
        try:
            from transformers.utils import is_accelerate_available
        except Exception:
            is_accelerate_available = lambda: True  # noqa: E731
        if not is_accelerate_available():
            raise ImportError(
                "Loading an FP8 quantized model requires accelerate "
                "(`pip install accelerate`)"
            )
        if getattr(self.quantization_config, "dequantize", False):
            return
        # Reject explicit cpu/disk shards in the device_map for non-pre-quantized
        # loads, mirroring upstream's invariant.
        device_map = kwargs.get("device_map")
        if (
            isinstance(device_map, dict)
            and not getattr(self, "pre_quantized", False)
            and len(device_map) > 1
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "FP8 on-the-fly quantization does not support cpu/disk shards "
                "in device_map; use a pre-quantized checkpoint or a single "
                "device target."
            )
        return  # MPS path: leave dequantize=False, weights stay FP8.

    quantizer_cls.validate_environment = _patched  # type: ignore[assignment]
    _fp8_quantizer_patched = True
    # Stash the original for diagnostics / future unpatch if ever needed.
    quantizer_cls._solar_host_original_validate_environment = _orig  # type: ignore[attr-defined]
    logger.info(
        f"Patched {quantizer_cls.__name__}.validate_environment to keep FP8 "
        "weights resident on MPS (no bf16 dequant)."
    )

    # ----- Phase 2: install a no-dequant scale-conversion path -----
    # Upstream's ``update_weight_conversions`` only injects the
    # ``.scale`` -> ``.weight_scale_inv`` rename and the parallel
    # scale-stacking converters when ``dequantize=True``. With
    # ``dequantize=False`` (our MPS path) the rename never runs, so
    # checkpoint scales fall on the floor and the model's
    # ``*.weight_scale_inv`` / ``*_scale_inv`` parameters get re-initialized
    # to random values -- producing garbage at inference time.
    #
    # Mirror the upstream logic minus the ``Fp8Dequantize`` op so scale
    # tensors land in the correct buckets (including per-expert stacking
    # via the existing ``MergeModulelist`` / ``Concatenate`` ops).
    _orig_update = getattr(quantizer_cls, "update_weight_conversions", None)
    if _orig_update is None:
        logger.warning(
            "FP8 quantizer has no update_weight_conversions; cannot install "
            "MPS scale-conversion patch."
        )
        return

    def _patched_update(self, weight_conversions):
        from transformers.core_model_loading import (
            WeightConverter,
            WeightRenaming,
        )

        # Defer to upstream when we don't own the situation: not pre-quantized
        # (nothing to convert), or dequantize is on (full upstream pipeline).
        if not getattr(self, "pre_quantized", False) or getattr(
            self.quantization_config, "dequantize", False
        ):
            return _orig_update(self, weight_conversions)

        # Generic ``foo.scale`` -> ``foo.weight_scale_inv`` rename.
        # The upstream ``conversion_mapping`` for deepseek_v4 leaves these
        # as-is and relies on this rename living in the FP8 quantizer.
        scale_rename = WeightRenaming(
            source_patterns=r"^(.+)\.scale$",
            target_patterns=r"\1.weight_scale_inv",
        )

        # For every WeightConverter that fuses ``.weight`` sources into a
        # stacked target (e.g. per-expert ``experts.*.w1.weight`` ->
        # ``experts.gate_up_proj``), build a parallel converter that does
        # the same merge/concat for the matching scale tensors and writes
        # them to ``<original_target>_scale_inv`` (matching the names
        # registered by FP8Experts in modeling_deepseek_v4.py).
        scale_converters: list = []
        for conv in weight_conversions:
            if not isinstance(conv, WeightConverter):
                continue
            sources = list(conv.source_patterns)
            weight_sources = [p for p in sources if p.endswith(".weight")]
            if not weight_sources:
                continue
            scale_sources = [
                p[: -len(".weight")] + ".weight_scale_inv"
                for p in weight_sources
            ]
            target = getattr(
                conv, "_original_target_patterns", conv.target_patterns
            )
            scale_target = (
                target + "_scale_inv"
                if not target.endswith(".weight")
                else target[: -len(".weight")] + ".weight_scale_inv"
            )
            scale_converters.append(
                WeightConverter(
                    source_patterns=scale_sources,
                    target_patterns=scale_target,
                    operations=list(conv.operations),
                )
            )

        # Order:
        #   1. ``.scale`` -> ``.weight_scale_inv`` rename runs first so the
        #      scale-stacking converters below see the canonical name.
        #   2. Original conversions (weight stacking + structural prefix
        #      renames). Prefix renames are suffix-agnostic so they also
        #      apply to the renamed ``.weight_scale_inv`` keys.
        #   3. Scale-stacking converters mirroring step 2.
        return [scale_rename] + list(weight_conversions) + scale_converters

    quantizer_cls.update_weight_conversions = _patched_update  # type: ignore[assignment]
    quantizer_cls._solar_host_original_update_weight_conversions = _orig_update  # type: ignore[attr-defined]
    logger.info(
        f"Patched {quantizer_cls.__name__}.update_weight_conversions to "
        "rename .scale -> .weight_scale_inv and stack per-expert scales on "
        "MPS (no dequant)."
    )


def _disable_caching_allocator_warmup_for_mps() -> None:
    """Replace ``transformers.modeling_utils.caching_allocator_warmup`` with a
    pass-through that skips MPS devices.

    The upstream warmup pre-allocates one giant tensor per device to prime
    CUDA's caching allocator. On MPS (Apple Silicon) that single allocation
    bumps against ``MTLDevice.maxBufferLength`` for any large model -- e.g.
    DeepSeek-V4-Flash dequantized to bf16 needs a single ~405 GiB buffer,
    which Metal rejects even on a 512 GB Mac Studio. The warmup provides no
    benefit on MPS, so neutralizing it for MPS-bound shards is safe.

    Idempotent: only patches once per process.
    """
    global _caching_allocator_warmup_patched
    if _caching_allocator_warmup_patched:
        return

    try:
        from transformers import modeling_utils as _hf_mu
    except Exception as exc:  # pragma: no cover - transformers is required
        logger.warning(f"Could not patch caching_allocator_warmup: {exc}")
        return

    _orig = getattr(_hf_mu, "caching_allocator_warmup", None)
    if _orig is None:
        return

    def _patched(model, expanded_device_map, hf_quantizer, *args, **kwargs):
        # Drop MPS entries before calling through; if every entry is MPS,
        # skip the warmup entirely.
        try:
            non_mps_map = {
                k: v
                for k, v in dict(expanded_device_map).items()
                if str(v) != "mps"
            }
        except Exception:
            non_mps_map = expanded_device_map

        if not non_mps_map:
            logger.info(
                "Skipping transformers caching_allocator_warmup: all shards "
                "are on MPS (Metal per-buffer limits make the warmup unsafe)."
            )
            return None
        return _orig(model, non_mps_map, hf_quantizer, *args, **kwargs)

    _hf_mu.caching_allocator_warmup = _patched
    _caching_allocator_warmup_patched = True
    logger.info(
        "Patched transformers.modeling_utils.caching_allocator_warmup to "
        "skip MPS-bound shards."
    )


# ============================================================================
# Global State
# ============================================================================


class ServerState:
    """Global server state holding the loaded model."""

    def __init__(self):
        self.model: Optional["PreTrainedModel"] = None
        self.tokenizer: Optional[
            Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"]
        ] = None
        # Multi-modal processor (used for vision/multimodal models). Wraps
        # the tokenizer plus image preprocessor.
        self.processor: Optional[Any] = None
        self.model_id: str = ""
        self.model_type: str = "causal"
        self.alias: str = ""
        self.device: str = "cpu"
        self.max_length: int = 4096
        self.labels: Optional[List[str]] = None
        self.normalize_embeddings: bool = True
        self.api_key: str = ""
        self.created_at: int = int(datetime.now(timezone.utc).timestamp())
        # Architecture flags + chat defaults. ``is_deepseek_v4`` switches the
        # causal chat path to the vendored DSV4 encoder/parser.
        self.is_deepseek_v4: bool = False
        self.default_thinking_mode: str = "thinking"
        self.default_reasoning_effort: Optional[str] = None

    def ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded, raise if not."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def get_device(self, device_str: str) -> str:
        """Resolve device string to actual device."""
        if device_str == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_str

    def get_dtype(self, dtype_str: str) -> torch.dtype:
        """Resolve dtype string to torch dtype."""
        if dtype_str == "auto":
            if self.device == "cuda":
                return torch.float16
            elif self.device == "mps":
                return torch.float16
            else:
                return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(dtype_str, torch.float32)

    def load_model(
        self,
        model_id: str,
        model_type: str,
        alias: str,
        device: str,
        dtype: str,
        max_length: int,
        labels: Optional[List[str]],
        trust_remote_code: bool,
        use_flash_attention: bool,
        normalize_embeddings: bool = True,
    ):
        """Load the model based on type."""
        from transformers import AutoConfig, AutoTokenizer

        self.model_id = model_id
        self.model_type = model_type
        self.alias = alias
        self.device = self.get_device(device)
        self.max_length = max_length
        self.labels = labels
        self.normalize_embeddings = normalize_embeddings

        # Inspect the model config up-front so we can pick a sensible dispatch
        # strategy (e.g. DeepSeek-V4 ships FP4+FP8 mixed weights and must keep
        # ``dtype="auto"`` so transformers honors the ``quantization_config``
        # block).
        try:
            hf_config = AutoConfig.from_pretrained(
                model_id, trust_remote_code=trust_remote_code
            )
            arch_model_type = getattr(hf_config, "model_type", "")
        except Exception as exc:  # pragma: no cover - logged for diagnostics
            logger.warning(
                f"Could not pre-load AutoConfig for {model_id}: {exc}. "
                "Falling back to defaults."
            )
            arch_model_type = ""

        self.is_deepseek_v4 = arch_model_type == "deepseek_v4"
        if self.is_deepseek_v4 and dtype != "auto":
            logger.warning(
                f"DeepSeek-V4 weights are FP4+FP8 mixed; ignoring requested dtype "
                f"'{dtype}' and using 'auto' so transformers can dequantize "
                "according to the model's quantization_config."
            )
            dtype = "auto"

        model_dtype = self.get_dtype(dtype)

        logger.info(f"Loading model: {model_id}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Architecture: {arch_model_type or '<unknown>'}")
        logger.info(f"Device: {self.device}")
        if self.is_deepseek_v4:
            # The actual ``dtype`` arg passed to ``from_pretrained`` is the literal
            # string ``"auto"`` (not the resolved torch dtype) so transformers can
            # honor the model's quantization_config block.
            logger.info("Dtype: auto (forced by deepseek_v4 quantization_config)")
        else:
            logger.info(f"Dtype: {dtype} -> {model_dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer = tokenizer

        # Load model based on type
        if model_type == "causal":
            from transformers import AutoModelForCausalLM

            # DeepSeek-V4 (and other large MoE models) need accelerate-driven
            # placement so FP4 expert shards land on the unified-memory device
            # alongside the FP8 backbone. For everything else we keep the
            # historic behavior (single-device on CUDA, manual ``.to()`` on MPS).
            if self.is_deepseek_v4:
                device_map: Optional[Union[str, Dict[str, Any]]] = "auto"
                pass_dtype: Any = "auto"
            else:
                device_map = self.device if self.device != "mps" else None
                pass_dtype = model_dtype

            model_kwargs: Dict[str, object] = {
                "dtype": pass_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": device_map,
            }

            # Add flash attention if supported and requested. Flash-Attn 2 is
            # CUDA-only, so MPS hosts (e.g. Mac Studio M3 Ultra) silently skip.
            if use_flash_attention and self.device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # transformers' ``caching_allocator_warmup`` is a CUDA optimization
            # but is invoked on every device. On MPS it tries to allocate a
            # single per-device buffer (e.g. ~405 GiB for V4-Flash dequantized
            # to bf16), which Metal rejects regardless of unified-memory size
            # because it exceeds ``MTLDevice.maxBufferLength``. Neutralize it
            # for MPS loads so the per-shard allocations during state-dict
            # loading can succeed.
            if self.device == "mps":
                _disable_caching_allocator_warmup_for_mps()
                # Allow transformers' FP8 quantizer to keep weights in FP8 on
                # MPS (requires fp4-fp8-for-torch-mps). Without this it forces
                # bf16 dequant and inflates V4-Flash to ~570 GB resident.
                _patch_fp8_quantizer_for_mps()

            # accelerate's ``device_map="auto"`` may decide some shards must
            # spill to disk if the unified-memory budget is tight; supply an
            # ``offload_folder`` so it can do that instead of bailing with
            # "Please provide an offload_folder".
            if device_map == "auto" or isinstance(device_map, dict):
                import os
                import tempfile

                offload_folder = os.path.join(
                    tempfile.gettempdir(),
                    f"solar-host-offload-{self.alias.replace('/', '_').replace(':', '_') or 'model'}",
                )
                os.makedirs(offload_folder, exist_ok=True)
                model_kwargs.setdefault("offload_folder", offload_folder)
                model_kwargs.setdefault("offload_state_dict", True)
                logger.info(f"Using offload folder: {offload_folder}")

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

            # For MPS without device_map, move model manually. With
            # device_map="auto" accelerate has already placed the modules.
            if self.device == "mps" and device_map is None:
                target_device = torch.device(self.device)
                model = model.to(device=target_device)  # type: ignore[call-overload]

            self.model = model

        elif model_type == "classification":
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
            )
            target_device = torch.device(self.device)
            model = model.to(device=target_device)  # type: ignore[call-overload]
            self.model = model

            # Get labels from model config if not provided
            if self.labels is None and hasattr(model.config, "id2label"):
                id2label = model.config.id2label
                if id2label is not None:
                    self.labels = [id2label[i] for i in range(len(id2label))]

        elif model_type == "embedding":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_id,
                dtype=model_dtype,
                trust_remote_code=trust_remote_code,
            )
            target_device = torch.device(self.device)
            model = model.to(device=target_device)  # type: ignore[call-overload]
            self.model = model
            logger.info(f"Normalize embeddings: {self.normalize_embeddings}")

        elif model_type == "vision":
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText as VisionModelCls
            except ImportError:
                from transformers import AutoModelForVision2Seq as VisionModelCls

            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
            )
            self.processor = processor

            model_kwargs: Dict[str, object] = {
                "dtype": model_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": self.device if self.device != "mps" else None,
            }
            if use_flash_attention and self.device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model = VisionModelCls.from_pretrained(model_id, **model_kwargs)
            if self.device == "mps":
                target_device = torch.device(self.device)
                model = model.to(device=target_device)  # type: ignore[call-overload]
            self.model = model

        if self.model is not None:
            self.model.eval()
        logger.info(f"Model loaded successfully on {self.device}")


state = ServerState()
security = HTTPBearer(auto_error=False)


# ============================================================================
# Auth
# ============================================================================


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Verify API key from Authorization header."""
    if not state.api_key:
        return True

    # Check Authorization header
    if credentials and credentials.credentials == state.api_key:
        return True

    # Check X-API-Key header as fallback
    api_key = request.headers.get("X-API-Key")
    if api_key and api_key == state.api_key:
        return True

    raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("HuggingFace server starting...")
    yield
    logger.info("HuggingFace server shutting down...")


# ============================================================================
# App Setup
# ============================================================================


# Custom JSONResponse that ensures UTF-8 encoding
class UTF8JSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")


app = FastAPI(
    title="HuggingFace Server",
    description="OpenAI-compatible API for HuggingFace models",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": state.alias,
        "model_type": state.model_type,
        "device": state.device,
    }


@app.get("/v1/models")
async def list_models(_: bool = Depends(verify_api_key)):
    """List available models.

    Returns both an Ollama-style ``models`` array and an OpenAI-style ``data``
    array (matching llama.cpp's shape), each carrying ``capabilities`` so that
    downstream apps (e.g. solar-control's gateway, the orchestrator) can
    differentiate text vs multimodal models without trial-and-error.
    """
    capabilities = MODEL_TYPE_CAPABILITIES.get(state.model_type, ["completion"])
    return {
        "models": [
            {
                "name": state.alias,
                "model": state.alias,
                "modified_at": "",
                "size": "",
                "digest": "",
                "type": "model",
                "description": "",
                "tags": [""],
                "capabilities": capabilities,
                "parameters": "",
                "details": {
                    "parent_model": "",
                    "format": "huggingface",
                    "family": "",
                    "families": [""],
                    "parameter_size": "",
                    "quantization_level": "",
                },
            }
        ],
        "object": "list",
        "data": [
            {
                "id": state.alias,
                "object": "model",
                "created": state.created_at,
                "owned_by": "huggingface",
                "capabilities": capabilities,
            }
        ],
    }


def _load_image_from_url(url: str):
    """Load a PIL Image from a data URI or http(s) URL.

    Imports PIL lazily so non-vision deployments don't pay the cost.
    """
    from PIL import Image

    if url.startswith("data:"):
        try:
            header, b64 = url.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed data URI: {url[:60]}...") from exc
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        import requests

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    raise ValueError(f"Unsupported image URL scheme: {parsed.scheme!r}")


def _normalize_messages_for_processor(
    messages: List[ChatMessage],
) -> tuple[List[Dict[str, Any]], List[Any]]:
    """Convert OpenAI-style messages into processor-friendly form + image list.

    Each `image_url` content part is downloaded into a PIL image and replaced
    by ``{"type": "image"}`` (the convention most multimodal HF processors expect
    for ``apply_chat_template``).
    """
    norm_messages: List[Dict[str, Any]] = []
    images: List[Any] = []

    for m in messages:
        if m.content is None:
            norm_messages.append({"role": m.role, "content": ""})
            continue
        if isinstance(m.content, str):
            norm_messages.append({"role": m.role, "content": m.content})
            continue

        parts: List[Dict[str, Any]] = []
        for part in m.content:
            ptype = part.get("type")
            if ptype == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                image_url = part.get("image_url", {})
                url = (
                    image_url.get("url") if isinstance(image_url, dict) else image_url
                )
                if not url:
                    raise ValueError("image_url part missing 'url'")
                images.append(_load_image_from_url(url))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported content part type: {ptype!r}")
        norm_messages.append({"role": m.role, "content": parts})

    return norm_messages, images


def _openai_messages_to_dsv4(
    request: ChatCompletionRequest,
) -> List[Dict[str, Any]]:
    """Convert OpenAI-style request messages into the dict shape expected by
    the vendored ``encoding_dsv4.encode_messages``.

    Tools, when present on the request, are attached to the first ``system``
    message (the DSV4 encoder injects the DSML schema block from there). If
    no system message exists, a synthetic empty one is prepended.

    NOTE: Streaming + incremental tool-call parsing is intentionally not
    supported here -- the vendored parser only handles a complete completion.
    """
    out: List[Dict[str, Any]] = []
    for m in request.messages:
        # Normalize ``content`` to a plain string. The DSV4 encoder also
        # supports a ``content_blocks`` form that we don't surface to callers.
        if isinstance(m.content, list):
            text_parts: List[str] = []
            for part in m.content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "DeepSeek-V4 causal model only accepts text content "
                            "parts; multimodal parts are not supported."
                        ),
                    )
            content_str: Optional[str] = "".join(text_parts)
        else:
            content_str = m.content

        msg: Dict[str, Any] = {"role": m.role}
        if content_str is not None:
            msg["content"] = content_str
        if m.reasoning_content is not None:
            msg["reasoning_content"] = m.reasoning_content
        if m.tool_calls:
            # Encoder expects OpenAI-style tool_calls: ``{function: {name, arguments}}``.
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in m.tool_calls
            ]
        if m.tool_call_id is not None:
            msg["tool_call_id"] = m.tool_call_id
        out.append(msg)

    if request.tools:
        tools_payload = [t.model_dump(exclude_none=True) for t in request.tools]
        # Attach tools to the first system message; create one if absent.
        for msg in out:
            if msg.get("role") == "system":
                msg["tools"] = tools_payload
                break
        else:
            out.insert(0, {"role": "system", "content": "", "tools": tools_payload})

    return out


def _dsv4_eos_token_ids(tokenizer: Any) -> Optional[List[int]]:
    """Resolve the DSV4 end-of-sentence token id list, falling back to the
    tokenizer's ``eos_token_id`` if the special token is missing.
    """
    from solar_host.servers.deepseek_v4 import eos_token as dsv4_eos

    ids: List[int] = []
    try:
        tid = tokenizer.convert_tokens_to_ids(dsv4_eos)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    fallback = getattr(tokenizer, "eos_token_id", None)
    if isinstance(fallback, int) and fallback >= 0 and fallback not in ids:
        ids.append(fallback)
    return ids or None


async def _chat_completion_causal_deepseek_v4(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Chat completion implementation for DeepSeek-V4 causal models.

    Uses the vendored ``encoding_dsv4`` encoder/parser instead of the
    tokenizer's (absent) Jinja chat template. Surfaces ``reasoning_content``
    and DSML-format tool calls in the OpenAI response.
    """
    from solar_host.servers.deepseek_v4 import (
        encode_messages,
        parse_message_from_completion_text,
        eos_token as dsv4_eos_token,
    )

    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/chat/completions")

    thinking_mode = request.thinking_mode or state.default_thinking_mode or "thinking"
    reasoning_effort = request.reasoning_effort or state.default_reasoning_effort
    # The upstream encoder only supports {"max", "high", None} for the prefix;
    # silently downgrade unsupported levels.
    if reasoning_effort not in (None, "high", "max"):
        reasoning_effort = None

    dsv4_messages = _openai_messages_to_dsv4(request)

    try:
        prompt = encode_messages(
            dsv4_messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
    except Exception as exc:
        logger.error(f"[ERROR] DSV4 encode failed: {exc}")
        raise HTTPException(
            status_code=400, detail=f"Failed to encode messages: {exc}"
        )

    # ``add_special_tokens=False``: BOS is already inside the encoded prompt.
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=state.max_length,
        add_special_tokens=False,
    )
    inputs = encoded.to(state.device)

    input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
    prompt_tokens: int = int(input_ids.shape[1])
    max_new_tokens = request.max_tokens or (state.max_length - prompt_tokens)

    eos_ids = _dsv4_eos_token_ids(tokenizer)

    # Per the model card: temperature=1.0, top_p=1.0 for both Flash and Pro.
    temperature = request.temperature if request.temperature is not None else 1.0
    top_p = request.top_p if request.top_p is not None else 1.0

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature != 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]
    completion_tokens = len(generated_ids)
    # Decode WITHOUT stripping specials so the parser can find <think>,
    # </think>, ｜DSML｜* and the EOS marker.
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

    # ``parse_message_from_completion_text`` requires a trailing EOS marker.
    if not raw_text.endswith(dsv4_eos_token):
        raw_text = raw_text + dsv4_eos_token

    try:
        parsed = parse_message_from_completion_text(
            raw_text, thinking_mode=thinking_mode
        )
    except Exception as exc:
        logger.error(f"[ERROR] DSV4 parse failed: {exc}; raw={raw_text!r}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse DeepSeek-V4 completion: {exc}",
        )

    parsed_tool_calls = parsed.get("tool_calls") or []
    response_tool_calls: Optional[List[ToolCall]] = None
    if parsed_tool_calls:
        response_tool_calls = [
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ToolFunctionCall(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                ),
            )
            for tc in parsed_tool_calls
        ]

    response_message = ChatMessage(
        role="assistant",
        content=parsed.get("content") or "",
        reasoning_content=parsed.get("reasoning_content") or None,
        tool_calls=response_tool_calls,
    )

    finish_reason = "tool_calls" if response_tool_calls else "stop"

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[COMPLETE] model={state.alias} tokens={completion_tokens} time_ms={elapsed_ms:.1f}"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=state.alias,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=response_message,
                finish_reason=finish_reason,
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


async def _chat_completion_causal(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Chat completion implementation for causal (text-only) models."""
    if state.is_deepseek_v4:
        return await _chat_completion_causal_deepseek_v4(request)

    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/chat/completions")

    # Causal text-only path expects plain string content.
    plain_messages: List[Dict[str, str]] = []
    for m in request.messages:
        if m.content is None:
            continue
        if not isinstance(m.content, str):
            raise HTTPException(
                status_code=400,
                detail="Causal model received non-text content; use a vision model",
            )
        plain_messages.append({"role": m.role, "content": m.content})

    prompt: str
    if hasattr(tokenizer, "apply_chat_template"):
        template_result = tokenizer.apply_chat_template(
            plain_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt = str(template_result)
    else:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in plain_messages])
        prompt += "\nassistant:"

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=state.max_length,
    )
    inputs = encoded.to(state.device)

    input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
    prompt_tokens: int = int(input_ids.shape[1])
    max_new_tokens = request.max_tokens or (state.max_length - prompt_tokens)

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            do_sample=request.temperature != 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[COMPLETE] model={state.alias} tokens={completion_tokens} time_ms={elapsed_ms:.1f}"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=state.alias,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


async def _chat_completion_vision(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Chat completion implementation for vision/multimodal models."""
    state.ensure_loaded()
    model = state.model
    processor = state.processor
    if processor is None:
        raise HTTPException(
            status_code=500, detail="Vision model loaded without a processor"
        )
    assert model is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/chat/completions")

    norm_messages, images = _normalize_messages_for_processor(request.messages)

    template_result = processor.apply_chat_template(
        norm_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt = str(template_result)

    proc_kwargs: Dict[str, Any] = {
        "text": prompt,
        "return_tensors": "pt",
    }
    if images:
        proc_kwargs["images"] = images

    encoded = processor(**proc_kwargs)
    inputs = encoded.to(state.device)

    input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
    prompt_tokens: int = int(input_ids.shape[1])
    max_new_tokens = request.max_tokens or (state.max_length - prompt_tokens)

    tokenizer = getattr(processor, "tokenizer", state.tokenizer)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer else None
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer else None

    with torch.no_grad():
        outputs = model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature or 1.0,
            top_p=request.top_p or 1.0,
            do_sample=request.temperature != 0,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    generated_ids = outputs[0][prompt_tokens:]
    if hasattr(processor, "decode"):
        response_text = processor.decode(generated_ids, skip_special_tokens=True)
    elif tokenizer is not None:
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else:
        raise HTTPException(
            status_code=500, detail="No decoder available for vision model output"
        )
    completion_tokens = len(generated_ids)

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[COMPLETE] model={state.alias} tokens={completion_tokens} time_ms={elapsed_ms:.1f}"
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=state.alias,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: bool = Depends(verify_api_key)
):
    """Chat completion endpoint (OpenAI compatible)."""
    if state.model_type not in ("causal", "vision"):
        raise HTTPException(
            status_code=400,
            detail="Chat completions only available for causal or vision models",
        )

    try:
        if state.model_type == "vision":
            return await _chat_completion_vision(request)
        return await _chat_completion_causal(request)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/completions")
async def completions(request: CompletionRequest, _: bool = Depends(verify_api_key)):
    """Text completion endpoint (OpenAI compatible)."""
    if state.model_type != "causal":
        raise HTTPException(
            status_code=400, detail="Completions only available for causal models"
        )

    # Ensure model is loaded
    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/completions")

    try:
        # Tokenize
        encoded = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=state.max_length,
        )
        inputs = encoded.to(state.device)

        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
        prompt_tokens: int = int(input_ids.shape[1])
        max_new_tokens = request.max_tokens or (state.max_length - prompt_tokens)

        # Generate
        with torch.no_grad():
            outputs = model.generate(  # type: ignore[operator]
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=request.temperature or 1.0,
                top_p=request.top_p or 1.0,
                do_sample=request.temperature != 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode response (only the generated part)
        generated_ids = outputs[0][prompt_tokens:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = len(generated_ids)

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[COMPLETE] model={state.alias} tokens={completion_tokens} time_ms={elapsed_ms:.1f}"
        )

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=state.alias,
            choices=[
                CompletionChoice(
                    index=0,
                    text=response_text,
                    finish_reason="stop",
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        )

    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/classify")
async def classify(request: ClassifyRequest, _: bool = Depends(verify_api_key)):
    """Classification endpoint for SequenceClassification models."""
    if state.model_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Classification only available for classification models",
        )

    # Ensure model is loaded
    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/classify")

    try:
        # Handle single or batch input
        texts: List[str] = (
            request.input if isinstance(request.input, list) else [request.input]
        )

        # Tokenize
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=state.max_length,
        )
        inputs = encoded.to(state.device)

        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
        total_tokens: int = int(input_ids.numel())

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get predictions
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        # Build choices
        choices = []
        for i, prob in enumerate(probs):
            best_idx: int = int(prob.argmax().item())
            best_score: float = float(prob[best_idx].item())

            label = str(best_idx)
            if state.labels and best_idx < len(state.labels):
                label = state.labels[best_idx]

            # Build all_scores if requested
            all_scores: Optional[List[ClassifyScoreItem]] = None
            if request.return_all_scores:
                all_scores = []
                for class_idx in range(prob.shape[0]):
                    class_label = str(class_idx)
                    if state.labels and class_idx < len(state.labels):
                        class_label = state.labels[class_idx]
                    all_scores.append(
                        ClassifyScoreItem(
                            label=class_label,
                            score=float(prob[class_idx].item()),
                        )
                    )

            choices.append(
                ClassifyChoice(
                    index=i,
                    label=label,
                    score=best_score,
                    all_scores=all_scores,
                )
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[COMPLETE] model={state.alias} tokens={total_tokens} time_ms={elapsed_ms:.1f}"
        )

        return ClassifyResponse(
            id=f"clf-{uuid.uuid4().hex[:8]}",
            model=state.alias,
            choices=choices,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        )

    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest, _: bool = Depends(verify_api_key)):
    """Embeddings endpoint (OpenAI compatible).

    Extracts the last hidden state from the model and applies mean pooling
    to generate fixed-size embedding vectors.
    """
    if state.model_type != "embedding":
        raise HTTPException(
            status_code=400,
            detail="Embeddings only available for embedding models",
        )

    # Ensure model is loaded
    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/embeddings")

    try:
        # Handle single or batch input
        texts: List[str] = (
            request.input if isinstance(request.input, list) else [request.input]
        )

        # Tokenize
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=state.max_length,
        )
        inputs = encoded.to(state.device)

        input_ids: torch.Tensor = inputs["input_ids"]  # type: ignore[assignment]
        attention_mask: torch.Tensor = inputs["attention_mask"]  # type: ignore[assignment]
        total_tokens: int = int(input_ids.numel())

        # Forward pass to get last hidden state
        with torch.no_grad():
            outputs = model(**inputs)

        # Get last hidden state: (batch_size, sequence_length, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        # Mean pooling with attention mask
        # Expand attention mask to match hidden state dimensions
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        embeddings_tensor = sum_embeddings / sum_mask

        # Optionally L2 normalize
        if state.normalize_embeddings:
            embeddings_tensor = torch.nn.functional.normalize(
                embeddings_tensor, p=2, dim=1
            )

        # Optional dimension truncation
        if request.dimensions is not None and request.dimensions > 0:
            embeddings_tensor = embeddings_tensor[:, : request.dimensions]

        # Convert to list
        embeddings_list = embeddings_tensor.cpu().tolist()

        # Build response data
        data = [
            EmbeddingData(
                embedding=emb,
                index=i,
            )
            for i, emb in enumerate(embeddings_list)
        ]

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(
            f"[COMPLETE] model={state.alias} tokens={total_tokens} time_ms={elapsed_ms:.1f}"
        )

        return EmbeddingResponse(
            data=data,
            model=state.alias,
            usage={
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            },
        )

    except Exception as e:
        logger.error(f"[ERROR] {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace Model Server")
    parser.add_argument(
        "--model-id", required=True, help="HuggingFace model ID or path"
    )
    parser.add_argument(
        "--model-type",
        choices=["causal", "classification", "embedding", "vision"],
        required=True,
    )
    parser.add_argument("--alias", required=True, help="Model alias")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, required=True, help="Port to bind to")
    parser.add_argument("--api-key", default="", help="API key for authentication")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument(
        "--dtype", default="auto", help="Data type: auto, float16, bfloat16, float32"
    )
    parser.add_argument(
        "--max-length", type=int, default=4096, help="Max sequence length"
    )
    parser.add_argument("--labels", default=None, help="Comma-separated label names")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Trust remote code"
    )
    parser.add_argument(
        "--use-flash-attention", action="store_true", help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="L2 normalize embedding vectors",
    )
    parser.add_argument(
        "--default-thinking-mode",
        choices=["thinking", "chat"],
        default="thinking",
        help=(
            "Default thinking mode for DeepSeek-V4 (and other thinking-capable "
            "models) when the request omits ``thinking_mode``."
        ),
    )
    parser.add_argument(
        "--default-reasoning-effort",
        choices=["low", "medium", "high", "max"],
        default=None,
        help=(
            "Default reasoning effort for DeepSeek-V4 when the request omits "
            "``reasoning_effort``. Only ``high``/``max`` actually change the "
            "encoded prompt; ``low``/``medium`` are accepted for compatibility."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse labels
    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]

    # Set API key
    state.api_key = args.api_key

    # Chat-mode defaults (DeepSeek-V4 etc.)
    state.default_thinking_mode = args.default_thinking_mode
    state.default_reasoning_effort = args.default_reasoning_effort

    # Load model
    state.load_model(
        model_id=args.model_id,
        model_type=args.model_type,
        alias=args.alias,
        device=args.device,
        dtype=args.dtype,
        max_length=args.max_length,
        labels=labels,
        trust_remote_code=args.trust_remote_code,
        use_flash_attention=args.use_flash_attention,
        normalize_embeddings=args.normalize_embeddings,
    )

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
