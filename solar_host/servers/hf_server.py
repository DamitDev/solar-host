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
from typing import Any, Optional, List, Dict, Union, TYPE_CHECKING
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


class ChatMessage(BaseModel):
    role: str
    # Either a plain string (text-only) or an OpenAI-style list of content parts
    # like [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"..."}}]
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


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
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.model_type = model_type
        self.alias = alias
        self.device = self.get_device(device)
        self.max_length = max_length
        self.labels = labels
        self.normalize_embeddings = normalize_embeddings

        model_dtype = self.get_dtype(dtype)

        logger.info(f"Loading model: {model_id}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Dtype: {model_dtype}")

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

            model_kwargs: Dict[str, object] = {
                "dtype": model_dtype,
                "trust_remote_code": trust_remote_code,
                "device_map": self.device if self.device != "mps" else None,
            }

            # Add flash attention if supported and requested
            if use_flash_attention and self.device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

            # For MPS, move model manually
            if self.device == "mps":
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
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if not url:
                    raise ValueError("image_url part missing 'url'")
                images.append(_load_image_from_url(url))
                parts.append({"type": "image"})
            else:
                raise ValueError(f"Unsupported content part type: {ptype!r}")
        norm_messages.append({"role": m.role, "content": parts})

    return norm_messages, images


async def _chat_completion_causal(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    """Chat completion implementation for causal (text-only) models."""
    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/chat/completions")

    # Causal text-only path expects plain string content.
    plain_messages: List[Dict[str, str]] = []
    for m in request.messages:
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


async def _chat_completion_vision(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse labels
    labels = None
    if args.labels:
        labels = [label.strip() for label in args.labels.split(",")]

    # Set API key
    state.api_key = args.api_key

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
