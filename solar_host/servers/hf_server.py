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
import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Union, TYPE_CHECKING
from contextlib import asynccontextmanager

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
    content: str
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
    """List available models (OpenAI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": state.alias,
                "object": "model",
                "created": state.created_at,
                "owned_by": "huggingface",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest, _: bool = Depends(verify_api_key)
):
    """Chat completion endpoint (OpenAI compatible)."""
    if state.model_type != "causal":
        raise HTTPException(
            status_code=400, detail="Chat completions only available for causal models"
        )

    # Ensure model is loaded
    state.ensure_loaded()
    model = state.model
    tokenizer = state.tokenizer
    assert model is not None and tokenizer is not None  # Type narrowing

    start_time = time.time()
    logger.info(f"[REQUEST] model={state.alias} endpoint=/v1/chat/completions")

    try:
        # Build prompt from messages
        prompt: str
        if hasattr(tokenizer, "apply_chat_template"):
            template_result = tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            # apply_chat_template returns str when tokenize=False
            prompt = str(template_result)
        else:
            # Fallback: simple concatenation
            prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
            prompt += "\nassistant:"

        # Tokenize
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

        # Decode response
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
        "--model-type", choices=["causal", "classification", "embedding"], required=True
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
