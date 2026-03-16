# Solar Host

A multi-backend process manager for model inference servers with REST API and WebSocket log streaming.

## Features

- **Multi-Backend Support:**
  - llama.cpp (llama-server) for GGUF models
  - HuggingFace AutoModelForCausalLM for text generation
  - HuggingFace AutoModelForSequenceClassification for classification
  - HuggingFace AutoModel for embeddings (last hidden state with mean pooling)
- **Socket.IO control client** - Connects to solar-control’s `/hosts` namespace for registration, heartbeat, and instance lifecycle (start/stop/restart, config updates). Supports pending-host and rejection events with post-approval sync.
- **Robust instance lifecycle** - Non-blocking process wait, state re-check after startup to avoid start/stop races, and full cleanup of log/state buffers on stop or delete.
- Auto-assign ports starting from 3500
- Persistent configuration with auto-restart on boot
- Real-time log streaming via WebSocket
- REST API for instance management
- API key authentication

## Installation

```bash
# Install core dependencies
pip install -r requirements.txt
```

### Backend-Specific Requirements

**For llama.cpp backend:**
- Install `llama-server` and ensure it's in your PATH

**For HuggingFace backends:**
```bash
pip install torch transformers accelerate
```

## Setup

### 1. Create .env file

Create a `.env` file in the `solar-host/` directory:

```bash
API_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=8001

# Solar-control connection (for Socket.IO registration and lifecycle)
SOLAR_CONTROL_URL=http://localhost:8000
SOLAR_CONTROL_API_KEY=your-solar-control-management-api-key
```

- **API_KEY** - Used by solar-control (and other callers) to access this host’s REST API.
- **SOLAR_CONTROL_URL** - Base URL of solar-control (HTTP; Socket.IO connects to the same origin).
- **SOLAR_CONTROL_API_KEY** - Management API key from solar-control. The host uses it to connect to the `/hosts` namespace; it must be approved via the management API or WebUI before it appears in the gateway pool.

### 2. Start the server

```bash
# Start the server (from solar-host directory)
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or with auto-reload for development
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

The server will:
- Create `config.json` automatically (if it doesn't exist)
- Create `logs/` directory for instance logs
- Auto-restart any instances that were running before shutdown

### 3. Verify it's running

```bash
curl http://localhost:8001/health
# Should return: {"status":"healthy","service":"solar-host","version":"2.0.0"}
```

### 4. Access Swagger UI

Open your browser to: **http://localhost:8001/docs**

1. Click the **"Authorize"** button
2. Enter your API key from `.env` file
3. Click **"Authorize"** and then **"Close"**
4. Now you can use the interactive API documentation!

## Backend Types

Solar Host supports four backend types:

| Backend Type | Model Type | Endpoints Supported |
|--------------|------------|---------------------|
| `llamacpp` | GGUF models via llama-server | `/v1/chat/completions`, `/v1/completions` |
| `huggingface_causal` | HuggingFace AutoModelForCausalLM | `/v1/chat/completions`, `/v1/completions` |
| `huggingface_classification` | HuggingFace AutoModelForSequenceClassification | `/v1/classify` |
| `huggingface_embedding` | HuggingFace AutoModel (last hidden state) | `/v1/embeddings` |

## Managing Instances

### Creating a llama.cpp Instance

```bash
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "backend_type": "llamacpp",
      "model": "/path/to/model.gguf",
      "alias": "llama-3:8b",
      "threads": 4,
      "n_gpu_layers": 999,
      "temp": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "min_p": 0.05,
      "ctx_size": 8192,
      "host": "0.0.0.0",
      "api_key": "instance-key"
    }
  }'
```

### Creating a HuggingFace Causal LM Instance

```bash
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "backend_type": "huggingface_causal",
      "model_id": "meta-llama/Llama-2-7b-chat-hf",
      "alias": "llama2-hf:7b",
      "device": "auto",
      "dtype": "auto",
      "max_length": 4096,
      "trust_remote_code": false,
      "use_flash_attention": true,
      "host": "0.0.0.0",
      "api_key": "instance-key"
    }
  }'
```

### Creating a HuggingFace Classification Instance

```bash
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "backend_type": "huggingface_classification",
      "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
      "alias": "sentiment:distilbert",
      "device": "auto",
      "dtype": "auto",
      "max_length": 512,
      "labels": ["negative", "positive"],
      "host": "0.0.0.0",
      "api_key": "instance-key"
    }
  }'
```

### Creating a HuggingFace Embedding Instance

```bash
curl -X POST http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "backend_type": "huggingface_embedding",
      "model_id": "sentence-transformers/all-MiniLM-L6-v2",
      "alias": "embed:minilm",
      "device": "auto",
      "dtype": "auto",
      "max_length": 512,
      "normalize_embeddings": true,
      "host": "0.0.0.0",
      "api_key": "instance-key"
    }
  }'
```

### Starting an Instance

```bash
curl -X POST http://localhost:8001/instances/{instance-id}/start \
  -H "X-API-Key: your-secret-key-here"
```

### Viewing All Instances

```bash
curl http://localhost:8001/instances \
  -H "X-API-Key: your-secret-key-here"
```

### Stopping an Instance

```bash
curl -X POST http://localhost:8001/instances/{instance-id}/stop \
  -H "X-API-Key: your-secret-key-here"
```

## API Endpoints

### Instance Management

- `POST /instances` - Create new instance
- `GET /instances` - List all instances
- `GET /instances/{id}` - Get instance details
- `PUT /instances/{id}` - Update instance config
- `DELETE /instances/{id}` - Remove instance
- `POST /instances/{id}/start` - Start instance
- `POST /instances/{id}/stop` - Stop instance
- `POST /instances/{id}/restart` - Restart instance
- `GET /instances/{id}/state` - Get runtime state
- `GET /instances/{id}/last-generation` - Get last generation metrics

### WebSocket

- `WS /instances/{id}/logs` - Stream logs with sequence numbers
- `WS /instances/{id}/state` - Stream runtime state updates

### System

- `GET /health` - Health check
- `GET /memory` - GPU/RAM memory usage

## Authentication

All requests require an `X-API-Key` header with your configured API key from the `.env` file.

## Configuration Reference

### llama.cpp Config Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `backend_type` | No | `"llamacpp"` | Backend type identifier |
| `model` | Yes | - | Full path to the GGUF model file |
| `alias` | Yes | - | Model alias (e.g., "llama-3:8b") used for routing |
| `threads` | No | 1 | Number of CPU threads to use |
| `n_gpu_layers` | No | 999 | Number of layers to offload to GPU (999 = all) |
| `temp` | No | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | No | 1.0 | Top-p sampling (0.0-1.0) |
| `top_k` | No | 0 | Top-k sampling (0 = disabled) |
| `min_p` | No | 0.0 | Min-p sampling (0.0-1.0) |
| `ctx_size` | No | 131072 | Context window size |
| `chat_template_file` | No | - | Path to Jinja chat template file |
| `special` | No | false | Enable llama-server `--special` flag |
| `ot` | No | - | Override tensor string (passed as `-ot` flag to llama-server) |
| `model_type` | No | `"llm"` | Model type: `"llm"`, `"embedding"`, or `"reranker"` |
| `pooling` | No | - | Pooling strategy for embedding models: `"none"`, `"mean"`, `"cls"`, `"last"`, `"rank"` (only valid when `model_type` is `"embedding"`) |
| `host` | No | "0.0.0.0" | Host to bind to |
| `port` | No | auto | Port (auto-assigned if not specified) |
| `api_key` | Yes | - | API key for this instance |

### HuggingFace Causal LM Config Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `backend_type` | Yes | - | Must be `"huggingface_causal"` |
| `model_id` | Yes | - | HuggingFace model ID or local path |
| `alias` | Yes | - | Model alias for routing |
| `device` | No | `"auto"` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `dtype` | No | `"auto"` | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `max_length` | No | 4096 | Maximum sequence length |
| `trust_remote_code` | No | false | Trust remote code from HuggingFace |
| `use_flash_attention` | No | true | Use Flash Attention 2 if available |
| `host` | No | "0.0.0.0" | Host to bind to |
| `port` | No | auto | Port (auto-assigned if not specified) |
| `api_key` | Yes | - | API key for this instance |

### HuggingFace Classification Config Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `backend_type` | Yes | - | Must be `"huggingface_classification"` |
| `model_id` | Yes | - | HuggingFace model ID or local path |
| `alias` | Yes | - | Model alias for routing |
| `device` | No | `"auto"` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `dtype` | No | `"auto"` | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `max_length` | No | 512 | Maximum sequence length |
| `labels` | No | auto | Label names (auto-detected from model if not provided) |
| `trust_remote_code` | No | false | Trust remote code from HuggingFace |
| `host` | No | "0.0.0.0" | Host to bind to |
| `port` | No | auto | Port (auto-assigned if not specified) |
| `api_key` | Yes | - | API key for this instance |

### HuggingFace Embedding Config Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `backend_type` | Yes | - | Must be `"huggingface_embedding"` |
| `model_id` | Yes | - | HuggingFace model ID or local path |
| `alias` | Yes | - | Model alias for routing |
| `device` | No | `"auto"` | Device: `auto`, `cuda`, `mps`, `cpu` |
| `dtype` | No | `"auto"` | Data type: `auto`, `float16`, `bfloat16`, `float32` |
| `max_length` | No | 512 | Maximum sequence length |
| `normalize_embeddings` | No | true | L2 normalize output embedding vectors |
| `trust_remote_code` | No | false | Trust remote code from HuggingFace |
| `host` | No | "0.0.0.0" | Host to bind to |
| `port` | No | auto | Port (auto-assigned if not specified) |
| `api_key` | Yes | - | API key for this instance |

### Device Options

| Device | Description |
|--------|-------------|
| `auto` | Automatically select best available (CUDA > MPS > CPU) |
| `cuda` | NVIDIA GPU (requires CUDA) |
| `mps` | Apple Silicon GPU (macOS) |
| `cpu` | CPU only |

## Example Configurations

### llama.cpp - Small Model

```json
{
  "backend_type": "llamacpp",
  "model": "/models/llama-3-7b.gguf",
  "alias": "llama-3:7b",
  "threads": 4,
  "n_gpu_layers": 999,
  "temp": 0.7,
  "top_p": 0.9,
  "ctx_size": 8192,
  "api_key": "llama3-7b-key"
}
```

### llama.cpp - Large Model with Custom Template

```json
{
  "backend_type": "llamacpp",
  "model": "/models/gpt-oss-120b-F16.gguf",
  "alias": "gpt-oss:120b",
  "threads": 1,
  "n_gpu_layers": 999,
  "ctx_size": 131072,
  "chat_template_file": "/models/templates/harmony.jinja",
  "api_key": "gpt-oss-key"
}
```

### HuggingFace - Text Generation

```json
{
  "backend_type": "huggingface_causal",
  "model_id": "microsoft/phi-2",
  "alias": "phi-2:2.7b",
  "device": "cuda",
  "dtype": "float16",
  "max_length": 2048,
  "api_key": "phi2-key"
}
```

### HuggingFace - Sentiment Classification

```json
{
  "backend_type": "huggingface_classification",
  "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "alias": "sentiment:roberta",
  "device": "cuda",
  "max_length": 512,
  "labels": ["negative", "neutral", "positive"],
  "api_key": "sentiment-key"
}
```

### HuggingFace - Embedding Model

```json
{
  "backend_type": "huggingface_embedding",
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "alias": "embed:minilm",
  "device": "cuda",
  "max_length": 512,
  "normalize_embeddings": true,
  "api_key": "embed-key"
}
```

## File Structure

```
solar-host/
├── .env                    # Configuration (not in git)
├── config.json             # Auto-generated instance storage (not in git)
├── logs/                   # Auto-generated log directory (not in git)
├── app/
│   ├── backends/           # Backend runners
│   │   ├── base.py         # Abstract BackendRunner
│   │   ├── llamacpp.py     # llama.cpp runner
│   │   └── huggingface.py  # HuggingFace runner
│   ├── models/             # Pydantic models
│   │   ├── base.py         # Base models
│   │   ├── llamacpp.py     # llama.cpp config
│   │   └── huggingface.py  # HuggingFace configs
│   ├── servers/            # Standalone server processes
│   │   └── hf_server.py    # HuggingFace model server
│   ├── routes/             # API routes
│   ├── config.py           # Configuration management
│   ├── main.py             # FastAPI application
│   └── process_manager.py  # Process lifecycle management
├── requirements.txt
└── README.md
```

## Troubleshooting

### Solar-host won't start

**Error: "Address already in use"**
- Another service is using port 8001
- Solution: Change `PORT` in `.env` or stop the other service

**Error: "No module named 'app'"**
- You're not in the correct directory
- Solution: `cd solar-host` and run from there

### llama.cpp Instance fails to start

1. **Verify llama-server is installed:**
   ```bash
   which llama-server
   ```

2. **Check model path:**
   ```bash
   ls -lh /path/to/your/model.gguf
   ```

3. **Check instance logs in `logs/` directory**

### HuggingFace Instance fails to start

1. **Verify dependencies:**
   ```bash
   python -c "import torch; import transformers; print('OK')"
   ```

2. **Check CUDA availability (if using GPU):**
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```

3. **Check MPS availability (macOS):**
   ```bash
   python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
   ```

4. **Check instance logs in `logs/` directory**

### Instance keeps retrying and failing

- Solar-host will retry starting an instance up to 2 times
- Check the `error_message` field:
  ```bash
  curl http://localhost:8001/instances/{instance-id} \
    -H "X-API-Key: your-key" | jq '.error_message'
  ```

### Conda Environment

When running solar-host from a conda environment, HuggingFace server subprocesses automatically inherit the same environment. Just ensure all dependencies are installed in your conda environment:

```bash
conda activate your-env
pip install torch transformers accelerate
```

## Integration with Solar Control

Solar-host connects to solar-control over **Socket.IO** (namespace `/hosts`). Set `SOLAR_CONTROL_URL` and `SOLAR_CONTROL_API_KEY` in `.env`. On startup the host registers and appears in solar-control’s **pending** list until approved.

**Approve the host** (via solar-control management API or WebUI):

```bash
# List pending hosts
curl http://your-control-server:8000/api/hosts/pending \
  -H "X-API-Key: your-management-api-key"

# Approve (use pending_id from the list)
curl -X POST http://your-control-server:8000/api/hosts/pending/{pending_id}/approve \
  -H "X-API-Key: your-management-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "GPU Server 1",
    "url": "http://192.168.1.100:8001",
    "api_key": "your-solar-host-api-key"
  }'
```

Alternatively, create a host directly (no pending step) with `POST /api/hosts` and the same JSON body.

Once approved, instances are accessible through solar-control’s OpenAI-compatible gateway:
- `/v1/chat/completions` - Chat completion (llamacpp, huggingface_causal)
- `/v1/completions` - Text completion (llamacpp, huggingface_causal)
- `/v1/classify` - Classification (huggingface_classification)
- `/v1/embeddings` - Embeddings (huggingface_embedding)

## Backward Compatibility

Existing configurations without `backend_type` are automatically treated as `llamacpp` instances. No migration required.
