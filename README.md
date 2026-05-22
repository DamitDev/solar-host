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
# Basic install (llama.cpp backend only)
pip install solar-host

# With HuggingFace backend support
pip install solar-host[huggingface]

# With NVIDIA GPU monitoring
pip install solar-host[nvidia]

# Everything
pip install solar-host[all]

# Development (editable install with test dependencies)
pip install -e ".[all,dev]"
```

### Backend-Specific Requirements

**For llama.cpp backend:**
- Install `llama-server` and ensure it's in your PATH

**For HuggingFace backends:**
- Install with the `huggingface` extra: `pip install solar-host[huggingface]`

## Setup

### 1. Create .env file

Create a `.env` file in the `solar-host/` directory:

```bash
API_KEY=your-secret-key-here
HOST=0.0.0.0
PORT=8001
MODELS_DIR=./models

# Solar-control connection (for Socket.IO registration and lifecycle)
SOLAR_CONTROL_URL=http://localhost:8000
SOLAR_CONTROL_API_KEY=your-solar-control-management-api-key
```

- **API_KEY** - Used by solar-control (and other callers) to access this host’s REST API.
- **MODELS_DIR** - Path to the models directory. Used for disk space reporting in the `/health` endpoint. Defaults to `./models`.
- **SOLAR_CONTROL_URL** - Base URL of solar-control (HTTP; Socket.IO connects to the same origin).
- **SOLAR_CONTROL_API_KEY** - Management API key from solar-control. The host uses it to connect to the `/hosts` namespace; it must be approved via the management API or WebUI before it appears in the gateway pool.

### 2. Start the server

```bash
# Start the server (reads HOST and PORT from .env)
solar-host

# Or with uvicorn directly (e.g. for --reload during development)
uvicorn solar_host.main:app --host 0.0.0.0 --port 8001 --reload
```

The server will:
- Create `config.json` automatically (if it doesn't exist)
- Create `logs/` directory for instance logs
- Auto-restart any instances that were running before shutdown

### 3. Verify it's running

```bash
curl http://localhost:8001/health
# Should return: {"status":"healthy","service":"solar-host","version":"2.0.0","disk":{"total_gb":500,"used_gb":120,"available_gb":380}}
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
├── pyproject.toml          # Package metadata and dependencies
├── solar_host/
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
│   ├── models_manager.py   # Managed models directory and manifest
│   └── process_manager.py  # Process lifecycle management
├── tests/
└── README.md
```

## Troubleshooting

### Solar-host won't start

**Error: "Address already in use"**
- Another service is using port 8001
- Solution: Change `PORT` in `.env` or stop the other service

**Error: "No module named 'solar_host'"**
- The package is not installed
- Solution: `pip install solar-host` or `pip install -e .` for development

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

## GPU Execution

Solar Host supports NVIDIA GPU execution for step containers via the `gpu` field on each step definition. GPU access requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html) to be installed and the Docker daemon to have the `nvidia` runtime registered.

### Job JSON examples

```json
{ "gpu": { "count": 1 } }
```
One GPU — Docker picks the device. Recommended default on multi-GPU hosts.

```json
{ "gpu": { "count": -1 } }
```
All available GPUs (explicit replacement for the legacy `gpu: true` boolean).

```json
{ "gpu": { "device_ids": ["0"] } }
```
Pin a specific GPU by index or UUID.

> **Note:** `"gpu": {}` (empty object) is a validation error. At least one of `count` or `device_ids` must be provided.

### NVIDIA environment variables

When a step requests GPU access, the following environment variables are injected automatically (callers may override them via `step.environment`):

| Variable | Default |
|---|---|
| `NVIDIA_VISIBLE_DEVICES` | `all` |
| `NVIDIA_DRIVER_CAPABILITIES` | `compute,utility` |

### Manual verification

To verify the NVIDIA Container Toolkit is working on a host without running a full job:

```bash
docker run --rm --gpus '"device=0"' nvidia/cuda:12.0-base nvidia-smi
```

## Step Log Streaming

Solar Host captures stdout/stderr from every step container and makes them available in two complementary ways.

### Data flow

```
Container stdout/stderr
        │
        ▼ (demuxed)
 JobStepExecutor._stream_logs()
        │
        ├──► JOBS_DIR/<job-id>/logs/<step>.log   (durable, combined)
        │
        └──► StepLogBuffer.append()
                  │
                  ├──► bounded in-memory deque  (latest 1000 lines)
                  └──► emit queue
                            │
                            ▼ (every 100 ms)
                  broadcast_step_log_batch()
                            │
                            ▼
                  Solar Control  →  "step_log" Socket.IO event
```

### Event payload

Each `step_log` event carries an `entries` array.  A normal log entry:

```json
{
  "job_id":     "job-a1b2",
  "step_name":  "train",
  "step_index": 2,
  "stream":     "stdout",
  "seq":        42,
  "timestamp":  "2026-05-21T19:00:00.123456+00:00",
  "line":       "Epoch 1/10 loss=0.42"
}
```

The final entry for a step is a **completion marker** on the same event type:

```json
{
  "job_id":     "job-a1b2",
  "step_name":  "train",
  "step_index": 2,
  "stream":     "stdout",
  "seq":        43,
  "timestamp":  "...",
  "line":       "",
  "completed":  true,
  "exit_code":  0
}
```

Using the same event shape for both regular lines and the completion marker means downstream consumers (S-032, Solar WebUI) only need to handle one event type.

### Durability vs. real-time

| | Durable | Real-time |
|---|---|---|
| Host log files | ✅ `JOBS_DIR/<job-id>/logs/<step>.log` | ❌ |
| Socket.IO `step_log` | ❌ best-effort, lost on disconnect | ✅ |

The host-side log file is always written regardless of Solar Control connectivity. It is the authoritative record.

### Reconnect behaviour

- `SolarControlClient._emit` is a no-op while disconnected — jobs never block or crash waiting for the WebSocket.
- The emit queue uses `put_nowait` with a 10 000-entry cap; entries are silently dropped when the queue is full (e.g. sustained disconnect + high-frequency logging).
- On reconnect, only new lines from that point forward are streamed; the durable log file can be read for historical lines.

## Job Lifecycle Events (S-026)

Solar Host emits structured Socket.IO lifecycle events at every job and step state transition so Solar Control and downstream consumers (SuperNova, Solar WebUI) can monitor progress without polling.

### Event catalog

| Socket.IO event  | Trigger                                                    | Key payload fields                                                                                  |
| ---------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `job_started`    | After `store.add(running)` in `JobExecutor.run_job`        | `job_id`, `host_id`, `name`, `status`, `timestamp`                                                 |
| `step_started`   | After `store.update_step(running)` in `JobStepExecutor`    | `job_id`, `host_id`, `step_name`, `step_index`, `status`, `timestamp`                              |
| `step_completed` | Successful container exit in `_wait_and_record`            | `job_id`, `host_id`, `step_name`, `step_index`, `status`, `timestamp`, `duration_s`, `exit_code`   |
| `step_failed`    | Non-zero exit or start error in `_wait_and_record` / `run` | `job_id`, `host_id`, `step_name`, `step_index`, `status`, `timestamp`, `duration_s`, `exit_code`, `error_summary` |
| `job_completed`  | All steps succeeded — `_finalise_job` completed path       | `job_id`, `host_id`, `status`, `timestamp`, `workspace_path`, `retention_deadline`                 |
| `job_failed`     | Any step failed or unexpected exception                    | `job_id`, `host_id`, `status`, `timestamp`, `error_message`                                        |
| `job_cancelled`  | Cancellation signal set — `_finalise_job` cancelled path   | `job_id`, `host_id`, `status`, `timestamp`                                                         |

`host_id` is the value received in `registration_ack` from Solar Control; it is `null` when the host has not yet registered.

`retention_deadline` is `finished_at + retention_hours` (ISO 8601 string).

`error_summary` for `step_failed` contains the last N lines of stderr captured in `ContainerNonZeroExitError.last_stderr_lines`.

### Consistency guarantee

Each lifecycle event is emitted **immediately after the matching `JobStore` mutation in the same coroutine**. A consumer that receives a `job_completed` event can safely call `GET /jobs/{id}` and find the status already set to `completed`.

### Fire-and-forget semantics

- Events are emitted directly by Socket.IO event name (e.g. `sio.emit("job_started", data)`) — no envelope, no batching.
- `SolarControlClient.send_job_lifecycle` is a no-op when disconnected; jobs never block waiting for the WebSocket.
- There is no retry queue for lifecycle events. State is always recoverable from `GET /jobs/{id}`.

### Implementation details

```
solar_host/jobs/events.py        — payload builders + async emit_* functions
solar_host/ws_client.py          — send_job_lifecycle / broadcast_job_lifecycle
solar_host/jobs/executor.py      — job_started, job_completed, job_failed, job_cancelled
solar_host/jobs/step_executor.py — step_started, step_completed, step_failed
```

## Backward Compatibility

Existing configurations without `backend_type` are automatically treated as `llamacpp` instances. No migration required.

## Job REST API (S-027)

> **Intended callers: Solar Control only.** These endpoints are not part of the public API surface.

Solar Host exposes three endpoints for submitting and managing containerised multi-step jobs. All endpoints require the `X-API-Key` header (same key as every other endpoint).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/jobs` | Submit a new job for background execution |
| `GET` | `/jobs/{job_id}` | Inspect the current state of a job |
| `DELETE` | `/jobs/{job_id}` | Cancel a running job and delete its workspace |

### `POST /jobs` — Submit a job

**Request body** (`JobDefinition` shape):

```json
{
  "job_id": "my-job-001",
  "name": "Fine-tune run",
  "steps": [
    {
      "name": "train",
      "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
      "command": ["python", "train.py"],
      "environment": {"EPOCHS": "5"},
      "gpu": {"count": 1}
    }
  ],
  "submission_id": "ctrl-sub-abc123",
  "correlation_id": "workflow-xyz"
}
```

- `submission_id` and `correlation_id` are optional strings Solar Control can use to correlate host jobs with its own records.
- `job_id` must be a simple alphanumeric slug (no path traversal characters).

**Response — 202 Accepted:**

```json
{
  "job_id": "my-job-001",
  "status": "running",
  "workspace_path": "/var/solar/jobs/my-job-001",
  "submission_id": "ctrl-sub-abc123",
  "correlation_id": "workflow-xyz"
}
```

**Error codes:**

| Code | Cause |
|------|-------|
| 400 | Invalid `job_id`, empty `steps`, or GPU validation error |
| 409 | A job with the same `job_id` already exists in the store |
| 503 | Docker daemon is unavailable (executor disabled) |
| 507 | Insufficient disk space |

### `GET /jobs/{job_id}` — Inspect a job

**Response — 200 OK:**

```json
{
  "job_id": "my-job-001",
  "name": "Fine-tune run",
  "status": "running",
  "current_step_index": 0,
  "workspace_path": "/var/solar/jobs/my-job-001",
  "created_at": "2026-05-21T19:00:00.000000+00:00",
  "started_at": "2026-05-21T19:00:01.000000+00:00",
  "finished_at": null,
  "retention_hours": 24.0,
  "error_message": null,
  "submission_id": "ctrl-sub-abc123",
  "correlation_id": "workflow-xyz",
  "steps": [
    {
      "name": "train",
      "status": "running",
      "container_id": "abc123def456",
      "started_at": "2026-05-21T19:00:01.500000+00:00",
      "finished_at": null,
      "duration_s": null,
      "exit_code": null,
      "error_message": null,
      "log_file": "/var/solar/jobs/my-job-001/logs/train.log",
      "recent_logs": [
        {
          "seq": 1,
          "stream": "stdout",
          "line": "Epoch 1/5 loss=0.87",
          "timestamp": "2026-05-21T19:00:05.000000+00:00"
        }
      ]
    }
  ]
}
```

- `log_file` is the host-side path to the durable combined stdout/stderr log for that step.
- `recent_logs` is a tail of the in-memory log buffer (up to 100 entries). Real-time streaming is available via Socket.IO `step_log` events (S-025).

**Error codes:**

| Code | Cause |
|------|-------|
| 404 | Unknown `job_id` |

### `DELETE /jobs/{job_id}` — Cancel a job

Performs synchronous cancellation per S-021 §6.3:

1. Signals the cancel event and stops the active container.
2. Waits up to 10 s for the `run_job` task to reach a terminal state.
3. Deletes the workspace directory.
4. Removes the job from the in-memory store and log buffer.

**Response — 200 OK:**

```json
{"detail": "cancelled", "job_id": "my-job-001"}
```

**Error codes:**

| Code | Cause |
|------|-------|
| 404 | Unknown `job_id` |
| 409 | Job is already in a terminal state (`completed`, `failed`, or `cancelled`) — the cleanup loop will remove it |

### Example curl commands

```bash
API_KEY="your-secret-key-here"
HOST="http://localhost:8001"

# Submit a job
curl -s -X POST "$HOST/jobs" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "demo-001",
    "name": "Demo job",
    "steps": [
      {
        "name": "hello",
        "image": "alpine:3.19",
        "command": ["echo", "hello from Solar Host"]
      }
    ]
  }' | jq .

# Poll state
curl -s "$HOST/jobs/demo-001" -H "X-API-Key: $API_KEY" | jq .status

# Cancel (if still running)
curl -s -X DELETE "$HOST/jobs/demo-001" -H "X-API-Key: $API_KEY" | jq .
```
