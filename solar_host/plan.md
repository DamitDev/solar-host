---
name: S-023 Job Executor
overview: "Implement the sequential Docker step executor for solar-host (S-023): job data models, workspace manager, in-memory job store, async executor with fail-fast and cancellation, background retention cleanup, and comprehensive unit tests."
todos:
  - id: step-1-models
    content: "Job data models: enums (JobStatus, StepStatus), input models (StepDefinition, JobDefinition), state models (StepState, JobState), error classes in solar_host/jobs/models.py and solar_host/jobs/errors.py"
    status: pending
  - id: step-2-workspace
    content: "Workspace manager: validate_job_id, check_disk_space, create_workspace, build_job_json, delete_workspace in solar_host/jobs/workspace.py"
    status: pending
  - id: step-3-store
    content: "Job store: thread-safe in-memory JobStore with add/get/update/update_step/remove in solar_host/jobs/store.py"
    status: pending
  - id: step-4-docker-ext
    content: Extend DockerService.create_container with is_preparation_step flag (models/data RW vs RO) + update existing tests
    status: pending
  - id: step-5-executor
    content: "Job executor: async JobExecutor with sequential step execution, env building, log capture, fail-fast, cancellation in solar_host/jobs/executor.py"
    status: pending
  - id: step-6-integration
    content: Background cleanup loop, package __init__.py, wire into main.py lifespan
    status: pending
  - id: step-7-tests
    content: "Unit tests: test_job_models.py, test_workspace.py, test_job_store.py, test_job_executor.py, test_docker_service.py additions"
    status: pending
isProject: false
---

# S-023: Sequential Docker Step Executor

## Context

Solar Host needs a job runner that executes an ordered list of Docker step containers. Each step mounts the same ephemeral workspace (S-021 layout), runs to completion, and passes files to later steps via that workspace. The executor composes [DockerService](host/solar_host/docker/service.py) (S-022) primitives without knowing container internals.

The executor does **not** own REST routes (S-027), Socket.IO events (S-026), or real-time log forwarding (S-025) -- those are downstream consumers. It **does** capture step logs to files and provide in-memory state for `GET /jobs/{id}`.

## New Package: `solar_host/jobs/`

```
solar_host/jobs/
    __init__.py       # re-exports public API
    models.py         # Pydantic models and enums for job/step definitions and state
    workspace.py      # Workspace creation, disk checks, job.json/training.json writing, cleanup
    executor.py       # Async JobExecutor: sequential steps, fail-fast, cancellation, log capture
    store.py          # Thread-safe in-memory job state store
```

---

## Step 1: Job Data Models (`solar_host/jobs/models.py`)

Define the Pydantic models and enums that represent job definitions (input) and job state (runtime tracking).

**Enums:**

- `JobStatus`: `pending`, `running`, `completed`, `failed`, `cancelled`
- `StepStatus`: `pending`, `running`, `completed`, `failed`, `cancelled`

**Input models (what the caller submits):**

- `StepDefinition`: `name` (str), `image` (str), `environment` (dict[str, str], step-specific vars from S-021 Section 4.3), `gpu` (bool, default False), `is_preparation_step` (bool, default `False`) — when `True`, `models/` and `data/` are bind-mounted **read-write** (e.g. `download_model`, `download_dataset`); when `False`, both are **read-only** (e.g. `train`, `convert_model`). `output/` and `config/` stay RW for every step; HF cache mount unchanged.
- `JobDefinition`: `job_id` (str), `name` (str), `steps` (list[StepDefinition]), `retention_hours` (float, default 24), `min_free_disk_gb` (float, default from settings), `base_model_uri` (str, optional), `training_data_uri` (str, optional), `training_config` (dict, optional inline config), `model_selection` (dict, optional), `deployment` (dict, optional)

**State models (runtime tracking):**

- `StepState`: `name`, `status` (StepStatus), `container_id` (optional), `started_at` (optional datetime), `finished_at` (optional datetime), `duration_s` (optional float), `exit_code` (optional int), `error_message` (optional str)
- `JobState`: `job_id`, `name`, `status` (JobStatus), `steps` (list[StepState]), `current_step_index` (int, -1 initially), `workspace_path` (str), `created_at` (datetime), `started_at` (optional), `finished_at` (optional), `retention_hours` (float), `error_message` (optional)

---

## Step 2: Workspace Manager (`solar_host/jobs/workspace.py`)

Handles all filesystem operations for the job workspace per S-021 lifecycle.

**Functions:**

- `validate_job_id(job_id: str) -> None`: Reject IDs containing `/`, `..`, or non-filesystem-safe characters. Raise `ValueError` on invalid input. (S-021 Section 8.2)
- `check_disk_space(jobs_dir: Path, min_free_gb: float) -> None`: Check free space on the `JOBS_DIR` partition. Raise a custom `InsufficientDiskError` if below threshold. Reuse `shutil.disk_usage`. (S-021 Section 6.1, 6.2)
- `create_workspace(job_def: JobDefinition, settings: Settings) -> Path`: Create `JOBS_DIR/<job-id>/{models,data,output,config,logs}` with `0o755` permissions, owned by `CONTAINER_UID:CONTAINER_GID`. Write `job.json` into `config/`. If `job_def.training_config` is provided, write `training.json` into `config/`. Return the workspace path. (S-021 Section 6.1)
- `build_job_json(job_def: JobDefinition) -> dict`: Build the `job.json` content from the job definition (fields from S-021 Section 5.2: job_id, name, created_at, pipeline, base_model_uri, training_data_uri, model_selection, deployment, retention_hours, empty `steps` object).
- `delete_workspace(workspace_path: Path) -> None`: `shutil.rmtree` the workspace directory. Log errors but don't raise on missing dirs.

**New error in `solar_host/jobs/errors.py`:**

- `InsufficientDiskError(required_gb, available_gb)`
- `InvalidJobIdError(job_id, reason)`
- `WorkspaceError(job_id, reason)` -- base for workspace-related failures

---

## Step 3: Job Store (`solar_host/jobs/store.py`)

Thread-safe in-memory store for active and recent job states. No persistence to disk -- this satisfies the spec's "in-memory or local state for GET /jobs/{id}".

**Class `JobStore`:**

- Internal `dict[str, JobState]` guarded by `threading.Lock`
- `add(state: JobState) -> None`
- `get(job_id: str) -> JobState | None`
- `get_all() -> list[JobState]`
- `update(job_id: str, **kwargs) -> None` -- partial update of JobState fields
- `update_step(job_id: str, step_index: int, **kwargs) -> None` -- partial update of a specific StepState
- `remove(job_id: str) -> None`

A module-level singleton `job_store = JobStore()` (same pattern as `config_manager` in [config.py](host/solar_host/config.py)).

---

## Step 4: Extend DockerService for models/data write toggle

Only `models/` and `data/` need to flip between writable (population) and read-only (consumption). There is **no** per-mount override API — a single `is_preparation_step` flag covers both directories. `output/` and `config/` are always RW; HF cache mount unchanged; logs remain host-only.

**Change in [solar_host/docker/service.py](host/solar_host/docker/service.py):**

- Add `is_preparation_step: bool = False` to `create_container`.
- **`is_preparation_step=True`:** bind-mount `models` and `data` as **read-write** (steps that download or populate those trees).
- **`is_preparation_step=False` (default):** bind-mount both as **read-only** (consume-only steps per S-021 Section 3.2).
- Replace the current hardcoded `_WORKSPACE_MOUNTS` matrix with logic driven by this flag.
- Update [tests/test_docker_service.py](host/tests/test_docker_service.py) for both modes.

---

## Step 5: Job Executor (`solar_host/jobs/executor.py`)

The core async orchestrator. Composes `DockerService`, `WorkspaceManager`, and `JobStore`.

**Class `JobExecutor`:**

- `__init__(docker_service: DockerService, store: JobStore, settings: Settings)`
- `async run_job(job_def: JobDefinition) -> JobState`: Main entry point.
  1. Validate job ID.
  2. Check disk space.
  3. Create workspace (via `create_workspace`).
  4. Register `JobState(status=running)` in store.
  5. Loop over steps sequentially.
  6. Return final `JobState`.
- `async cancel_job(job_id: str) -> None`: Set cancellation flag, stop active container.

**Per-step execution (`_run_step`):**

1. Check cancellation flag (`asyncio.Event`).
2. Pre-step disk check.
3. Build full environment dict: workspace paths (S-021 Section 4.1), credentials from Settings (Section 4.2), step-specific vars from `StepDefinition.environment` (Section 4.3), plus `STEP_NAME` and `STEP_INDEX`.
4. `await asyncio.to_thread(docker_service.create_container, ...)` -- pass step's image, env, gpu flag, and `is_preparation_step=step.is_preparation_step`.
5. `await asyncio.to_thread(docker_service.start_container, container_id)`.
6. Concurrently in background thread: stream logs via `docker_service.stream_logs(follow=True)` and write to `JOBS_DIR/<job-id>/logs/<step_name>.log`.
7. `await asyncio.to_thread(docker_service.wait_container, container_id)` -- blocks until container exits.
8. Catch `ContainerNonZeroExitError`: update step state to `failed`, set `error_message` to last stderr lines, trigger fail-fast (skip remaining steps).
9. On success: update step state to `completed`.
10. Always: `await asyncio.to_thread(docker_service.remove_container, container_id, force=True)`.

**Cancellation mechanism:**

- Per-job `asyncio.Event` (`_cancel_events: dict[str, asyncio.Event]`).
- `cancel_job` sets the event and calls `docker_service.stop_container` on the active container.
- Before each step, check if the event is set. If so, mark remaining steps as `cancelled`.

**Environment building (`_build_step_environment`):**

- Workspace paths: `JOB_ID`, `WORKSPACE_MODELS=/workspace/models`, `WORKSPACE_DATA=/workspace/data`, `WORKSPACE_OUTPUT=/workspace/output`, `WORKSPACE_CONFIG=/workspace/config`, `JOB_CONFIG=/workspace/config/job.json`, `STEP_NAME`, `STEP_INDEX`.
- Infrastructure credentials: `HARBOR_URL`, `HARBOR_USERNAME`, `HARBOR_PASSWORD`, `HF_TOKEN`, `HF_HOME=/workspace/.cache/huggingface` from Settings. (Note: `DATA_REPOSITORY_URL` and `WANDB_API_KEY` are not yet in Settings; pass through from step environment if present.)
- Step-specific vars: merge in `StepDefinition.environment` (these come from the job definition submitted by Solar Control).

---

## Step 6: Background Cleanup + Integration

**Cleanup loop (in `solar_host/jobs/executor.py` or `workspace.py`):**

- `async cleanup_loop(store: JobStore, poll_interval_s: float = 300)`: Runs as a background `asyncio.Task`. Periodically scans `JobStore` for jobs in terminal states (`completed`, `failed`, `cancelled`) whose `finished_at + retention_hours` has passed. Calls `delete_workspace` and removes from store.

**Integration in [solar_host/main.py](host/solar_host/main.py):**

- In `lifespan`: instantiate `DockerService`, `JobExecutor`, start `cleanup_loop` as a background task.
- Store references on `app.state` so routes (S-027) can access them later.
- Cancel the cleanup task on shutdown; stop any active jobs gracefully.

**Package init (`solar_host/jobs/__init__.py`):**

- Re-export: `JobDefinition`, `StepDefinition`, `JobStatus`, `StepStatus`, `JobState`, `StepState`, `JobExecutor`, `JobStore`, `job_store`.

---

## Step 7: Unit Tests

`**tests/test_job_models.py`:**

- Validate `JobDefinition` and `StepDefinition` construction and defaults.
- Test `JobStatus` / `StepStatus` enum values.
- Test `JobState` / `StepState` serialization.

`**tests/test_workspace.py`:**

- `validate_job_id`: valid IDs pass, IDs with `/`, `..`, special chars rejected.
- `check_disk_space`: mock `shutil.disk_usage`, test pass and fail cases.
- `create_workspace`: verify directory structure, permissions, `job.json` content, optional `training.json`.
- `delete_workspace`: verify removal, no error on missing directory.

`**tests/test_job_store.py`:**

- Add, get, update, remove operations.
- `get_all` returns all jobs.
- `update_step` modifies the correct step.
- Thread-safety: concurrent add/get from multiple threads.

`**tests/test_job_executor.py`:**

- Mock `DockerService` (same pattern as [tests/test_docker_service.py](host/tests/test_docker_service.py)).
- Successful 3-step job: all steps complete, final state is `completed`.
- Step failure mid-pipeline: first step passes, second fails (non-zero exit), third skipped, job state is `failed`.
- Cancellation: cancel during second step, active container stopped, remaining steps `cancelled`.
- Disk space failure before step: raises `InsufficientDiskError`, job fails.
- Container start failure: step marked `failed`, remaining steps skipped.
- Environment building: verify workspace paths, credentials, and step-specific vars are all present.
- Preparation vs consumption: mock asserts `create_container` receives `is_preparation_step=True` for a download-style step and `False` for a train-style step.

`**tests/test_docker_service.py` additions:**

- Test `create_container(..., is_preparation_step=True)`: both `models` and `data` volume entries use `rw`.
- Test `create_container(..., is_preparation_step=False)` (or omitted): both `models` and `data` use `ro`.
- Assert `output` / `config` remain `rw` in both modes.

---

## Design Decisions Summary

- **models/data write access**: Single boolean `is_preparation_step` on `StepDefinition`, forwarded to `DockerService.create_container`. `True` = RW on `models/` and `data/`; `False` = RO on both. No per-mount lists.
- **Callback hooks**: Not included in this step. The executor updates `JobStore` synchronously; S-025 and S-026 can observe state changes or be wired via callbacks when those issues are implemented. The architecture keeps the executor testable and focused.
- **Log capture**: Concurrent `stream_logs` in a background thread writing to the log file, alongside `wait_container` for exit code. Both run via `asyncio.to_thread`.
- **No REST routes**: Routes (`POST /jobs`, `GET /jobs/{id}`, `DELETE /jobs/{id}`) are S-027 scope. This implementation provides the executor and store that S-027 will consume.

