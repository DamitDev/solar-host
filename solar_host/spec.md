## Description

Solar Host needs an executor that runs a job as an ordered list of Docker step containers. Each step mounts the same ephemeral job workspace, runs to completion, and passes files to later steps through that workspace.

Etalon and all other step containers are black boxes. The executor should only manage image execution, workspace mounts, environment/config injection, exit status, and cancellation.

## Goal

- Implement a job runner that accepts an ordered step list with image, name, command/args, environment, resource hints, and workspace mount settings.
- Create and mount one shared job workspace for every step, using the S-021 layout.
- Execute steps sequentially and fail fast when a step exits non-zero or cannot start.
- Track job and per-step state: pending, running, completed, failed, cancelled, timestamps, duration, exit code, and error message.
- Persist enough in-memory or local state for `GET /jobs/{id}` in S-027 to report active and recent jobs.
- Provide cancellation handling that stops the active container and marks remaining steps cancelled.

## Additional Notes

- Repo: `solar-host` at `/home/csakyzsolt/Projects/DAMIT/solar/repositories/host`
- Key files likely: new job executor/service module, job state models, workspace manager, tests.
- Depends on S-022 (Docker lifecycle primitives) and S-021 (workspace layout).
- Downstream consumers: S-024 GPU validation, S-025 logs, S-026 lifecycle events, S-027 REST API, and S-032 Solar Control proxy.

### S-021 Workspace Reference

The `/home/csakyzsolt/Projects/DAMIT/supernova/repositories/training-platform-project/docs/specs/job-step-workspace.md` defines the contract this executor implements:

- **Workspace creation:** Before the first step, create `JOBS_DIR/<job-id>/{models,data,output,config,logs}` on the host with `0755` permissions owned by `CONTAINER_UID:CONTAINER_GID` (default `1000:1000`). Write `job.json` into `config/`. If the job definition includes an inline training config, also write `training.json` into `config/` at this point.
- **Pre-step disk check:** Verify free space on `JOBS_DIR` partition ≥ `min_free_disk_gb` (default 2 GB) before each step.
- **Mounts per step:** Bind-mount the four canonical dirs plus the HF cache into every container — `models/` (RW for download steps, RO for train), `data/` (same pattern), `output/` (always RW), `config/` (always RW), and `HF_CACHE_DIR` → `HF_HOME` (host-global, shared across jobs). Log capture is host-side only (`stdout/stderr → logs/<step_name>.log`).
- **Environment injection:** Pass all env vars from the workspace spec (Sections 4.1–4.3) into each container. Derive step-specific vars (`MODEL_URI`, `DATASET_URI`, `TRAINING_CONFIG`, etc.) from the job definition received from Solar Control.
- **Inter-step state:** The executor writes `job.json` initially; step containers update `steps.<step_name>` in it. The executor should NOT modify `job.json` after creation.
- **Cleanup:** After terminal state, preserve workspace for `retention_hours` (default 24h), then `rm -rf`. Use a background cleanup loop (e.g. `asyncio` task or thread that wakes every N minutes, scans for expired workspaces, deletes them). Immediate cleanup on `DELETE /jobs/{id}`.
- **Fail-fast:** If a step exits non-zero, skip remaining steps and preserve workspace for debugging.
