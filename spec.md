## Description

Solar Host needs a REST API for Solar Control to submit, inspect, and cancel step execution jobs. This API is host-local execution control; callers above Solar Host must go through Solar Control.

The API should expose the S-023 executor without leaking Docker internals to Solar Control or SuperNova.

## Goal

- Implement `POST /jobs` to accept job config, ordered step list, workspace options, resource hints, and caller correlation metadata.
- Return a job ID immediately after successful validation and job start/enqueue.
- Implement `GET /jobs/{id}` to return job status, current step, per-step statuses, timestamps, exit codes, and recent log buffer references or snippets.
- Implement `DELETE /jobs/{id}` to cancel an active job and stop the running container.
- Require Solar Host API authentication consistently with existing management endpoints.
- Return structured 400/404/409/500-style errors for invalid step payloads, missing jobs, already-terminal jobs, and executor failures.

## Additional Notes

- Repo: `solar-host` at `/mnt/nvme/AI/solar/solar-host`
- Key files likely: new `app/routes/jobs.py`, job request/response schemas, executor service, auth wiring.
- Depends on S-023 (step executor).
- Consumed by S-032 (Solar Control job submission proxy).

### S-021 Workspace Reference

The `/mnt/nvme/AI/damit-aiops/training-platform-project/docs/specs/job-step-workspace.md` defines the job config fields that `POST /jobs` must accept:

- **Workspace options:** `job_id` (validated against path traversal), `retention_hours` (default 24), `min_free_disk_gb` (per-job override, default 2).
- **Step list:** Each step specifies `name`, `image`, `environment` (step-specific vars from Section 4.3), and optional `resource_hints`.
- **Base inputs:** `base_model_uri`, `training_data_uri` — used by the executor to derive `MODEL_URI`/`DATASET_URI` for download steps and `MODEL_DIR`/`DATASET_DIR` for the train step.
- **Training config:** Provided inline or as a path; Solar Host writes `/workspace/config/training.json` from this.
- **Model selection policy:** `model_selection` object (strategy, metric, direction) passed through to the train step.
- **Deployment target:** `deployment` block (target model name, replicas, strategy) for the upload step's registration.
- **`GET /jobs/{id}` response** must include workspace path (`JOBS_DIR/<job-id>/`), per-step status with exit codes, and log file references.
- **`DELETE /jobs/{id}`** triggers immediate workspace cleanup (bypass retention).
