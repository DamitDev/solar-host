## Description

Solar Host needs a small Docker execution layer before it can run SuperNova job steps. This layer should wrap the Docker SDK for Python (`docker-py`) with lifecycle primitives that the step executor can compose without knowing low-level Docker API details.

Solar Host owns container execution. Solar Control routes requests and aggregates state; SuperNova never calls Solar Hosts directly.

## Goal

- Add `docker` / Docker SDK dependency and configuration for connecting to the local Docker daemon.
- Implement primitives to pull images, create containers, start containers, stop/cancel containers, remove containers, inspect status, and stream stdout/stderr logs.
- Support bind-mounting a host job workspace into the container at `/workspace`.
- Return structured errors for image pull failures, Docker daemon unavailability, container start failures, and non-zero exits.
- Add focused tests or fakes for lifecycle behavior where practical, without requiring real step images in unit tests.

## Additional Notes

- Repo: `solar-host` at `/home/csakyzsolt/Projects/DAMIT/solar/repositories/host`
- Key files likely: new Docker client/service module, `app/config.py`, dependency manifests, tests.
- Depends on no prior issue.
- Used by S-023 (sequential step executor), S-024 (GPU allocation), S-025 (log streaming), and S-027 (job API).

### S-021 Workspace Reference

The `/home/csakyzsolt/Projects/DAMIT/supernova/repositories/training-platform-project/docs/specs/job-step-workspace.md` defines the exact bind-mount contract this module must support:

- **Four mounts per container:** `JOBS_DIR/<job-id>/models` → `/workspace/models`, `data` → `/workspace/data`, `output` → `/workspace/output`, `config` → `/workspace/config`. Some mounts may be read-only per the step's role (Section 3.2).
- **Container user:** Run containers as `CONTAINER_UID:CONTAINER_GID` (default `1000:1000`, configurable in Solar Host settings). The workspace directories are created with this ownership.
- **No privileged mode.** No host network mode. No additional volume mounts beyond the workspace.
- **GPU access:** If the step requires GPU (train, convert), the Docker run config must include the appropriate `--gpus` or `device_requests`. This is consumed by S-024.
- **Environment injection:** All workspace paths and credentials are passed via `--env`; no files containing secrets are mounted.

### Required Solar Host Config Entries

When extending `app/config.py`, add these settings (all prefixed with `SOLAR_` or not, follow existing convention):

| Setting | Env Var | Default | Used by |
|---------|---------|---------|---------|
| Jobs root directory | `JOBS_DIR` | `./jobs` | S-023 workspace creation |
| Container runtime UID | `CONTAINER_UID` | `1000` | S-022 run container, S-023 workspace ownership |
| Container runtime GID | `CONTAINER_GID` | `1000` | S-022 run container, S-023 workspace ownership |
| Min free disk GB | `MIN_FREE_DISK_GB` | `2.0` | Already exists; S-023 uses it for pre-step checks |
| HF cache directory | `HF_CACHE_DIR` | `./hf-cache` | Mounted into containers as `HF_HOME` for shared model caching across jobs |

The `HF_CACHE_DIR` is a host-global cache (like `MODELS_DIR`) so HuggingFace models downloaded during training aren't re-fetched per job. This is NOT part of the per-job workspace — it's a persistent host-level cache mounted into containers at `HF_HOME`.
