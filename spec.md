## Description

Training-capable Solar Hosts with NVIDIA GPUs need to run step containers with explicit GPU access. Solar Host should translate resource hints into Docker GPU settings using NVIDIA Container Toolkit, while keeping GPU execution local to the host.

This issue integrates with S-023 so the executor can validate and apply GPU options before each step starts.

## Goal

- Detect whether the host supports NVIDIA Container Toolkit and return a clear error when GPU execution is requested but unavailable.
- Extend step execution options with GPU device selection and GPU count, using Docker SDK equivalents for `--gpus`.
- Pass through environment needed by NVIDIA containers when required, without hardcoding Etalon-specific behavior.
- Validate requested GPUs against Solar Host resource data where available.
- Exercise the integration through S-023 step execution with at least one GPU-enabled test or documented manual verification command.

## Additional Notes

- Repo: `solar-host` at `/mnt/nvme/AI/solar/solar-host`
- Key files likely: Docker lifecycle module from S-022, executor module from S-023, host resource/config modules.
- Depends on S-022 formally, and should integrate with S-023 for validation.
- Downstream consumers: SuperNova training jobs, S-032 host routing, and S-033 active workload reporting.
