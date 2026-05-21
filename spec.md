## Description

Solar Host needs to capture stdout/stderr from running step containers and stream it through the existing Solar event path. Logs should flow from Solar Host to Solar Control, then onward to connected clients such as Solar WebUI and SuperNova observers.

Solar Control routes and broadcasts log events; SuperNova should not connect directly to Solar Hosts.

## Goal

- Capture container stdout and stderr for each running step with job ID, step ID/name, stream, sequence number, and timestamp.
- Emit Socket.IO `step_log` events from Solar Host to Solar Control.
- Maintain bounded per-step log buffers so recent logs are available for status/debug endpoints without unbounded memory growth.
- Preserve ordering as much as practical and mark stream completion when a step exits.
- Handle client disconnects or Solar Control reconnects without crashing the running job.

## Additional Notes

- Repo: `solar-host` at `/mnt/nvme/AI/solar/solar-host`
- Key files likely: Docker log streaming wrapper, Socket.IO client/emitter, job state/log buffer modules.
- Depends on S-023 (step executor).
- Consumed by S-032 event forwarding, SuperNova job monitoring, and Solar WebUI operations views.

### S-021 Workspace Reference

The `/mnt/nvme/AI/damit-aiops/training-platform-project/docs/specs/job-step-workspace.md` specifies log placement:

- **Host-side log directory:** `JOBS_DIR/<job-id>/logs/`. Each step's combined stdout/stderr is written to `logs/<step_name>.log`.
- **Not mounted into containers:** `/workspace/logs/` is intentionally not a container mount point. Containers write to stdout/stderr; the executor captures and writes to the host-side directory.
- **Retention:** Logs are preserved alongside the workspace for `retention_hours` (default 24h), then deleted with the workspace.
- **Streaming path:** Host captures container stdout/stderr → emits `step_log` Socket.IO events → Solar Control broadcasts to clients. The host-side log files are the durable copy; Socket.IO events are the real-time stream.
