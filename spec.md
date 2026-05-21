# Description

Solar Host should publish structured lifecycle events as jobs and steps move through execution. These events let Solar Control aggregate state and let clients monitor progress without polling every host.

Events must describe Solar execution state only. SuperNova remains the orchestrator above Solar Control and does not communicate with Solar Hosts directly.

## Goal

- Emit lifecycle events for `job_started`, `step_started`, `step_completed`, `step_failed`, `job_completed`, `job_failed`, and `job_cancelled`.
- Include job ID, host ID, step ID/name, status, timestamps, duration, exit code where available, and concise error details.
- Update events from the same state transitions used by the S-023 executor so API status and emitted events agree.
- Ensure terminal events are emitted exactly once per job/step.
- Add tests or a local verification path for success, failure, and cancellation flows.

## Additional Notes

- Repo: `solar-host` at `/mnt/nvme/AI/solar/solar-host`
- Key files likely: job executor, Socket.IO event emitter, job state models.
- Depends on S-023 (step executor).
- Consumed by S-032 (Solar Control proxy/event forwarding), S-033 host status, SuperNova job monitoring, and Solar WebUI.

### S-021 Workspace Reference

The `/mnt/nvme/AI/damit-aiops/training-platform-project/docs/specs/job-step-workspace.md` specifies the event fields:

- **step_failed** event includes `exit_code` (int) and `error_summary` (last N lines of stderr). This lets consumers show why a step failed without reading the full log.
- **Event payload:**
  `{job_id, host_id, step_name, step_index, status, timestamp, duration_s, exit_code (optional), error_summary (optional)}`
- **Job-level events** (`job_completed`, `job_failed`) include the final workspace path and retention deadline so consumers know where artifacts live until cleanup.
- The executor transitions state; events are emitted from the same state transitions so API status (`GET /jobs/{id}`) and event payloads are always consistent.
