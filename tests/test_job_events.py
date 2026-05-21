"""Unit tests for solar_host.jobs.events lifecycle event builders and emitters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from solar_host.jobs.events import (
    build_job_cancelled,
    build_job_completed,
    build_job_failed,
    build_job_started,
    build_step_completed,
    build_step_failed,
    build_step_started,
    emit_job_cancelled,
    emit_job_completed,
    emit_job_failed,
    emit_job_started,
    emit_step_completed,
    emit_step_failed,
    emit_step_started,
)

_NOW = datetime(2026, 5, 21, 10, 0, 0, tzinfo=UTC)
_JOB_ID = "job-abc123"

_BROADCAST = "solar_host.jobs.events.broadcast_job_lifecycle"
_GET_CLIENT = "solar_host.jobs.events.get_client"


# ---------------------------------------------------------------------------
# Payload builder shape tests
# ---------------------------------------------------------------------------


def test_build_job_started_shape() -> None:
    payload = build_job_started(_JOB_ID, "My Job", _NOW)
    assert payload["job_id"] == _JOB_ID
    assert payload["name"] == "My Job"
    assert payload["status"] == "running"
    assert payload["timestamp"] == _NOW.isoformat()
    assert "host_id" in payload


def test_build_job_completed_shape() -> None:
    payload = build_job_completed(_JOB_ID, _NOW, "/workspace/job-abc123", 24.0)
    assert payload["job_id"] == _JOB_ID
    assert payload["status"] == "completed"
    assert payload["workspace_path"] == "/workspace/job-abc123"
    expected_deadline = (_NOW + timedelta(hours=24.0)).isoformat()
    assert payload["retention_deadline"] == expected_deadline


def test_build_job_failed_shape() -> None:
    payload = build_job_failed(_JOB_ID, _NOW, "Something went wrong")
    assert payload["job_id"] == _JOB_ID
    assert payload["status"] == "failed"
    assert payload["error_message"] == "Something went wrong"


def test_build_job_failed_none_error() -> None:
    payload = build_job_failed(_JOB_ID, _NOW, None)
    assert payload["error_message"] is None


def test_build_job_cancelled_shape() -> None:
    payload = build_job_cancelled(_JOB_ID, _NOW)
    assert payload["job_id"] == _JOB_ID
    assert payload["status"] == "cancelled"
    assert payload["timestamp"] == _NOW.isoformat()


def test_build_step_started_shape() -> None:
    payload = build_step_started(_JOB_ID, "train", 2, _NOW)
    assert payload["job_id"] == _JOB_ID
    assert payload["step_name"] == "train"
    assert payload["step_index"] == 2
    assert payload["status"] == "running"


def test_build_step_completed_shape() -> None:
    payload = build_step_completed(_JOB_ID, "train", 2, _NOW, 12.5, 0)
    assert payload["status"] == "completed"
    assert payload["duration_s"] == 12.5
    assert payload["exit_code"] == 0


def test_build_step_failed_shape() -> None:
    payload = build_step_failed(_JOB_ID, "train", 2, _NOW, 3.0, 1, "Out of memory")
    assert payload["status"] == "failed"
    assert payload["exit_code"] == 1
    assert payload["error_summary"] == "Out of memory"
    assert payload["duration_s"] == 3.0


def test_build_step_failed_no_exit_code() -> None:
    payload = build_step_failed(_JOB_ID, "train", 2, _NOW, 1.0, None, "start error")
    assert payload["exit_code"] is None


# ---------------------------------------------------------------------------
# host_id is None when no client connected
# ---------------------------------------------------------------------------


def test_host_id_none_when_no_client() -> None:
    with patch(_GET_CLIENT, return_value=None):
        payload = build_job_started(_JOB_ID, "Job", _NOW)
    assert payload["host_id"] is None


def test_host_id_from_client_when_connected() -> None:
    mock_client = type("Client", (), {"host_id": "host-42"})()
    with patch(_GET_CLIENT, return_value=mock_client):
        payload = build_job_started(_JOB_ID, "Job", _NOW)
    assert payload["host_id"] == "host-42"


# ---------------------------------------------------------------------------
# emit_* call broadcast_job_lifecycle with correct event name
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_emit_job_started_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_job_started(_JOB_ID, "My Job", _NOW)
    mock_bc.assert_awaited_once()
    event_name, payload = mock_bc.call_args.args
    assert event_name == "job_started"
    assert payload["job_id"] == _JOB_ID
    assert payload["name"] == "My Job"


@pytest.mark.anyio
async def test_emit_job_completed_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_job_completed(_JOB_ID, _NOW, "/workspace/abc", 24.0)
    event_name, payload = mock_bc.call_args.args
    assert event_name == "job_completed"
    assert payload["workspace_path"] == "/workspace/abc"


@pytest.mark.anyio
async def test_emit_job_failed_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_job_failed(_JOB_ID, _NOW, "err")
    event_name, _ = mock_bc.call_args.args
    assert event_name == "job_failed"


@pytest.mark.anyio
async def test_emit_job_cancelled_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_job_cancelled(_JOB_ID, _NOW)
    event_name, _ = mock_bc.call_args.args
    assert event_name == "job_cancelled"


@pytest.mark.anyio
async def test_emit_step_started_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_step_started(_JOB_ID, "train", 0, _NOW)
    event_name, payload = mock_bc.call_args.args
    assert event_name == "step_started"
    assert payload["step_name"] == "train"
    assert payload["step_index"] == 0


@pytest.mark.anyio
async def test_emit_step_completed_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_step_completed(_JOB_ID, "train", 0, _NOW, 5.0, 0)
    event_name, payload = mock_bc.call_args.args
    assert event_name == "step_completed"
    assert payload["exit_code"] == 0
    assert payload["duration_s"] == 5.0


@pytest.mark.anyio
async def test_emit_step_failed_calls_broadcast() -> None:
    with patch(_BROADCAST, new_callable=AsyncMock) as mock_bc:
        await emit_step_failed(_JOB_ID, "train", 0, _NOW, 2.0, 1, "OOM")
    event_name, payload = mock_bc.call_args.args
    assert event_name == "step_failed"
    assert payload["error_summary"] == "OOM"
    assert payload["exit_code"] == 1
