"""Unit tests for solar_host.jobs.step_log_buffer.StepLogBuffer."""

from __future__ import annotations

import asyncio
import queue
from unittest.mock import AsyncMock, patch

import pytest

from solar_host.jobs.step_log_buffer import MAX_QUEUE_SIZE, StepLogBuffer

_JOB_ID = "job-abc"
_STEP = "train"
_IDX = 0


def _make_buffer(maxlen: int = 1000) -> StepLogBuffer:
    return StepLogBuffer(maxlen=maxlen)


# ---------------------------------------------------------------------------
# append — seq increment
# ---------------------------------------------------------------------------


def test_append_increments_seq_per_step() -> None:
    buf = _make_buffer()
    buf.append(_JOB_ID, _STEP, _IDX, "stdout", "line1")
    buf.append(_JOB_ID, _STEP, _IDX, "stdout", "line2")
    buf.append(_JOB_ID, _STEP, _IDX, "stderr", "err")

    msgs = buf.get_buffer(_JOB_ID, _STEP)
    assert [m.seq for m in msgs] == [0, 1, 2]


def test_seq_is_independent_per_step() -> None:
    buf = _make_buffer()
    buf.append(_JOB_ID, "step-a", 0, "stdout", "a")
    buf.append(_JOB_ID, "step-b", 1, "stdout", "b")
    buf.append(_JOB_ID, "step-a", 0, "stdout", "c")

    msgs_a = buf.get_buffer(_JOB_ID, "step-a")
    msgs_b = buf.get_buffer(_JOB_ID, "step-b")
    assert [m.seq for m in msgs_a] == [0, 1]
    assert [m.seq for m in msgs_b] == [0]


# ---------------------------------------------------------------------------
# append — buffer maxlen
# ---------------------------------------------------------------------------


def test_buffer_respects_maxlen() -> None:
    buf = _make_buffer(maxlen=3)
    for i in range(10):
        buf.append(_JOB_ID, _STEP, _IDX, "stdout", f"line {i}")

    msgs = buf.get_buffer(_JOB_ID, _STEP)
    assert len(msgs) == 3
    # Oldest entries were evicted — only the last 3 remain.
    assert msgs[-1].line == "line 9"


# ---------------------------------------------------------------------------
# mark_completed
# ---------------------------------------------------------------------------


def test_mark_completed_enqueues_completion_entry() -> None:
    buf = _make_buffer()
    buf.append(_JOB_ID, _STEP, _IDX, "stdout", "some output")
    buf.mark_completed(_JOB_ID, _STEP, _IDX, exit_code=0)

    msgs = buf.get_buffer(_JOB_ID, _STEP)
    completion = msgs[-1]
    assert completion.completed is True
    assert completion.exit_code == 0
    assert completion.line == ""


def test_mark_completed_nonzero_exit_code() -> None:
    buf = _make_buffer()
    buf.mark_completed(_JOB_ID, _STEP, _IDX, exit_code=2)

    # Even without prior append the buffer key may be absent; completion
    # still goes to the emit queue.
    entry: dict = buf._queue.get_nowait()
    assert entry["completed"] is True
    assert entry["exit_code"] == 2


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


def test_remove_clears_all_step_keys_for_job() -> None:
    buf = _make_buffer()
    buf.append(_JOB_ID, "step-a", 0, "stdout", "a")
    buf.append(_JOB_ID, "step-b", 1, "stdout", "b")
    buf.append("other-job", "step-x", 0, "stdout", "x")

    buf.remove(_JOB_ID)

    assert buf.get_buffer(_JOB_ID, "step-a") == []
    assert buf.get_buffer(_JOB_ID, "step-b") == []
    # Unrelated job is intact.
    assert len(buf.get_buffer("other-job", "step-x")) == 1


def test_remove_idempotent_for_unknown_job() -> None:
    buf = _make_buffer()
    buf.remove("no-such-job")  # must not raise


# ---------------------------------------------------------------------------
# Queue overflow — silent drop
# ---------------------------------------------------------------------------


def test_queue_overflow_drops_silently() -> None:
    buf = _make_buffer()
    # Fill the queue to capacity.
    for _ in range(MAX_QUEUE_SIZE):
        buf._queue.put_nowait({"dummy": True})

    # Appending now should not raise; the entry is silently dropped.
    buf.append(_JOB_ID, _STEP, _IDX, "stdout", "overflow line")


# ---------------------------------------------------------------------------
# flush — drains queue and calls broadcast
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_flush_drains_queue_and_broadcasts() -> None:
    buf = _make_buffer()
    buf.append(_JOB_ID, _STEP, _IDX, "stdout", "hello")
    buf.append(_JOB_ID, _STEP, _IDX, "stderr", "world")

    with patch(
        "solar_host.jobs.step_log_buffer.broadcast_step_log_batch",
        new_callable=AsyncMock,
    ) as mock_broadcast:
        await buf.flush()

    mock_broadcast.assert_called_once()
    entries = mock_broadcast.call_args[0][0]
    assert len(entries) == 2
    assert entries[0]["line"] == "hello"
    assert entries[1]["line"] == "world"

    # Queue is now empty.
    assert buf._queue.empty()


@pytest.mark.anyio
async def test_flush_no_op_when_queue_empty() -> None:
    buf = _make_buffer()
    with patch(
        "solar_host.jobs.step_log_buffer.broadcast_step_log_batch",
        new_callable=AsyncMock,
    ) as mock_broadcast:
        await buf.flush()

    mock_broadcast.assert_not_called()
