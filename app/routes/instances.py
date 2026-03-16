from fastapi import APIRouter, HTTPException
from typing import List, Optional

from app.models import (
    Instance,
    InstanceCreate,
    InstanceUpdate,
    InstanceResponse,
    InstanceStatus,
    InstanceRuntimeState,
    GenerationMetrics,
    LogMessage,
)
from app.config import config_manager, parse_instance_config
from app.process_manager import process_manager

router = APIRouter(prefix="/instances", tags=["instances"])


@router.post("", response_model=InstanceResponse)
async def create_instance(data: InstanceCreate):
    """Create a new model instance (llama.cpp or HuggingFace)"""
    try:
        instance = process_manager.create_instance(data.config)
        return InstanceResponse(
            instance=instance, message=f"Instance {instance.id} created successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=List[Instance])
async def list_instances():
    """List all instances"""
    return config_manager.get_all_instances()


@router.get("/{instance_id}", response_model=Instance)
async def get_instance(instance_id: str):
    """Get instance details"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    return instance


@router.put("/{instance_id}", response_model=InstanceResponse)
async def update_instance(instance_id: str, data: InstanceUpdate):
    """Update instance configuration (only when stopped)"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    if instance.status not in (InstanceStatus.STOPPED, InstanceStatus.FAILED):
        raise HTTPException(
            status_code=400, detail="Cannot update running instance. Stop it first."
        )

    try:
        # Parse config if it's a dict (from FastAPI request body)
        config = data.config
        if isinstance(config, dict):
            config = parse_instance_config(config)

        instance.config = config
        config_manager.update_instance(instance_id, instance)
        return InstanceResponse(
            instance=instance, message=f"Instance {instance_id} updated successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{instance_id}", response_model=InstanceResponse)
async def delete_instance(instance_id: str):
    """Delete an instance (must be stopped first)"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    if instance.status not in (InstanceStatus.STOPPED, InstanceStatus.FAILED):
        raise HTTPException(
            status_code=400, detail="Cannot delete running instance. Stop it first."
        )

    try:
        # Use process_manager to delete (notifies solar-control)
        process_manager.delete_instance(instance_id)
        return InstanceResponse(
            instance=instance, message=f"Instance {instance_id} deleted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{instance_id}/start", response_model=InstanceResponse)
async def start_instance(instance_id: str):
    """Start an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    success = await process_manager.start_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after start")

    if success:
        return InstanceResponse(
            instance=instance, message=f"Instance {instance_id} started successfully"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start instance: {instance.error_message}",
        )


@router.post("/{instance_id}/stop", response_model=InstanceResponse)
async def stop_instance(instance_id: str):
    """Stop an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    success = await process_manager.stop_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after stop")

    if success:
        return InstanceResponse(
            instance=instance, message=f"Instance {instance_id} stopped successfully"
        )
    else:
        raise HTTPException(
            status_code=500, detail=f"Failed to stop instance: {instance.error_message}"
        )


@router.post("/{instance_id}/restart", response_model=InstanceResponse)
async def restart_instance(instance_id: str):
    """Restart an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    success = await process_manager.restart_instance(instance_id)
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found after restart")

    if success:
        return InstanceResponse(
            instance=instance, message=f"Instance {instance_id} restarted successfully"
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to restart instance: {instance.error_message}",
        )


@router.get("/{instance_id}/state", response_model=InstanceRuntimeState)
async def get_instance_state(instance_id: str):
    """Get ephemeral runtime state for an instance"""
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    # Build current snapshot (ephemeral values default to safe values)
    now_iso = (
        __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat()
    )
    return InstanceRuntimeState(
        instance_id=instance_id,
        busy=getattr(instance, "busy", False),
        prefill_progress=getattr(instance, "prefill_progress", None),
        active_slots=getattr(instance, "active_slots", 0),
        timestamp=now_iso,
    )


@router.get("/{instance_id}/logs", response_model=List[LogMessage])
async def get_instance_logs(instance_id: str):
    """Get buffered logs for an instance.

    Returns the in-memory log buffer (last N log lines).
    """
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    logs = process_manager.get_log_buffer(instance_id)
    return logs


@router.get("/{instance_id}/last-generation", response_model=GenerationMetrics)
async def get_last_generation(
    instance_id: str, after: Optional[str] = None, within_s: Optional[int] = None
):
    """Return most recent finished generation metrics for the instance.

    Optional filters:
    - after: ISO8601 timestamp; only return if finished_at >= after
    - within_s: only return if finished within the last N seconds
    """
    # Validate instance
    instance = config_manager.get_instance(instance_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")

    metrics = process_manager.get_last_generation(instance_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="No generation metrics available")

    # Apply filters
    try:
        from datetime import datetime, timezone

        def parse_iso(ts: str | None):
            if not ts:
                return None
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(
                    timezone.utc
                )
            except Exception:
                return None

        finished_dt = parse_iso(metrics.finished_at)
        if after:
            after_dt = parse_iso(after)
            if after_dt and finished_dt and finished_dt < after_dt:
                raise HTTPException(
                    status_code=404,
                    detail="No generation metrics after the specified timestamp",
                )
        if within_s is not None and within_s >= 0:
            now_dt = datetime.now(timezone.utc)
            if finished_dt and (now_dt - finished_dt).total_seconds() > float(within_s):
                raise HTTPException(
                    status_code=404,
                    detail="No recent generation metrics within the specified window",
                )
    except HTTPException:
        raise
    except Exception:
        # On any parsing error, return the metrics without filtering
        pass

    return metrics
