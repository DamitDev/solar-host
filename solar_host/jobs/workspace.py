"""Workspace filesystem operations for job execution (S-021)."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from solar_host.jobs.errors import InsufficientDiskError

if TYPE_CHECKING:
    from solar_host.config import Settings
    from solar_host.jobs.models import JobDefinition

logger = logging.getLogger(__name__)

# Characters allowed in job IDs: alphanumerics, hyphens, underscores, dots.
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

_WORKSPACE_SUBDIRS = ("models", "data", "output", "config", "logs")


def validate_job_id(job_id: str) -> None:
    """Reject job IDs containing unsafe characters or path-traversal sequences.

    Raises:
        ValueError: when the ID contains ``/``, ``..``, or non-filesystem-safe chars.
    """
    if not job_id:
        raise ValueError("job_id must not be empty")
    if "/" in job_id:
        raise ValueError(f"job_id must not contain '/': {job_id!r}")
    if ".." in job_id:
        raise ValueError(f"job_id must not contain '..': {job_id!r}")
    if not _JOB_ID_PATTERN.match(job_id):
        raise ValueError(
            f"job_id contains invalid characters (only A-Z, a-z, 0-9, '.', '-', '_'"
            f" allowed): {job_id!r}"
        )


def check_disk_space(jobs_dir: Path, min_free_gb: float) -> None:
    """Raise InsufficientDiskError when free space on *jobs_dir* is below threshold.

    Args:
        jobs_dir: Path on the filesystem partition to measure.
        min_free_gb: Required free space in gibibytes.
    """
    usage = shutil.disk_usage(jobs_dir)
    available_gb = usage.free / (1024**3)
    if available_gb < min_free_gb:
        raise InsufficientDiskError(required_gb=min_free_gb, available_gb=available_gb)


def build_job_json(job_def: JobDefinition) -> dict:  # type: ignore[type-arg]
    """Build the job.json payload written to *config/* (S-021 Section 5.2)."""
    return {
        "job_id": job_def.job_id,
        "name": job_def.name,
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline": [step.name for step in job_def.steps],
        "base_model_uri": job_def.base_model_uri,
        "training_data_uri": job_def.training_data_uri,
        "model_selection": job_def.model_selection,
        "deployment": job_def.deployment,
        "retention_hours": job_def.retention_hours,
        "steps": {},
    }


def create_workspace(job_def: JobDefinition, settings: Settings) -> Path:
    """Create the job workspace directory tree and write config files.

    Creates ``JOBS_DIR/<job-id>/{models,data,output,config,logs}`` with mode
    0o755, owned by ``CONTAINER_UID:CONTAINER_GID``.  Writes ``job.json`` into
    ``config/`` and, if ``job_def.training_config`` is provided, also writes
    ``training.json``.

    Returns:
        The workspace root path (``JOBS_DIR/<job-id>``).
    """
    jobs_dir = Path(settings.jobs_dir)
    workspace = jobs_dir / job_def.job_id
    workspace.mkdir(parents=True, exist_ok=True)

    for subdir in _WORKSPACE_SUBDIRS:
        subpath = workspace / subdir
        subpath.mkdir(mode=0o755, exist_ok=True)
        try:
            os.chown(subpath, settings.container_uid, settings.container_gid)
        except PermissionError:
            logger.warning(
                "Could not chown %s to %d:%d (permission denied; continuing)",
                subpath,
                settings.container_uid,
                settings.container_gid,
            )

    # chown workspace root as well
    try:
        os.chown(workspace, settings.container_uid, settings.container_gid)
    except PermissionError:
        logger.warning(
            "Could not chown workspace root %s (permission denied; continuing)",
            workspace,
        )

    config_dir = workspace / "config"

    job_json_path = config_dir / "job.json"
    job_json_path.write_text(json.dumps(build_job_json(job_def), indent=2))

    if job_def.training_config is not None:
        training_json_path = config_dir / "training.json"
        training_json_path.write_text(json.dumps(job_def.training_config, indent=2))

    return workspace


def delete_workspace(workspace_path: Path) -> None:
    """Remove the workspace directory tree.

    Logs errors but does not raise if the directory is missing or removal fails.
    """
    try:
        shutil.rmtree(workspace_path)
    except FileNotFoundError:
        logger.debug("Workspace %s already absent — nothing to delete", workspace_path)
    except Exception:
        logger.exception("Failed to delete workspace %s", workspace_path)
