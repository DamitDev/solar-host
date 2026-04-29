"""Configuration management for solar-host."""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic_settings import BaseSettings

from solar_host.models.base import Instance, InstanceStatus
from solar_host.models.llamacpp import LlamaCppConfig
from solar_host.models.huggingface import (
    HuggingFaceCausalConfig,
    HuggingFaceClassificationConfig,
    HuggingFaceEmbeddingConfig,
    HuggingFaceVisionConfig,
)

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings."""

    api_key: str = "change-me-please"
    host: str = "0.0.0.0"
    port: int = 8001
    config_file: str = "config.json"
    log_dir: str = "logs"
    start_port: int = 3500
    max_retries: int = 2
    log_buffer_size: int = 1000
    models_dir: str = "./models"
    min_free_disk_gb: float = 2.0

    # Solar-control connection settings (WebSocket 2.0)
    # URL(s) to solar-control's host channel endpoint
    # Single: "ws://localhost:8000/ws/host-channel"
    # Multiple: "ws://dev:8000/ws/host-channel,ws://prod:8000/ws/host-channel"
    solar_control_url: str = ""  # Supports comma-separated URLs for multi-control
    # Human-readable host name (optional, for display in webui)
    host_name: str = ""
    ws_reconnect_delay: float = 1.0  # Initial reconnect delay in seconds
    ws_reconnect_max_delay: float = 30.0  # Maximum reconnect delay
    ws_ping_interval: float = 25.0  # Ping interval in seconds

    # Allow connecting to solar-control over HTTPS/WSS with invalid certificates
    insecure: bool = False

    # Harbor registry credentials (required for repo:// pulls)
    harbor_url: str = ""
    harbor_username: str = ""
    harbor_password: str = ""

    # HuggingFace Hub token (optional for public models, required for gated models)
    hf_token: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()


def migrate_config_data(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate legacy config data to new format.

    Adds backend_type to configs that don't have it (defaults to llamacpp).
    """
    if "backend_type" not in config_data:
        # Check if this looks like a llama.cpp config (has 'model' field for GGUF path)
        if "model" in config_data:
            config_data["backend_type"] = "llamacpp"
        # Check if this looks like a HuggingFace config (has 'model_id' field)
        elif "model_id" in config_data:
            # Default to causal, but could be classification if labels present
            if "labels" in config_data:
                config_data["backend_type"] = "huggingface_classification"
            else:
                config_data["backend_type"] = "huggingface_causal"
        else:
            # Default fallback to llamacpp
            config_data["backend_type"] = "llamacpp"

    return config_data


def parse_instance_config(config_data: Dict[str, Any]) -> Any:
    """Parse config data into the appropriate config type based on backend_type."""
    # Migrate first
    config_data = migrate_config_data(config_data)

    backend_type = config_data.get("backend_type", "llamacpp")

    if backend_type == "llamacpp":
        return LlamaCppConfig(**config_data)
    elif backend_type == "huggingface_causal":
        return HuggingFaceCausalConfig(**config_data)
    elif backend_type == "huggingface_classification":
        return HuggingFaceClassificationConfig(**config_data)
    elif backend_type == "huggingface_embedding":
        return HuggingFaceEmbeddingConfig(**config_data)
    elif backend_type == "huggingface_vision":
        return HuggingFaceVisionConfig(**config_data)
    else:
        raise ValueError(f"Unknown backend_type: {backend_type!r}")


class ConfigManager:
    """Manages instance configurations and persistence.

    All public methods are protected by a threading lock so the instance
    dict and on-disk config.json stay consistent across the asyncio event
    loop thread and background log-reader threads.
    """

    def __init__(self, config_file: Optional[str] = None):
        self._lock = threading.Lock()
        self.config_file = Path(config_file or settings.config_file)
        self.instances: Dict[str, Instance] = {}
        self.roles: List[str] = ["inference"]
        self.load()

    def load(self):
        """Load configuration from disk with backward compatibility."""
        with self._lock:
            self._load_unlocked()

    def _load_unlocked(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.roles = data.get("roles", ["inference"])
                    self.instances = {}
                    for instance_data in data.get("instances", []):
                        try:
                            if "config" in instance_data:
                                instance_data["config"] = migrate_config_data(
                                    instance_data["config"]
                                )
                            config = parse_instance_config(
                                instance_data.get("config", {})
                            )
                            instance_data_copy = instance_data.copy()
                            instance_data_copy["config"] = config
                            instance = Instance(**instance_data_copy)
                            self.instances[instance.id] = instance
                        except Exception as e:
                            logger.warning("Skipping instance during load: %s", e)
                            continue
            except Exception as e:
                logger.error("Error loading config: %s", e)
                self.instances = {}
        else:
            self.instances = {}

    def _save_unlocked(self):
        """Write config to disk atomically (tmp + os.replace)."""
        try:
            data = {
                "roles": self.roles,
                "instances": [
                    instance.model_dump(mode="json")
                    for instance in self.instances.values()
                ],
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.config_file.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, self.config_file)
        except Exception as e:
            logger.error("Error saving config: %s", e)

    def save(self):
        """Save configuration to disk (thread-safe)."""
        with self._lock:
            self._save_unlocked()

    def add_instance(self, instance: Instance):
        """Add a new instance."""
        with self._lock:
            self.instances[instance.id] = instance
            self._save_unlocked()

    def update_instance(self, instance_id: str, instance: Instance):
        """Update an existing instance."""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id] = instance
                self._save_unlocked()

    def update_instance_runtime(self, instance_id: str, **kwargs):
        """Update ephemeral runtime fields for an instance (no disk write)."""
        with self._lock:
            instance = self.instances.get(instance_id)
            if not instance:
                return
            if "busy" in kwargs:
                instance.busy = bool(kwargs["busy"])
            if "prefill_progress" in kwargs:
                instance.prefill_progress = kwargs["prefill_progress"]
            if "active_slots" in kwargs:
                instance.active_slots = int(kwargs["active_slots"])

    def remove_instance(self, instance_id: str):
        """Remove an instance."""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
                self._save_unlocked()

    def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get an instance by ID."""
        with self._lock:
            return self.instances.get(instance_id)

    def get_all_instances(self) -> List[Instance]:
        """Get all instances."""
        with self._lock:
            return list(self.instances.values())

    def get_running_instances(self) -> List[Instance]:
        """Get all running instances."""
        with self._lock:
            return [
                instance
                for instance in self.instances.values()
                if instance.status == InstanceStatus.RUNNING
            ]


# Global config manager instance
config_manager = ConfigManager()
