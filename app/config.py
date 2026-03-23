"""Configuration management for solar-host."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic_settings import BaseSettings

from app.models.base import Instance, InstanceStatus
from app.models.llamacpp import LlamaCppConfig
from app.models.huggingface import (
    HuggingFaceCausalConfig,
    HuggingFaceClassificationConfig,
    HuggingFaceEmbeddingConfig,
)


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
    else:
        # Fallback to llamacpp
        return LlamaCppConfig(**config_data)


class ConfigManager:
    """Manages instance configurations and persistence."""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = Path(config_file or settings.config_file)
        self.instances: Dict[str, Instance] = {}
        self.roles: List[str] = ["inference"]
        self.load()

    def load(self):
        """Load configuration from disk with backward compatibility."""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    data = json.load(f)
                    self.roles = data.get("roles", ["inference"])
                    for instance_data in data.get("instances", []):
                        try:
                            # Migrate config if needed
                            if "config" in instance_data:
                                instance_data["config"] = migrate_config_data(
                                    instance_data["config"]
                                )

                            # Parse config into appropriate type
                            config = parse_instance_config(
                                instance_data.get("config", {})
                            )

                            # Create instance with parsed config
                            instance_data_copy = instance_data.copy()
                            instance_data_copy["config"] = config

                            instance = Instance(**instance_data_copy)
                            self.instances[instance.id] = instance
                        except Exception as e:
                            print(f"Error loading instance: {e}")
                            continue
            except Exception as e:
                print(f"Error loading config: {e}")
                self.instances = {}
        else:
            self.instances = {}

    def save(self):
        """Save configuration to disk."""
        try:
            data = {
                "roles": self.roles,
                "instances": [
                    instance.model_dump(mode="json")
                    for instance in self.instances.values()
                ],
            }
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving config: {e}")

    def add_instance(self, instance: Instance):
        """Add a new instance."""
        self.instances[instance.id] = instance
        self.save()

    def update_instance(self, instance_id: str, instance: Instance):
        """Update an existing instance."""
        if instance_id in self.instances:
            self.instances[instance_id] = instance
            self.save()

    def update_instance_runtime(self, instance_id: str, **kwargs):
        """Update ephemeral runtime fields for an instance (no disk write)."""
        instance = self.instances.get(instance_id)
        if not instance:
            return
        # Only mutate known runtime fields
        if "busy" in kwargs:
            instance.busy = bool(kwargs["busy"])
        if "prefill_progress" in kwargs:
            instance.prefill_progress = kwargs["prefill_progress"]
        if "active_slots" in kwargs:
            instance.active_slots = int(kwargs["active_slots"])

    def remove_instance(self, instance_id: str):
        """Remove an instance."""
        if instance_id in self.instances:
            del self.instances[instance_id]
            self.save()

    def get_instance(self, instance_id: str) -> Optional[Instance]:
        """Get an instance by ID."""
        return self.instances.get(instance_id)

    def get_all_instances(self) -> List[Instance]:
        """Get all instances."""
        return list(self.instances.values())

    def get_running_instances(self) -> List[Instance]:
        """Get all running instances."""
        return [
            instance
            for instance in self.instances.values()
            if instance.status == InstanceStatus.RUNNING
        ]


# Global config manager instance
config_manager = ConfigManager()
