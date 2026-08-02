"""Tests for S-036: instance priority, managed_by, and intent_id fields."""

import pytest
from pydantic import ValidationError

from solar_host.models.base import (
    InstancePriority,
    Instance,
    InstanceCreate,
    InstanceUpdate,
)


class TestInstancePriorityEnum:
    """Tests for the InstancePriority enum."""

    def test_three_values(self):
        assert len(InstancePriority) == 3

    def test_production(self):
        assert InstancePriority.PRODUCTION.value == "production"

    def test_staging(self):
        assert InstancePriority.STAGING.value == "staging"

    def test_ephemeral(self):
        assert InstancePriority.EPHEMERAL.value == "ephemeral"


class TestInstancePriorityDefault:
    """Default priority for new and migrated instances."""

    def test_default_is_production(self):
        instance = Instance(
            id="test-default",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
        )
        assert instance.priority == InstancePriority.PRODUCTION

    def test_explicit_none_rejected(self):
        """Explicit None for priority is rejected — use omit for default."""
        with pytest.raises(ValidationError):
            Instance(
                id="test-none",
                config={
                    "backend_type": "llamacpp",
                    "model": "/tmp/test.gguf",
                    "alias": "test",
                },
                priority=None,
            )

    def test_omitted_priority_defaults_to_production(self):
        """When priority is omitted (not in dict), default is production."""
        instance = Instance.model_validate(
            {
                "id": "test-omitted",
                "config": {
                    "backend_type": "llamacpp",
                    "model": "/tmp/test.gguf",
                    "alias": "test",
                },
            }
        )
        assert instance.priority == InstancePriority.PRODUCTION


class TestExplicitPriority:
    """Setting explicit priorities."""

    def test_staging(self):
        instance = Instance(
            id="test-staging",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            priority=InstancePriority.STAGING,
        )
        assert instance.priority == InstancePriority.STAGING

    def test_ephemeral(self):
        instance = Instance(
            id="test-ephemeral",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            priority=InstancePriority.EPHEMERAL,
        )
        assert instance.priority == InstancePriority.EPHEMERAL

    def test_production_explicit(self):
        instance = Instance(
            id="test-prod",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            priority=InstancePriority.PRODUCTION,
        )
        assert instance.priority == InstancePriority.PRODUCTION


class TestOwnershipFields:
    """managed_by and intent_id ownership (S-036 / deployment-intent §5)."""

    def test_defaults_are_none(self):
        instance = Instance(
            id="test-own-default",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
        )
        assert instance.managed_by is None
        assert instance.intent_id is None

    def test_explicit_managed(self):
        instance = Instance(
            id="test-managed",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            managed_by="intent",
            intent_id="abc-123",
        )
        assert instance.managed_by == "intent"
        assert instance.intent_id == "abc-123"

    def test_managed_by_without_intent_id_allowed(self):
        """managed_by can be set without intent_id (future use)."""
        instance = Instance(
            id="test-managed-only",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            managed_by="intent",
        )
        assert instance.managed_by == "intent"
        assert instance.intent_id is None


class TestPriorityValidation:
    """Invalid priorities are rejected."""

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            Instance(
                id="test-invalid",
                config={
                    "backend_type": "llamacpp",
                    "model": "/tmp/test.gguf",
                    "alias": "test",
                },
                priority="dev",
            )


class TestSerialization:
    """Priority appears in JSON serialization."""

    def test_json_serialization(self):
        instance = Instance(
            id="test-serialize",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            priority=InstancePriority.STAGING,
            managed_by="intent",
            intent_id="abc-123",
        )
        data = instance.model_dump(mode="json")
        assert data["priority"] == "staging"
        assert data["managed_by"] == "intent"
        assert data["intent_id"] == "abc-123"

    def test_json_roundtrip(self):
        """model_dump(mode='json') → model_validate roundtrips."""
        instance = Instance(
            id="test-roundtrip",
            config={
                "backend_type": "llamacpp",
                "model": "/tmp/test.gguf",
                "alias": "test",
            },
            priority=InstancePriority.EPHEMERAL,
            managed_by="intent",
            intent_id="xyz-789",
        )
        data = instance.model_dump(mode="json")
        restored = Instance.model_validate(data)
        assert restored.priority == InstancePriority.EPHEMERAL
        assert restored.managed_by == "intent"
        assert restored.intent_id == "xyz-789"


class TestInstanceCreate:
    """InstanceCreate model carries priority/ownership."""

    def test_create_with_priority(self):
        create = InstanceCreate(
            config={"backend_type": "llamacpp", "model": "/tmp/t.gguf", "alias": "t"},
            priority="staging",
        )
        assert create.priority == "staging"
        assert create.managed_by is None
        assert create.intent_id is None

    def test_create_with_all_fields(self):
        create = InstanceCreate(
            config={"backend_type": "llamacpp", "model": "/tmp/t.gguf", "alias": "t"},
            priority="ephemeral",
            managed_by="intent",
            intent_id="int-001",
        )
        assert create.priority == "ephemeral"
        assert create.managed_by == "intent"
        assert create.intent_id == "int-001"

    def test_create_without_priority_defaults_none(self):
        create = InstanceCreate(
            config={"backend_type": "llamacpp", "model": "/tmp/t.gguf", "alias": "t"},
        )
        assert create.priority is None
        assert create.managed_by is None
        assert create.intent_id is None


class TestInstanceUpdate:
    """InstanceUpdate carries optional ownership-marker fields (D-017 disown)."""

    def test_config_only_update(self):
        update = InstanceUpdate(
            config={"backend_type": "llamacpp", "model": "/tmp/t.gguf", "alias": "t"},
        )
        assert update.config is not None
        # Not in model_fields_set → markers untouched by the route.
        assert "managed_by" not in update.model_fields_set
        assert "intent_id" not in update.model_fields_set

    def test_marker_clearing_update(self):
        update = InstanceUpdate(
            config={"backend_type": "llamacpp", "model": "/tmp/t.gguf", "alias": "t"},
            managed_by=None,
            intent_id=None,
        )
        assert "managed_by" in update.model_fields_set
        assert "intent_id" in update.model_fields_set
        assert update.managed_by is None
        assert update.intent_id is None

    def test_config_optional_for_marker_update(self):
        update = InstanceUpdate(managed_by=None, intent_id=None)
        assert update.config is None
        assert "managed_by" in update.model_fields_set
        assert "intent_id" in update.model_fields_set
