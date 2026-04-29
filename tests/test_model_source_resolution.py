"""Tests for model source URI resolution and instance configuration parsing."""

import pytest
from pathlib import Path
from solar_host.config import resolve_model_source, parse_instance_config
from solar_host.models.llamacpp import LlamaCppConfig
from solar_host.models.huggingface import HuggingFaceCausalConfig


@pytest.fixture
def mock_models_dir(tmp_path: Path, monkeypatch):
    """Point settings.models_dir at a temporary directory for tests."""
    models_dir = (tmp_path / "models").resolve()
    models_dir.mkdir()
    # Need to monkeypatch where it's used
    monkeypatch.setattr("solar_host.config.settings.models_dir", str(models_dir))
    return models_dir


class TestModelSourceResolution:
    def test_resolve_plain_path(self):
        # Plain path should be returned as-is
        path = "/tmp/model.gguf"
        assert resolve_model_source(path) == path

    def test_resolve_local_relative(self, mock_models_dir):
        # local://path resolves relative to MODELS_DIR
        model_rel_path = "subdir/model.gguf"
        uri = f"local://{model_rel_path}"
        expected = str((mock_models_dir / model_rel_path).resolve())
        assert resolve_model_source(uri) == expected

    def test_resolve_local_absolute_inside(self, mock_models_dir):
        # local:///<abs_path> resolves as absolute but must be inside MODELS_DIR
        abs_path = (mock_models_dir / "abs_model.gguf").resolve()
        uri = f"local://{abs_path}"
        assert resolve_model_source(uri) == str(abs_path)

    def test_resolve_local_absolute_outside_raises(self, tmp_path, mock_models_dir):
        # local:///<path> outside MODELS_DIR raises ValueError
        outside_path = (tmp_path / "outside.gguf").resolve()
        uri = f"local://{outside_path}"
        with pytest.raises(ValueError, match="outside of MODELS_DIR"):
            resolve_model_source(uri)

    def test_reject_repo_uri(self):
        with pytest.raises(
            ValueError, match="Model must be resolved via POST /models/pull"
        ):
            resolve_model_source("repo://some-model:v1")

    def test_reject_huggingface_uri(self):
        with pytest.raises(
            ValueError, match="Model must be resolved via POST /models/pull"
        ):
            resolve_model_source("huggingface://some-org/some-model")


class TestParseInstanceConfigWithModelSource:
    def test_llamacpp_with_model_source(self, mock_models_dir):
        config_data = {
            "backend_type": "llamacpp",
            "model_source": "local://model.gguf",
            "alias": "test-alias",
        }
        config = parse_instance_config(config_data)
        assert isinstance(config, LlamaCppConfig)
        assert config.model == str((mock_models_dir / "model.gguf").resolve())
        assert config.model_source == "local://model.gguf"

    def test_huggingface_with_model_source(self, mock_models_dir):
        config_data = {
            "backend_type": "huggingface_causal",
            "model_source": "local://hf-model",
            "alias": "hf-alias",
        }
        config = parse_instance_config(config_data)
        assert isinstance(config, HuggingFaceCausalConfig)
        assert config.model_id == str((mock_models_dir / "hf-model").resolve())
        assert config.model_source == "local://hf-model"

    def test_backward_compatibility_llamacpp(self):
        config_data = {
            "backend_type": "llamacpp",
            "model": "/path/to/model.gguf",
            "alias": "test-alias",
        }
        config = parse_instance_config(config_data)
        assert isinstance(config, LlamaCppConfig)
        assert config.model == "/path/to/model.gguf"
        assert config.model_source is None

    def test_missing_both_raises(self):
        config_data = {"backend_type": "llamacpp", "alias": "test-alias"}
        # parse_instance_config calls LlamaCppConfig(**config_data) which will trigger validator
        with pytest.raises(
            ValueError, match="Either 'model' or 'model_source' must be provided"
        ):
            parse_instance_config(config_data)
