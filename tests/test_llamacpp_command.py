"""Tests for llama.cpp command construction."""

from types import SimpleNamespace

from solar_host.backends.llamacpp import LlamaCppRunner
from solar_host.models.llamacpp import LlamaCppConfig


def build_command(**config_overrides: object) -> list[str]:
    config = LlamaCppConfig(model="/models/test.gguf", alias="test", **config_overrides)
    instance = SimpleNamespace(config=config, port=8080)
    return LlamaCppRunner().build_command(instance)


def test_speculative_decoding_flags_are_omitted_by_default() -> None:
    command = build_command()

    assert "--spec-type" not in command
    assert "--spec-draft-n-max" not in command


def test_draft_mtp_speculative_decoding_flags_are_added_together() -> None:
    command = build_command(spec_type="draft-mtp", spec_draft_n_max=2)

    spec_type_index = command.index("--spec-type")
    assert command[spec_type_index : spec_type_index + 4] == [
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
    ]


def test_speculative_decoding_flags_are_ignored_for_non_generation_models() -> None:
    command = build_command(model_type="embedding", spec_type="draft-mtp", spec_draft_n_max=2)

    assert "--spec-type" not in command
    assert "--spec-draft-n-max" not in command
