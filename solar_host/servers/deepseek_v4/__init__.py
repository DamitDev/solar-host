"""DeepSeek-V4 prompt encoding/decoding helpers (vendored from upstream).

DeepSeek-V4 ships no Jinja chat template; instead the model card provides a
Python encoder/parser at
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py
which we vendor verbatim under MIT (see VENDORED.md).

This package re-exports the public surface so callers can do::

    from solar_host.servers.deepseek_v4 import encode_messages, parse_message_from_completion_text
"""

from .encoding_dsv4 import (
    encode_messages,
    parse_message_from_completion_text,
    merge_tool_messages,
    sort_tool_results_by_call_order,
    bos_token,
    eos_token,
    thinking_start_token,
    thinking_end_token,
    dsml_token,
    USER_SP_TOKEN,
    ASSISTANT_SP_TOKEN,
)

__all__ = [
    "encode_messages",
    "parse_message_from_completion_text",
    "merge_tool_messages",
    "sort_tool_results_by_call_order",
    "bos_token",
    "eos_token",
    "thinking_start_token",
    "thinking_end_token",
    "dsml_token",
    "USER_SP_TOKEN",
    "ASSISTANT_SP_TOKEN",
]
