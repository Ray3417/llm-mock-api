"""OpenAI chat-completions 线路格式实现。

路由: POST /v1/chat/completions

对应 TS: export const chatCompletionsFormat: Format = { ... }
"""

from __future__ import annotations

from ... import FormatImpl
from ...request_helpers import is_streaming
from .parse import parse_request
from .serialize import serialize, serialize_complete, serialize_error


chat_completions_format = FormatImpl(
    name="openai",
    route="/v1/chat/completions",
    parse_request=parse_request,
    is_streaming=is_streaming,
    serialize=serialize,
    serialize_complete=serialize_complete,
    serialize_error=serialize_error,
)
"""OpenAI chat-completions 格式单例。"""
