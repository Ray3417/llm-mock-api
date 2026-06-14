"""Anthropic Messages API 线路格式实现。

路由: POST /v1/messages

对应 TS: export const anthropicFormat: Format = { ... }
"""

from __future__ import annotations

from .. import FormatImpl
from ..request_helpers import is_streaming
from .parse import parse_request
from .serialize import serialize, serialize_complete, serialize_error


anthropic_format = FormatImpl(
    name="anthropic",
    route="/v1/messages",
    parse_request=parse_request,
    is_streaming=is_streaming,
    serialize=serialize,
    serialize_complete=serialize_complete,
    serialize_error=serialize_error,
)
