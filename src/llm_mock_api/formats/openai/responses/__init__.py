"""OpenAI Responses 线路格式实现。

路由: POST /v1/responses

对应 TS: export const responsesFormat: Format = { ... }
"""

from __future__ import annotations

from ... import FormatImpl
from ...request_helpers import is_streaming
from .parse import parse_request
from .serialize import serialize, serialize_complete, serialize_error


responses_format = FormatImpl(
    name="responses",
    route="/v1/responses",
    parse_request=parse_request,
    is_streaming=is_streaming,
    serialize=serialize,
    serialize_complete=serialize_complete,
    serialize_error=serialize_error,
)
