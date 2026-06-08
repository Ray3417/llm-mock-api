"""
OpenAI chat-completions 线路格式实现。

路由: POST /v1/chat/completions
    ├─ format.parse_request(body, meta) → MockRequest
    ├─ format.is_streaming(body) → bool
    ├─ 规则引擎匹配 → ReplyObject
    └─ format.serialize() / serialize_complete() / serialize_error() → 响应

对应 TS: export const chatCompletionsFormat: Format = { ... }
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...request_helpers import RequestMeta, is_streaming
from ...types import SSEChunk
from ....types import FormatName, MockRequest
from ....types.reply import ErrorReply, ReplyObject, ReplyOptions

from .parse import parse_request
from .serialize import serialize, serialize_complete, serialize_error


# ── Format 实现 ──────────────────────────────────────────

class _ChatCompletionsFormat:
    """OpenAI /v1/chat/completions 格式。实现 Format Protocol。"""

    @property
    def name(self) -> FormatName:
        return "openai"

    @property
    def route(self) -> str:
        return "/v1/chat/completions"

    def parse_request(self, body: Any, meta: RequestMeta | None = None) -> MockRequest:
        return parse_request(body, meta)

    def is_streaming(self, body: Any) -> bool:
        return is_streaming(body)

    def serialize(
        self,
        reply: ReplyObject,
        model: str,
        options: ReplyOptions | None = None,
    ) -> Sequence[SSEChunk]:
        return serialize(reply, model, options)

    def serialize_complete(self, reply: ReplyObject, model: str) -> dict[str, Any]:
        return serialize_complete(reply, model)

    def serialize_error(self, error: ErrorReply) -> dict[str, Any]:
        return serialize_error(error)


chat_completions_format = _ChatCompletionsFormat()
"""OpenAI chat-completions 格式单例。对应 TS: chatCompletionsFormat。"""
