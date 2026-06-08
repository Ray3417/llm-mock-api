"""
Formats 模块基础类型。

用户调用 format.serialize(reply, model, options)
    ↓
返回 SSEChunk 列表，用于 write_sse 流式输出
    ↓
或 format.serialize_complete(reply, model) 用于非流式 JSON 响应
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ..types import ErrorReply, FormatName, MockRequest, ReplyObject, ReplyOptions
from .request_helpers import RequestMeta


@dataclass(frozen=True, slots=True, kw_only=True)
class SSEChunk:
    """SSE 数据块。被 write_sse 格式化为 "event: ...\ndata: ...\n\n" 字符串。"""

    event: str | None = None
    """可选事件名。若提供，输出 "event: xxx\n" 前缀。"""

    data: str
    """SSE 事件的 data 内容（通常是 JSON 字符串）。"""


class Format(Protocol):
    """API 线路格式接口。每种格式（OpenAI、Anthropic、Responses 等）实现此 Protocol。"""

    @property
    def name(self) -> FormatName:
        """格式标识名，用于 MockRequest.format 字段。"""
        ...

    @property
    def route(self) -> str:
        """此格式匹配的 HTTP 路由路径，如 "/v1/chat/completions"。"""
        ...

    def parse_request(
        self, body: Any, meta: RequestMeta | None = None
    ) -> MockRequest:
        """解析原始请求体为规范化的 MockRequest。"""
        ...

    def is_streaming(self, body: Any) -> bool:
        """判断请求是否要求 SSE 流式响应。"""
        ...

    def serialize(
        self,
        reply: ReplyObject,
        model: str,
        options: ReplyOptions | None = None,
    ) -> Sequence[SSEChunk]:
        """将回复序列化为 SSE 数据块列表（流式响应）。"""
        ...

    def serialize_complete(self, reply: ReplyObject, model: str) -> dict[str, Any]:
        """将回复序列化为完整的 JSON 对象（非流式响应）。"""
        ...

    def serialize_error(self, error: ErrorReply) -> dict[str, Any]:
        """将错误对象序列化为格式特定的 JSON 错误响应。"""
        ...
