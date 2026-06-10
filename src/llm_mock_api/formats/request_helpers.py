"""
请求解析辅助函数。

route-handler 接收原始 HTTP body
    ↓
format.parse_request(body, meta) 内部调用
    ├─ is_streaming(body) → 判断是否流式请求
    └─ build_mock_request(...) → 统一为 MockRequest 结构
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..types.request import FormatName, Message, MockRequest, ToolDef


@dataclass(frozen=True, slots=True)
class RequestMeta:
    """请求元数据：HTTP 头和路径。由 route-handler 传递给 parse_request。"""

    headers: Mapping[str, str | None] = field(default_factory=dict)
    """请求的 HTTP 头，保持原值不做处理。"""

    path: str = ""
    """被访问的 URL 路径。"""


EMPTY_META: RequestMeta = RequestMeta()
"""空的默认元数据。"""


def is_streaming(body: Any) -> bool:
    """检查请求体是否要求流式响应。

    语义：仅当 `stream` 字段显式为 `False` 时返回 False；
    其他所有情况（缺失、True、非布尔值）默认视为流式，
    与 LLM API 的普遍默认行为一致。
    """
    if not isinstance(body, Mapping):
        return True
    return body.get("stream") is not False


def build_mock_request(
    format: FormatName,
    body: Mapping[str, Any],
    messages: Sequence[Message],
    tools: Sequence[ToolDef] | None,
    default_model: str,
    raw: Any,
    meta: RequestMeta = EMPTY_META,
) -> MockRequest:
    """从格式特定数据构造规范化的 MockRequest。

    从 messages 中提取最后一条用户消息、系统提示以及工具调用信息。
    """
    user_messages = [m for m in messages if m.role == "user"]
    tool_call_messages = [m for m in messages if m.tool_call_id is not None]

    return MockRequest(
        format=format,
        model=body.get("model") or default_model,
        streaming=body.get("stream") is True,
        messages=messages,
        last_message=user_messages[-1].content if user_messages else "",
        system_message=next((m.content for m in messages if m.role == "system"), ""),
        tools=tools,
        tool_names=tuple(t.name for t in tools) if tools is not None else (),
        last_tool_call_id=(tool_call_messages[-1].tool_call_id if tool_call_messages else None),
        raw=raw,
        headers=dict(meta.headers),
        path=meta.path,
    )
