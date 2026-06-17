from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


FormatName: TypeAlias = Literal["openai", "anthropic", "responses"]
"""检测到的请求所使用的 LLM API 线路格式。"""


@dataclass(frozen=True, slots=True)
class MockRequest:
    """
    传入请求的规范化视图，与原始线路格式无关。
    这是规则匹配器和解析器接收的数据结构。
    """

    format: FormatName
    """请求来源的 API 格式路由。"""

    model: str
    """请求中的模型字符串，如 "gpt-5.4" 或 "claude-sonnet-4-6"。"""

    streaming: bool
    """客户端是否请求 SSE 流式响应（来自 `stream` 字段）。"""

    messages: Sequence[Message]
    """完整对话，已从原始格式归一化。"""

    last_message: str
    """最后一条用户消息的文本。这是大多数匹配器检查的内容。"""

    system_message: str
    """系统提示文本，如果没有则为空字符串。"""

    tools: Sequence[ToolDef] | None = field(default=None)
    """请求中的工具定义（如果有）。"""

    tool_names: Sequence[str] = field(default_factory=tuple)
    """从 `tools` 中提取的工具名称，用于 `when_tool()` 快速查找。"""

    last_tool_call_id: str | None = field(default=None)
    """当最后一条消息是工具结果时设置。用于 `when_tool_result()`。"""

    raw: dict[str, Any] | None = field(default=None)
    """原始请求体，用于我们未提取的字段。"""

    headers: dict[str, str | None] = field(default_factory=dict)
    """传入请求的 HTTP 头。"""

    path: str = field(default="")
    """被访问的 URL 路径，如 `/v1/chat/completions`。"""


@dataclass(frozen=True, slots=True)
class Message:
    """跨所有支持格式归一化的单条对话消息。"""

    role: Literal["system", "user", "assistant", "tool"]
    """消息发送者的角色。"""

    content: str
    """消息的文本内容。"""

    tool_call_id: str | None = field(default=None)
    """将结果关联回工具调用。只在 "tool" 角色消息上设置。"""


@dataclass(frozen=True, slots=True)
class ToolDef:
    """请求中 tools 数组的工具定义，已归一化跨格式。"""

    name: str
    """工具函数名称。"""

    description: str | None = field(default=None)
    """工具功能的描述。"""

    parameters: dict[str, Any] | None = field(default=None)
    """工具参数的 JSON Schema，原样传递。"""
