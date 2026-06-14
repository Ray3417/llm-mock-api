"""
Anthropic Messages API schema。

工作流示例:
    原始 JSON body
        ↓ AnthropicRequest.model_validate(body)  ← Pydantic 验证 + 结构化
        ↓ 提取 model / system / messages / tools
        ↓ 交由 parse_request 归一化为 MockRequest
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AnthropicMessage(BaseModel):
    """Anthropic 消息对象。

    对应 TS: AnthropicRequest["messages"][number]。
    content 可以是纯字符串或 content block 数组（text、tool_use、tool_result 等）。
    """

    role: str
    """消息角色，"user" 或 "assistant"。"""

    content: str | list[dict[str, Any]] = ""
    """消息内容：字符串或 content block 列表。"""

    model_config = {"extra": "allow"}


class AnthropicTool(BaseModel):
    """Anthropic 工具定义。

    对应 TS: AnthropicRequest["tools"][number]。
    注意字段名是 input_schema，而非 OpenAI 的 parameters。
    """

    name: str
    """工具函数名称。"""

    description: str | None = None
    """工具功能描述，用于让模型理解工具用途。"""

    input_schema: dict[str, Any] | None = None
    """工具参数的 JSON Schema。"""

    model_config = {"extra": "allow"}


class AnthropicRequest(BaseModel):
    """Anthropic Messages API 请求体。

    对应 TS: AnthropicRequestSchema.parse(body)。
    仅定义我们需要的字段，额外字段用 extra="allow" 容忍（等价于 TS Zod 的 .passthrough()）。
    """

    model: str | None = None
    """使用的模型名称。"""

    system: str | list[dict[str, Any]] | None = None
    """系统提示（独立字段，非 messages 中的 role="system"）。"""

    messages: list[AnthropicMessage] = []
    """对话消息列表。"""

    tools: list[AnthropicTool] | None = None
    """可选的工具定义列表。"""

    stream: bool | None = None
    """是否要求 SSE 流式响应。"""

    model_config = {"extra": "allow"}
