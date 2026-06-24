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


# ✨ NEW：以下两个模型用于区分用户自定义工具与 Anthropic 内置工具（如 web_search_20250305）。
# AnthropicRequest.tools 数组中的每一项可能是用户自定义 tool（含 input_schema）
# 也可能是内置工具（含 type 字段，如 "web_search_20250305"）。我们在 parse_tools 中
# 用这两个模型分别 safeParse 来归类。

class ToolDefinitionSchema(BaseModel):
    """用户自定义 tool 的最小 schema。对应 TS: ToolDefinitionSchema。"""

    name: str
    """工具函数名称。"""

    description: str | None = None
    """工具功能描述。"""

    input_schema: dict[str, Any] | None = None
    """工具参数的 JSON Schema。"""

    model_config = {"extra": "allow"}


class ServerToolSchema(BaseModel):
    """Anthropic 内置 server-side tool 的最小 schema。对应 TS: ServerToolSchema。"""

    type: str
    """内置工具类型标识符，如 "web_search_20250305"。"""

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

    # ✨ NEW：保持 tools 为原始 dict，不提前解析为 AnthropicTool 实例。
    # 因为 parse 阶段会用 ToolDefinitionSchema / ServerToolSchema 两个独立
    # schema 逐次 safeParse（对应 TS 的 `.flatMap(t => ToolDefinitionSchema.safeParse(t))`）。
    # 如果这里提前解析，Pydantic 的 model_validate 会拒绝跨类实例。
    tools: list[dict[str, Any]] | None = None
    """可选的工具定义列表（保持为原始 dict，不做提前解析）。"""

    stream: bool | None = None
    """是否要求 SSE 流式响应。"""

    model_config = {"extra": "allow"}
