"""
OpenAI chat-completions schema。

请求验证流程：
    原始 JSON body
        ↓ OpenAIRequest.model_validate(body)  ← Pydantic 运行时验证
        ↓ 提取 model / messages / tools / stream

响应字段约定（不使用 Pydantic，直接构造 dict）：
    流式 chunk:   { id, object: "chat.completion.chunk", created, model, choices:[{ delta }], ... }
    完整响应:     { id, object: "chat.completion", created, model, choices:[{ message }], ... }
    错误响应:     { error: { message, type, code } }
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


# ── 请求内容部分（多模态） ───────────────────────────────

class OpenAIContentPart(BaseModel):
    """消息 content 列表中的单个部分。

    只关心 type=="text" 的文本部分，其他内容类型（如 image_url）透传保留。
    """

    type: str
    """内容部分类型，如 "text"、"image_url"。"""

    text: str | None = None
    """当 type=="text" 时存在的文本内容。"""

    model_config = {"extra": "allow"}


# ── 请求消息 ─────────────────────────────────────────────

class OpenAIMessage(BaseModel):
    """请求中的单条消息，对应 OpenAI messages 数组元素。"""

    role: str
    """发送者角色："system"、"user"、"assistant"、"tool"、"developer" 等。"""

    content: str | list[OpenAIContentPart] | None = None
    """消息内容：字符串 或 多模态 part 列表 或 None（如纯 tool_calls 消息）。"""

    tool_call_id: str | None = None
    """关联到工具调用的 ID，仅 role=="tool" 时存在。"""

    model_config = {"extra": "allow"}


# ── 请求工具定义 ──────────────────────────────────────────

class OpenAIToolFunction(BaseModel):
    """工具函数描述。"""

    name: str
    """函数名。"""

    description: str | None = None
    """函数描述，供模型理解用途。"""

    parameters: dict[str, Any] | None = None
    """JSON Schema 描述的参数。"""


class OpenAITool(BaseModel):
    """请求中的 tool 定义。"""

    type: str = "function"
    """工具类型，目前总是 "function"。"""

    function: OpenAIToolFunction
    """函数的具体定义。"""


# ── 完整请求 ─────────────────────────────────────────────

class OpenAIRequest(BaseModel):
    """OpenAI Chat Completions API 请求体。

    仅定义我们需要的字段，额外字段用 extra="allow" 容忍，
    等价于 TS Zod 的 `.passthrough()`。
    """

    model: str | None = None
    """使用的模型名称。"""

    messages: list[OpenAIMessage]
    """对话消息列表。"""

    tools: list[OpenAITool] | None = None
    """可选的工具定义列表。"""

    stream: bool | None = None
    """是否要求流式响应。"""

    model_config = {"extra": "allow"}
