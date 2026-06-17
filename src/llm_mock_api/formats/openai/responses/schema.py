"""
OpenAI Responses 格式 - 请求模型定义。

请求体（任意 dict）
    ├─ ResponsesRequest.model_validate() → ResponsesRequest 实例
    ├─ input: str | list → messages（user/system/assistant/tool）
    ├─ instructions: str → system role
    └─ tools: list[FunctionTool] → 工具定义
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FunctionTool(BaseModel):
    """Responses 格式中的函数工具定义。"""

    name: str
    description: str | None = None
    parameters: dict[str, object] | None = None


class ResponsesRequestInputItem(BaseModel):
    """input 数组中的单个条目（消息或工具调用引用）。"""

    role: str | None = None
    content: str | list[dict[str, object]] | None = None
    call_id: str | None = None
    output: str | None = None
    arguments: str | None = None

    model_config = {"extra": "allow"}


class ResponsesRequest(BaseModel):
    """OpenAI Responses API 请求体。

    与 chat/completions 的核心区别：
    - 用 input（字符串或 item 数组）代替 messages
    - 用 instructions 作为单独的 system content
    - 工具函数定义使用 FunctionTool
    """

    model: str | None = None
    input: str | list[ResponsesRequestInputItem] | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}
