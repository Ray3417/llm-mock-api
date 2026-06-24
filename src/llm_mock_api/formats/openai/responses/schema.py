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


class FunctionCallOutputSchema(BaseModel):
    """✨ NEW：Responses 中工具调用的"输出"项（item 含 output + call_id）。"""

    output: str | None = None
    call_id: str

    model_config = {"extra": "allow"}


class FunctionCallInputSchema(BaseModel):
    """✨ NEW：Responses 中工具调用的"输入"项（item 含 arguments + call_id）。"""

    arguments: str | None = None
    call_id: str

    model_config = {"extra": "allow"}


class InputMessageSchema(BaseModel):
    """✨ NEW：Responses 中普通消息项（item 含 role + content，无 call_id）。"""

    role: str | None = None
    content: str | list[dict[str, object]] | None = None

    model_config = {"extra": "allow"}


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
    # ✨ NEW: input 仅做基础解析：字符串或 item 字典列表，由 parse.py 中
    # 通过三个 schema 逐次 safeParse 出每条消息。这样 parse 阶段
    # 才能通过 schema 安全地做 flatMap。
    input: str | list[dict[str, Any]] | None = None
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None

    model_config = {"extra": "allow"}
