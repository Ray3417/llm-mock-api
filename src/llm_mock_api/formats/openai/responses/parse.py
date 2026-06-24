"""
OpenAI Responses 格式 - 请求解析。

请求体（JSON/dict）
    ├─ ResponsesRequest.model_validate(body) → 结构化对象
    ├─ _extract_input_content() → 从 content 提取纯文本
    ├─ _parse_input() → instructions（system）+ input → Message[]
    ├─ _parse_tools() → FunctionTool → ToolDef[]
    └─ build_mock_request("responses", ...) → MockRequest
"""

from __future__ import annotations

from typing import Any, Literal, cast

from pydantic import ValidationError

from ...request_helpers import RequestMeta, build_mock_request
from ....types.request import Message, MockRequest, ToolDef
from .schema import (
    FunctionCallInputSchema,
    FunctionCallOutputSchema,
    FunctionTool,
    InputMessageSchema,
    ResponsesRequest,
)


def _extract_input_content(content: str | list[dict[str, object]] | None) -> str:
    """从 content 字段提取纯文本。

    字符串直接返回；数组筛选 type 为 input_text/text 的项并按行拼接。
    """
    if content is None or isinstance(content, str):
        return content or ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("input_text", "text"):
            parts.append(str(block.get("text", "")))
    return "\n".join(parts)


def _parse_input(req: ResponsesRequest) -> list[Message]:
    """✨ NEW：instructions → system role；input item 用三个 schema 逐次 safeParse。

    解析顺序（对应 TS flatMap safeParse）：
      1. FunctionCallOutputSchema（含 output + call_id）→ tool role
      2. FunctionCallInputSchema（含 arguments + call_id）→ tool role
      3. InputMessageSchema（含 role + content）→ 对应角色
    解析不到任何一个 schema 的 item 被静默跳过。
    """
    messages: list[Message] = []

    if req.instructions:
        messages.append(Message(role="system", content=req.instructions))

    if req.input is None:
        return messages

    if isinstance(req.input, str):
        messages.append(Message(role="user", content=req.input))
        return messages

    for item in req.input:
        # 尝试 FunctionCallOutput
        try:
            output = FunctionCallOutputSchema.model_validate(item)
            messages.append(Message(
                role="tool",
                content=output.output or "",
                tool_call_id=output.call_id,
            ))
            continue
        except ValidationError:
            pass

        # 尝试 FunctionCallInput
        try:
            call = FunctionCallInputSchema.model_validate(item)
            messages.append(Message(
                role="tool",
                content=call.arguments or "",
                tool_call_id=call.call_id,
            ))
            continue
        except ValidationError:
            pass

        # 尝试 InputMessage
        try:
            msg = InputMessageSchema.model_validate(item)
            role = "system" if msg.role == "developer" else (msg.role or "user")
            messages.append(Message(
                role=cast(Literal["system", "user", "assistant", "tool"], role),
                content=_extract_input_content(msg.content),
            ))
            continue
        except ValidationError:
            pass

    return messages


def _parse_tools(req: ResponsesRequest) -> list[ToolDef] | None:
    """✨ NEW：遍历 tools 数组，用 FunctionTool 逐项目 safeParse，flatMap 出结果。"""
    if not req.tools:
        return None

    result: list[ToolDef] = []
    for raw_tool in req.tools:
        try:
            ft = FunctionTool.model_validate(raw_tool)
        except ValidationError:
            continue
        result.append(ToolDef(
            name=ft.name,
            description=ft.description,
            parameters=ft.parameters,
        ))
    return result


def _parse_server_tool_types(req: ResponsesRequest) -> list[str]:
    """✨ NEW：提取 tools 中所有非 function 工具的 type。

    对应 TS: parseServerToolTypes(req) —— 在 Responses 中，web_search、
    file_search 等内置工具是 tools 数组中的一项（type != "function"）。
    """
    if not req.tools:
        return []

    types: list[str] = []
    for raw_tool in req.tools:
        tool_type = raw_tool.get("type") if isinstance(raw_tool, dict) else None
        if isinstance(tool_type, str) and tool_type != "function":
            types.append(tool_type)
    return types


def parse_request(body: dict[str, Any], meta: RequestMeta | None = None) -> MockRequest:
    """Responses API 请求解析入口。

    ✨ NEW：新增 parseServerToolTypes 调用，用于收集内置工具类型。
    使用 "responses" 作为格式名，默认模型 "codex-mini"。
    """
    parsed = ResponsesRequest.model_validate(body)
    return build_mock_request(
        "responses",
        parsed.model_dump(),
        _parse_input(parsed),
        _parse_tools(parsed),
        _parse_server_tool_types(parsed),  # ✨ NEW
        "codex-mini",
        body,
        meta if meta is not None else RequestMeta(),
    )
