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

from typing import Any, cast

from pydantic import ValidationError

from ...request_helpers import RequestMeta, build_mock_request
from ....types import Message, MockRequest, ToolDef
from .schema import FunctionTool, ResponsesRequest


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
    """instructions → system role；input → messages 数组。

    developer 角色在 Responses 中需映射为 system（与 chat_completions 相同的角色归一化）。
    含 call_id 的项视为工具调用输出，映射为 tool role。
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
        if item.call_id is not None:
            # 工具调用结果：call_id 存在
            # TS: "output" in item ? item.output : item.arguments
            if "output" in item.model_fields_set:
                content = item.output or ""
            else:
                content = item.arguments or ""
            messages.append(Message(
                role="tool",
                content=content,
                tool_call_id=item.call_id,
            ))
        else:
            raw_role = item.role
            role = "system" if raw_role == "developer" else raw_role
            messages.append(Message(
                role=cast(Any, role),
                content=_extract_input_content(item.content),
            ))

    return messages


def _parse_tools(req: ResponsesRequest) -> list[ToolDef] | None:
    """遍历 tools 数组，用 FunctionTool 逐项目验证并提取 ToolDef。

    与 chat_completions 的区别：FunctionTool 使用 Responses 专用模型而非 OpenAI 的 tool 格式。
    验证失败的单个工具会被跳过（对应 TS: safeParse + filter(r.success)）。
    """
    if not req.tools:
        return None

    result: list[ToolDef] = []
    for raw_tool in req.tools:
        try:
            ft = FunctionTool.model_validate(raw_tool)
            result.append(ToolDef(
                name=ft.name,
                description=ft.description,
                parameters=ft.parameters,
            ))
        except ValidationError:
            # 验证失败的工具静默跳过，不影响其他工具
            continue
    return result


def parse_request(body: Any, meta: RequestMeta | None = None) -> MockRequest:
    """Responses API 请求解析入口。

    使用 "responses" 作为格式名，默认模型 "codex-mini"。
    """
    parsed = ResponsesRequest.model_validate(body)
    return build_mock_request(
        "responses",
        parsed.model_dump(),
        _parse_input(parsed),
        _parse_tools(parsed),
        "codex-mini",
        body,
        meta if meta is not None else RequestMeta(),
    )
