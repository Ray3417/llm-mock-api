"""
Anthropic Messages API 请求解析。

工作流示例:
    原始请求体 (dict/JSON)
        ↓ AnthropicRequest.model_validate(body)  ← Pydantic 验证
        ↓ _extract_system()  ← 处理独立的 system 字段（可 str 或 block 数组）
        ↓ _extract_content() ← 从 content 提取文本，识别 tool_result 块
        ↓ _parse_messages()  ← 归一化为通用 Message 列表
        ↓ _parse_tools()     ← 提取工具定义（input_schema 映射为 parameters）
        ↓ build_mock_request("anthropic", ...)  ← 生成 MockRequest
"""

from __future__ import annotations

from typing import Any, cast, Literal

from ..request_helpers import RequestMeta, build_mock_request
from ...types.request import Message, MockRequest, ToolDef
from .schema import AnthropicRequest


def _extract_system(system: str | list[dict[str, Any]] | None) -> str:
    """从 system 字段提取纯文本。

    Anthropic 的 system 可以是：
    - None → 空字符串
    - 字符串 → 直接返回
    - content block 数组 → 提取所有 type=="text" 块的 text 字段，换行连接

    对应 TS: extractSystem(req.system)
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system

    parts: list[str] = []
    for block in system:
        if isinstance(block, dict) and block.get("type") == "text":
            text_val = block.get("text")
            if text_val is not None:
                parts.append(str(text_val))
    return "\n".join(parts)


def _extract_content(content: str | list[dict[str, Any]]) -> tuple[str, str | None]:
    """从 content 字段提取 (文本, 工具结果 id)。

    Anthropic 的 content 可以是：
    - 字符串 → 作为纯文本，无 tool_result
    - content block 数组 → 提取所有 type=="text" 块的文本；
                     检查是否有 type=="tool_result" 并提取其 tool_use_id

    返回: (文本内容, tool_use_id 或 None)
    对应 TS: extractContent(req.messages[i].content) — 返回值同时含文本和 tool_use_id
    """
    if isinstance(content, str):
        return content, None

    text_parts: list[str] = []
    tool_result_id: str | None = None

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")

        if block_type == "text":
            text_val = block.get("text")
            if text_val is not None:
                text_parts.append(str(text_val))
        elif block_type == "tool_result":
            # 记录工具结果的 id（对应 TS: toolResult?.tool_use_id）
            id_val = block.get("tool_use_id")
            if id_val is not None:
                tool_result_id = str(id_val)
            # tool_result 中的 content 字段可能是字符串或块数组
            inner_content = block.get("content")
            if isinstance(inner_content, str):
                text_parts.append(inner_content)
            elif isinstance(inner_content, list):
                for sub_block in inner_content:
                    if isinstance(sub_block, dict) and sub_block.get("type") == "text":
                        sub_text = sub_block.get("text")
                        if sub_text is not None:
                            text_parts.append(str(sub_text))

    return "\n".join(text_parts), tool_result_id


def _parse_messages(req: AnthropicRequest) -> list[Message]:
    """将 Anthropic messages 归一化为通用 Message 列表。

    对应 TS: parseMessages(req)。
    特殊处理：
    - system 字段作为一条独立的 Message（role="system"）
    - assistant/user 消息按 role 原样传递
    - 含 tool_result 块的消息在 Message.tool_call_id 中记录其 tool_use_id
    """
    messages: list[Message] = []

    # system 字段 → 独立的 system role 消息
    system_text = _extract_system(req.system)
    if system_text:
        messages.append(Message(role="system", content=system_text))

    # 对话历史 → user/assistant 消息
    for m in req.messages:
        content_text, tool_use_id = _extract_content(m.content)

        role = cast(Literal["user", "assistant", "tool"], m.role)
        if tool_use_id is not None:
            messages.append(Message(
                role=role,
                content=content_text,
                tool_call_id=tool_use_id,
            ))
        else:
            messages.append(Message(
                role=role,
                content=content_text,
            ))

    return messages


def _parse_tools(req: AnthropicRequest) -> list[ToolDef] | None:
    """将 Anthropic tools 归一化为通用 ToolDef 列表。

    对应 TS: parseTools(req)。
    字段映射：AnthropicTool.input_schema → ToolDef.parameters
    """
    if not req.tools:
        return None

    return [
        ToolDef(
            name=t.name,
            description=t.description,
            parameters=t.input_schema,
        )
        for t in req.tools
    ]


def parse_request(body: dict[str, Any], meta: RequestMeta | None = None) -> MockRequest:
    """解析原始请求体为规范化的 MockRequest。

    等价于 TS:
        const req = AnthropicRequestSchema.parse(body);
        return buildMockRequest("anthropic", req, parseMessages(req), parseTools(req), ...);

    默认模型: "claude-sonnet-4-6"
    """
    parsed = AnthropicRequest.model_validate(body)
    return build_mock_request(
        "anthropic",
        parsed.model_dump(),
        _parse_messages(parsed),
        _parse_tools(parsed),
        "claude-sonnet-4-6",
        body,
        meta if meta is not None else RequestMeta(),
    )
