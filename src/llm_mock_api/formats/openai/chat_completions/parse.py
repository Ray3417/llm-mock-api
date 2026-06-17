"""
OpenAI 请求解析。

原始请求体 (dict/JSON)
    ↓ OpenAIRequest.model_validate(body)  ← Pydantic 验证 + 结构化
    ↓ _parse_messages()  ← 含 developer→system 角色映射
    ↓ _parse_tools()     ← 提取工具定义
    ↓ build_mock_request("openai", ...)   ← 归一化为通用 MockRequest
"""

from __future__ import annotations

from typing import Any, Literal, cast

from ...request_helpers import RequestMeta, build_mock_request
from ....types.request import Message, MockRequest, ToolDef
from .schema import OpenAIRequest


Role = Literal["system", "user", "assistant", "tool"]


# ── content 解析 ──────────────────────────────────────────

def _extract_content(content: str | list[Any] | None) -> str:
    """将 OpenAI 的 content 字段归一化为纯文本。

    支持三种形态（对应 TS 的 extractContent）：
      - None → ""
      - 纯字符串 → 直接返回
      - 多模态 part 列表 → 提取所有 type=="text" 的部分，换行连接
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and part.get("text") is not None
    )


# ── messages 解析 ────────────────────────────────────────

def _parse_messages(req: OpenAIRequest) -> list[Message]:
    """将 OpenAI messages 归一化为通用 Message 列表。

    对应 TS parseMessages，特殊处理：
      - role=="developer" → 映射为 "system"（新 API 用 developer 替代 system）
      - role 缺失/空 → 默认 "user"
      - 只有显式含 tool_call_id 的消息才携带此字段
    """
    messages: list[Message] = []
    for m in req.messages:
        if m.role == "developer":
            raw_role = "system"
        elif m.role:
            raw_role = m.role
        else:
            raw_role = "user"

        role = cast(Role, raw_role)
        content = _extract_content(m.content)

        if m.tool_call_id is not None:
            messages.append(Message(role=role, content=content, tool_call_id=m.tool_call_id))
        else:
            messages.append(Message(role=role, content=content))
    return messages


# ── tools 解析 ────────────────────────────────────────────

def _parse_tools(req: OpenAIRequest) -> list[ToolDef] | None:
    """将 OpenAI tools 归一化为通用 ToolDef 列表。"""
    if not req.tools:
        return None
    return [
        ToolDef(
            name=t.function.name,
            description=t.function.description,
            parameters=t.function.parameters,
        )
        for t in req.tools
    ]


# ── 入口：parse_request ─────────────────────────────────

def parse_request(body: dict[str, Any], meta: RequestMeta | None = None) -> MockRequest:
    """解析原始请求体为规范化的 MockRequest。

    等价于 TS:
        const req = OpenAIRequestSchema.parse(body);
        return buildMockRequest("openai", req, parseMessages(req), ...);
    """
    parsed = OpenAIRequest.model_validate(body)
    return build_mock_request(
        "openai",
        parsed.model_dump(),
        _parse_messages(parsed),
        _parse_tools(parsed),
        "gpt-5.4",
        body,
        meta if meta is not None else RequestMeta(),
    )
