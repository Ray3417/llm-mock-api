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
from .schema import OpenAICustomTool, OpenAIRequest, OpenAIToolFunction


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


# ── 角色归一化（显式穷举，对应 TS normaliseRole） ─────────────

def _normalise_role(role: str | None) -> Role:
    """✨ NEW：将 OpenAI messages 中的 role 归一化为内部 Role。

    显式穷举处理：
      - None / "user"         → "user"
      - "developer" / "system" → "system"
      - "function" / "tool"   → "tool"
      - "assistant"            → "assistant"
    对应 TS 的 switch 穷举 + default exhaustive 校验。
    """
    if role is None or role == "user":
        return "user"
    if role == "developer" or role == "system":
        return "system"
    if role == "function" or role == "tool":
        return "tool"
    if role == "assistant":
        return "assistant"
    # 兜底：未知角色一律视为 user，避免抛错
    return cast(Role, "user")


# ── messages 解析 ────────────────────────────────────────

def _parse_messages(req: OpenAIRequest) -> list[Message]:
    """将 OpenAI messages 归一化为通用 Message 列表。

    使用显式穷举的 `_normalise_role`；只有显式含 tool_call_id 的消息才带上该字段。
    """
    messages: list[Message] = []
    for m in req.messages:
        role = _normalise_role(m.role)
        content = _extract_content(m.content)

        if m.tool_call_id is not None:
            messages.append(Message(role=role, content=content, tool_call_id=m.tool_call_id))
        else:
            messages.append(Message(role=role, content=content))
    return messages


# ── tools 解析 ────────────────────────────────────────────

def _parse_tools(req: OpenAIRequest) -> list[ToolDef] | None:
    """将 OpenAI tools 归一化为通用 ToolDef 列表。

    ✨ NEW：除传统 function tool 外，还识别 GPT-5 family 的 custom 工具
    （无 parameters，但保留 name + description 用于匹配）。
    """
    if not req.tools:
        return None
    result: list[ToolDef] = []
    for t in req.tools:
        if t.type == "custom":
            try:
                custom = OpenAICustomTool.model_validate(t.custom or {})
            except Exception:
                continue
            result.append(ToolDef(name=custom.name, description=custom.description))
            continue
        # 默认视为 function tool
        try:
            func = OpenAIToolFunction.model_validate(t.function or {})
        except Exception:
            continue
        result.append(ToolDef(
            name=func.name,
            description=func.description,
            parameters=func.parameters,
        ))
    return result


# ── 入口：parse_request ─────────────────────────────────

def parse_request(body: dict[str, Any], meta: RequestMeta | None = None) -> MockRequest:
    """解析原始请求体为规范化的 MockRequest。

    ✨ NEW：Chat Completions 不使用 server-side tool（内置工具如 web_search
    在独立的字段上），这里显式传入空的 server_tool_types。
    """
    parsed = OpenAIRequest.model_validate(body)
    return build_mock_request(
        "openai",
        parsed.model_dump(),
        _parse_messages(parsed),
        _parse_tools(parsed),
        [],  # ✨ NEW：chat-completions 没有 server-side tool
        "gpt-5.4",
        body,
        meta if meta is not None else RequestMeta(),
    )
