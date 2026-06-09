"""
序列化辅助函数。各格式的 serialize 模块共用这些工具。

format.serialize(reply, model, options) 内部调用:
    ├─ split_text(text, chunk_size) → 文本分段
    ├─ gen_id(prefix) → 生成事件 ID
    ├─ tool_id(tool, prefix, index) → 生成工具调用 ID
    ├─ should_emit_text(reply) → 是否需生成文本 delta
    └─ finish_reason(reply, on_tools, on_stop) → 结束原因
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping

from ..types.reply import ReplyObject, Usage


MS_PER_SECOND: int = 1000
"""毫秒/秒，用于时间戳转换（Date.now() / MS_PER_SECOND）。"""

DEFAULT_USAGE: Usage = Usage(input=10, output=5)
"""默认 token 用量（无具体值时使用）。"""


def split_text(text: str, chunk_size: int) -> list[str]:
    """按字符数将文本分块。

    chunk_size <= 0 或文本不大于 chunk_size → 原样返回单元素列表。
    """
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


_ID_SUFFIX_LENGTH: int = 12


def _random_suffix() -> str:
    """生成 12 位十六进制随机字符串。"""
    return uuid.uuid4().hex[:_ID_SUFFIX_LENGTH]


def gen_id(prefix: str) -> str:
    """生成格式: `prefix_随机12字符` 的 ID。"""
    return f"{prefix}_{_random_suffix()}"


def tool_id(tool: Mapping[str, object], prefix: str, index: int) -> str:
    """生成工具调用 ID。

    优先使用 `tool["id"]` 的值，否则自动生成 `prefix_随机_index`。
    """
    tool_id_value = tool.get("id")
    if tool_id_value is not None:
        return str(tool_id_value)
    return f"{prefix}_{_random_suffix()}_{index}"


def should_emit_text(reply: ReplyObject) -> bool:
    """判断是否应为此回复生成文本 delta。

    规则：有 text 内容，或（无 tools 也无 reasoning）时返回 True。
    """
    tools_empty = not reply.tools
    return bool(reply.text) or (tools_empty and not reply.reasoning)


def finish_reason(
    reply: ReplyObject,
    on_tools: str,
    on_stop: str,
) -> str:
    """根据回复内容决定 finish_reason 字符串。

    若回复包含工具调用 → 返回 `on_tools`；否则 → 返回 `on_stop`。
    """
    return on_tools if reply.tools else on_stop
