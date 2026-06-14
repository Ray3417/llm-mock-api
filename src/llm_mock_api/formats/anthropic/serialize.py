"""
Anthropic Messages API 响应序列化。

工作流示例:
    reply (ReplyObject)
        ├─ serialize() → SSEChunk 列表（流式）
        │     message_start
        │     ├─ reasoning?  ← content block（thinking）
        │     ├─ text?       ← content block（text，可分片）
        │     └─ tools           ← content blocks（tool_use）
        │     message_delta （stop_reason + usage）
        │     message_stop
        ├─ serialize_complete() → dict（非流式完整 JSON）
        └─ serialize_error()  → { type: "error", error: {...} }
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from ..serialize_helpers import (
    DEFAULT_USAGE,
    finish_reason,
    gen_id,
    split_text,
    tool_id,
)
from ..types import SSEChunk
from ...types.reply import ErrorReply, ReplyObject, ReplyOptions, Usage, ToolCall


def _build_usage(usage: Usage) -> dict[str, object]:
    """将内部 Usage 转换为 Anthropic 格式的 usage dict。

    Anthropic 的 usage 只包含 input_tokens 和 output_tokens，
    不汇总 total。
    """
    return {"input_tokens": usage.input, "output_tokens": usage.output}


def _content_block(
    index: int,
    start_block: dict[str, object],
    deltas: list[SSEChunk],
) -> list[SSEChunk]:
    """构造单个 content block 的完整 chunk 序列。

    对应 TS: contentBlock(index, startBlock, deltas) → SSEChunk[]
    三段式：
        content_block_start → [content_block_start]
        deltas          → [content_block_delta, ...]
        content_block_stop → [content_block_stop]
    """
    result: list[SSEChunk] = [
        SSEChunk(
            event="content_block_start",
            data=json.dumps({"type": "content_block_start", "index": index, "content_block": start_block}),
        ),
    ]
    result.extend(deltas)
    result.append(
        SSEChunk(
            event="content_block_stop",
            data=json.dumps({"type": "content_block_stop", "index": index}),
        )
    )
    return result


def _delta(index: int, payload: dict[str, object]) -> SSEChunk:
    """构造单个 content_block_delta chunk。

    对应 TS: delta(index, payload)。
    """
    return SSEChunk(
        event="content_block_delta",
        data=json.dumps({"type": "content_block_delta", "index": index, "delta": payload}),
    )


def _reasoning_block(i: int, reasoning: str) -> list[SSEChunk]:
    """构造 thinking 类型的 content block。

    对应 TS: reasoningBlock(i, reasoning)
    → 1 个 start + 1 个 thinking_delta + stop
    """
    return _content_block(
        i,
        {"type": "thinking", "thinking": ""},
        [_delta(i, {"type": "thinking_delta", "thinking": reasoning})],
    )


def _text_block(i: int, text: str, chunk_size: int) -> list[SSEChunk]:
    """构造 text 类型的 content block（可分片）。

    对应 TS: textBlock(i, text, chunkSize)
    → 1 个 start + N 个 text_delta + stop
    """
    return _content_block(
        i,
        {"type": "text", "text": ""},
        [_delta(i, {"type": "text_delta", "text": piece}) for piece in split_text(text, chunk_size)],
    )


def _tool_blocks(start_index: int, tools: Sequence[ToolCall]) -> list[SSEChunk]:
    """为每个 ToolCall 构造一个 tool_use 类型的 content block。

    对应 TS: toolBlocks(startIndex, tools)。
    每个 tool: 1 个 start → 1 个 input_json_delta（整个 JSON 字符串）→ stop
    """
    all_chunks: list[SSEChunk] = []
    for i, tool in enumerate(tools):
        idx = start_index + i
        call_id = tool_id({"id": tool.id}, "toolu", i)
        all_chunks.extend(
            _content_block(
                idx,
                {"type": "tool_use", "id": call_id, "name": tool.name, "input": {}},
                [_delta(idx, {"type": "input_json_delta", "partial_json": json.dumps(tool.args)})],
            )
        )
    return all_chunks


def serialize(
    reply: ReplyObject,
    model: str,
    options: ReplyOptions | None = None,
) -> Sequence[SSEChunk]:
    """将 ReplyObject 序列化为 SSEChunk 列表（流式响应）。

    chunk 顺序严格对应 TS:
        1. message_start — 声明 id, 消息开始，usage 中的 output_tokens = 0
        2. reasoning? — thinking block
        3. text? — text block（可分片）
        4. tools — tool_use blocks
        5. message_delta — stop_reason + output_tokens
        6. message_stop — 结束标志
    """
    msg_id = gen_id("msg")
    usage = reply.usage or DEFAULT_USAGE
    opts = options or ReplyOptions()

    chunks: list[SSEChunk] = []
    idx = 0

    # 1. message_start
    chunks.append(
        SSEChunk(
            event="message_start",
            data=json.dumps(
                {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": usage.input, "output_tokens": 0},
                    },
                },
            ),
        ),
    )

    # 2. reasoning
    if reply.reasoning:
        chunks.extend(_reasoning_block(idx, reply.reasoning))
        idx += 1

    # 3. text
    if reply.text:
        chunks.extend(_text_block(idx, reply.text, opts.chunk_size or 0))
        idx += 1

    # 4. tools
    if reply.tools:
        chunks.extend(_tool_blocks(idx, reply.tools))

    # 5. message_delta
    chunks.append(
        SSEChunk(
            event="message_delta",
            data=json.dumps(
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": finish_reason(reply, "tool_use", "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": usage.output},
                },
            ),
        ),
    )

    # 6. message_stop
    chunks.append(
        SSEChunk(event="message_stop", data=json.dumps({"type": "message_stop"}))
    )

    return chunks


def serialize_complete(reply: ReplyObject, model: str) -> dict[str, object]:
    """将 ReplyObject 序列化为完整的 Anthropic 响应 dict（非流式）。

    对应 TS: serializeComplete(reply, model)
    content 数组顺序: thinking → text → tool_use
    tool_use 的 input 直接是 dict（不是 JSON 字符串）
    """
    msg_id = gen_id("msg")
    usage = reply.usage or DEFAULT_USAGE

    content: list[dict[str, object]] = []

    # reasoning 块
    if reply.reasoning:
        content.append({"type": "thinking", "thinking": reply.reasoning})

    # text 块（空文本也输出，保持与 TS 一致
    content.append({"type": "text", "text": reply.text or ""})

    # tool_use 块
    if reply.tools:
        for i, tool in enumerate(reply.tools):
            call_id = tool_id({"id": tool.id}, "toolu", i)
            content.append(
                {
                    "type": "tool_use",
                    "id": call_id,
                    "name": tool.name,
                    "input": tool.args,
                },
            )

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": finish_reason(reply, "tool_use", "end_turn"),
        "stop_sequence": None,
        "usage": _build_usage(usage),
    }


def serialize_error(error: ErrorReply) -> dict[str, object]:
    """将 ErrorReply 序列化为 Anthropic 格式的错误响应。

    对应 TS: serializeError(error)。
    默认错误类型为 "api_error"。
    """
    return {
        "type": "error",
        "error": {
            "type": error.type or "api_error",
            "message": error.message,
        },
    }
