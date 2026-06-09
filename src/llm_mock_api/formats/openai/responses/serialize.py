"""
OpenAI Responses 格式 - 响应序列化。

reply（ReplyObject）
    ├─ serialize() → SSEChunk 列表（流式）
    │   header(2: response.created + response.in_progress)
    │   ├─ reasoning? → 6 个 chunk（output_item.added → part → delta → done）
    │   ├─ text? → 7+ 个 chunk（同上，带 split_text 分块）
    │   ├─ tools → 每个 4 个 chunk（added → args delta → done → item done）
    │   └─ footer: response.completed（含 usage）
    ├─ serialize_complete() → dict（非流式完整 JSON）
    └─ serialize_error() → { error: {...} }

每个 output_item 都关联一组 SSE chunk（StreamBlock 抽象），所有 chunk 都带
递增的 sequence_number（通过 _create_chunk 闭包实现）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import NamedTuple

from ...serialize_helpers import (
    DEFAULT_USAGE,
    gen_id,
    should_emit_text,
    split_text,
    tool_id,
)
from ...types import SSEChunk
from ....types.reply import ErrorReply, ReplyObject, ReplyOptions, ToolCall, Usage


# ── 类型别名 ────────────────────────────────────────────────

type ChunkFn = Callable[[dict[str, object]], SSEChunk]


class _StreamBlock(NamedTuple):
    """output_item 与其对应的一组 SSE chunk。"""

    chunks: list[SSEChunk]
    output_item: dict[str, object]


# ── 辅助函数 ────────────────────────────────────────────────

def _build_usage(usage: Usage) -> dict[str, object]:
    """将内部 Usage 转换为 Responses 格式的 usage dict。"""
    return {
        "input_tokens": usage.input,
        "output_tokens": usage.output,
        "total_tokens": usage.input + usage.output,
    }


def _create_chunk() -> ChunkFn:
    """创建一个 chunk 构造器，每个返回的 chunk 都带自动递增的 sequence_number。"""
    seq = 0

    def chunk(payload: dict[str, object]) -> SSEChunk:
        nonlocal seq
        result: dict[str, object] = dict(payload)
        result["sequence_number"] = seq
        seq += 1
        return SSEChunk(data=json.dumps(result))

    return chunk


# ── 三种 StreamBlock 构造器 ────────────────────────────────

def _reasoning_stream_block(c: ChunkFn, i: int, reasoning: str) -> _StreamBlock:
    """构造 reasoning 类型的 output_item 及其 6 个 chunk。"""
    item_id = f"rs_{gen_id('rs')}"
    summary_part = {"type": "summary_text", "text": reasoning}
    output_item: dict[str, object] = {
        "type": "reasoning",
        "id": item_id,
        "status": "completed",
        "summary": [summary_part],
    }

    chunks: list[SSEChunk] = [
        c({
            "type": "response.output_item.added",
            "output_index": i,
            "item": {
                "type": "reasoning",
                "id": item_id,
                "status": "in_progress",
                "summary": [],
            },
        }),
        c({
            "type": "response.reasoning_summary_part.added",
            "item_id": item_id,
            "output_index": i,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        }),
        c({
            "type": "response.reasoning_summary_text.delta",
            "item_id": item_id,
            "output_index": i,
            "summary_index": 0,
            "delta": reasoning,
        }),
        c({
            "type": "response.reasoning_summary_text.done",
            "item_id": item_id,
            "output_index": i,
            "summary_index": 0,
            "text": reasoning,
        }),
        c({
            "type": "response.reasoning_summary_part.done",
            "item_id": item_id,
            "output_index": i,
            "summary_index": 0,
            "part": summary_part,
        }),
        c({"type": "response.output_item.done", "output_index": i, "item": output_item}),
    ]

    return _StreamBlock(chunks=chunks, output_item=output_item)


def _text_stream_block(c: ChunkFn, i: int, text: str, chunk_size: int) -> _StreamBlock:
    """构造 message 类型的 output_item 及其 7+ 个 chunk（按 chunk_size 切分文本）。"""
    item_id = f"msg_{gen_id('msg')}"
    output_text: dict[str, object] = {
        "type": "output_text",
        "text": text,
        "annotations": [],
    }
    output_item: dict[str, object] = {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [output_text],
    }

    chunks: list[SSEChunk] = [
        c({
            "type": "response.output_item.added",
            "output_index": i,
            "item": {
                "type": "message",
                "id": item_id,
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            },
        }),
        c({
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": i,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }),
    ]

    # 文本 delta（可能多个）
    for piece in split_text(text, chunk_size):
        chunks.append(c({
            "type": "response.output_text.delta",
            "item_id": item_id,
            "output_index": i,
            "content_index": 0,
            "delta": piece,
        }))

    chunks.extend([
        c({
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": i,
            "content_index": 0,
            "text": text,
        }),
        c({
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": i,
            "content_index": 0,
            "part": output_text,
        }),
        c({"type": "response.output_item.done", "output_index": i, "item": output_item}),
    ])

    return _StreamBlock(chunks=chunks, output_item=output_item)


def _tool_stream_block(c: ChunkFn, i: int, tool: ToolCall) -> _StreamBlock:
    """构造 function_call 类型的 output_item 及其 4 个 chunk。"""
    call_id = tool_id({"id": tool.id}, "call", i)
    args_json = json.dumps(tool.args)
    output_item: dict[str, object] = {
        "type": "function_call",
        "id": call_id,
        "status": "completed",
        "name": tool.name,
        "call_id": call_id,
        "arguments": args_json,
    }

    chunks: list[SSEChunk] = [
        c({
            "type": "response.output_item.added",
            "output_index": i,
            "item": {**output_item, "status": "in_progress", "arguments": ""},
        }),
        c({
            "type": "response.function_call_arguments.delta",
            "item_id": call_id,
            "output_index": i,
            "delta": args_json,
        }),
        c({
            "type": "response.function_call_arguments.done",
            "item_id": call_id,
            "output_index": i,
            "arguments": args_json,
        }),
        c({"type": "response.output_item.done", "output_index": i, "item": output_item}),
    ]

    return _StreamBlock(chunks=chunks, output_item=output_item)


# ── 对外函数 ────────────────────────────────────────────────

def serialize(
    reply: ReplyObject,
    model: str,
    options: ReplyOptions | None = None,
) -> Sequence[SSEChunk]:
    """构造流式响应的 SSEChunk 列表。

    header(2) → reasoning → text → tools → completed(1)，全部 chunk 共享同一
    sequence_number 计数器。
    """
    event_id = gen_id("resp")
    created = int(time.time())
    usage = reply.usage or DEFAULT_USAGE
    opts = options or ReplyOptions()
    c = _create_chunk()
    i = 0

    base_response: dict[str, object] = {
        "id": event_id,
        "object": "response",
        "created_at": created,
        "model": model,
    }

    # header：response.created + response.in_progress
    all_chunks: list[SSEChunk] = [
        c({"type": "response.created", "response": {
            **base_response, "status": "in_progress", "output": []}}),
        c({"type": "response.in_progress", "response": {
            **base_response, "status": "in_progress", "output": []}}),
    ]

    # StreamBlocks：reasoning? → text? → tools
    blocks: list[_StreamBlock] = []
    if reply.reasoning:
        blocks.append(_reasoning_stream_block(c, i, reply.reasoning))
        i += 1
    if should_emit_text(reply):
        blocks.append(_text_stream_block(c, i, reply.text or "", opts.chunk_size or 0))
        i += 1
    for tool in reply.tools or []:
        blocks.append(_tool_stream_block(c, i, tool))
        i += 1

    # 收集所有 chunks 和 output_items
    for block in blocks:
        all_chunks.extend(block.chunks)
    output_items = [block.output_item for block in blocks]

    # footer：response.completed
    all_chunks.append(c({
        "type": "response.completed",
        "response": {
            **base_response,
            "status": "completed",
            "output": output_items,
            "usage": _build_usage(usage),
        },
    }))

    return all_chunks


def serialize_complete(reply: ReplyObject, model: str) -> dict[str, object]:
    """构造非流式完整响应（单一 JSON 对象）。"""
    event_id = gen_id("resp")
    created = int(time.time())
    usage = reply.usage or DEFAULT_USAGE

    output_items: list[dict[str, object]] = []

    if reply.reasoning:
        output_items.append({
            "type": "reasoning",
            "id": f"rs_{gen_id('rs')}",
            "status": "completed",
            "summary": [{"type": "summary_text", "text": reply.reasoning}],
        })

    if should_emit_text(reply):
        output_items.append({
            "type": "message",
            "id": f"msg_{gen_id('msg')}",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": reply.text if reply.text is not None else "",
                "annotations": [],
            }],
        })

    for idx, tool in enumerate(reply.tools or []):
        call_id = tool_id({"id": tool.id}, "call", idx)
        output_items.append({
            "type": "function_call",
            "id": call_id,
            "status": "completed",
            "name": tool.name,
            "call_id": call_id,
            "arguments": json.dumps(tool.args),
        })

    return {
        "id": event_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": _build_usage(usage),
    }


def serialize_error(error: ErrorReply) -> dict[str, object]:
    """构造错误响应。"""
    error_type = error.type or "server_error"
    return {
        "type": "error",
        "error": {
            "message": error.message,
            "type": error_type,
            "code": error_type,
        },
    }
