"""
OpenAI 响应序列化。

reply (ReplyObject)
    ├─ serialize() → SSEChunk 列表（流式）
    │   [role] → [text chunks] → [tool chunks] → [finish+usage] → [DONE]
    ├─ serialize_complete() → dict（非流式完整 JSON）
    └─ serialize_error() → { error: {...} }
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

from ...serialize_helpers import (
    DEFAULT_USAGE,
    finish_reason,
    gen_id,
    split_text,
    tool_id,
)
from ...types import SSEChunk
from ....types.reply import ErrorReply, ReplyObject, ReplyOptions, Usage


# ── usage 字段构造 ───────────────────────────────────────

def _build_usage(usage: Usage) -> dict[str, object]:
    """将内部 Usage 转换为 OpenAI 格式的 usage dict。"""
    return {
        "prompt_tokens": usage.input,
        "completion_tokens": usage.output,
        "total_tokens": usage.input + usage.output,
        "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": 0,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


# ── SSE chunk 封装 ──────────────────────────────────────

def _chunk_envelope(
    event_id: str,
    created: int,
    model: str,
    delta: dict[str, object],
    finish_reason_val: str | None = None,
    usage: dict[str, object] | None = None,
) -> SSEChunk:
    """将 delta/usage 等内容封装为 SSEChunk。

    对应 TS: chunkEnvelope(id, created, model, delta, finish_reason, usage)。
    参数名 event_id 避免遮蔽 Python 内置 id。
    """
    return SSEChunk(
        data=json.dumps(
            {
                "id": event_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "system_fingerprint": None,
                "service_tier": "default",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "logprobs": None,
                        "finish_reason": finish_reason_val,
                    }
                ],
                "usage": usage,
            }
        )
    )


# ── 流式序列化 ───────────────────────────────────────────

def serialize(
    reply: ReplyObject,
    model: str,
    options: ReplyOptions | None = None,
) -> Sequence[SSEChunk]:
    """将 ReplyObject 序列化为 SSEChunk 列表。

    chunk 顺序严格对应 TS：
      1. { role: "assistant" }  — 声明角色
      2. [text chunk 1, ...]   — 文本分片
      3. [tool chunk 1, ...]   — 工具调用
      4. { finish_reason }      — 结束标志
      5. { usage }              — token 统计
      6. [DONE]                 — 流结束
    """
    event_id = gen_id("chatcmpl")
    created = int(time.time())  # 对应 TS: Math.floor(Date.now() / 1000)
    usage = reply.usage or DEFAULT_USAGE
    opts = options or ReplyOptions()

    chunks: list[SSEChunk] = []

    # 1. 声明角色
    chunks.append(_chunk_envelope(event_id, created, model, {"role": "assistant"}))

    # 2. 文本分片（仅当有 text 时）
    if reply.text:
        for piece in split_text(reply.text, opts.chunk_size or 0):
            chunks.append(_chunk_envelope(event_id, created, model, {"content": piece}))

    # 3. 工具调用分片（仅当有 tools 时）
    if reply.tools:
        for i, tool in enumerate(reply.tools):
            chunks.append(
                _chunk_envelope(
                    event_id,
                    created,
                    model,
                    {
                        "tool_calls": [
                            {
                                "index": i,
                                "id": tool_id({"id": tool.id}, "call", i),
                                "type": "function",
                                "function": {
                                    "name": tool.name,
                                    "arguments": json.dumps(tool.args),
                                },
                            }
                        ]
                    },
                )
            )

    # 4. finish_reason chunk
    chunks.append(
        _chunk_envelope(
            event_id,
            created,
            model,
            {},
            finish_reason(reply, "tool_calls", "stop"),
        )
    )

    # 5. usage chunk
    chunks.append(_chunk_envelope(event_id, created, model, {}, None, _build_usage(usage)))

    # 6. [DONE] 结束标志
    chunks.append(SSEChunk(data="[DONE]"))

    return chunks


# ── 非流式序列化 ─────────────────────────────────────────

def serialize_complete(reply: ReplyObject, model: str) -> dict[str, object]:
    """将 ReplyObject 序列化为完整的 OpenAI 响应 dict。

    对应 TS serializeComplete。
    """
    event_id = gen_id("chatcmpl")
    created = int(time.time())  # 对应 TS: Math.floor(Date.now() / 1000)
    usage = reply.usage or DEFAULT_USAGE

    message: dict[str, object] = {
        "role": "assistant",
        "content": reply.text if reply.text is not None else None,
    }
    if reply.tools:
        message["tool_calls"] = [
            {
                "id": tool_id({"id": tool.id}, "call", i),
                "type": "function",
                "function": {
                    "name": tool.name,
                    "arguments": json.dumps(tool.args),
                },
            }
            for i, tool in enumerate(reply.tools)
        ]

    return {
        "id": event_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "service_tier": "default",
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason(reply, "tool_calls", "stop"),
            }
        ],
        "usage": _build_usage(usage),
    }


# ── 错误序列化 ───────────────────────────────────────────

def serialize_error(error: ErrorReply) -> dict[str, object]:
    """将 ErrorReply 序列化为 OpenAI 格式的错误响应。"""
    return {
        "error": {
            "message": error.message,
            "type": error.type or "server_error",
            "code": None,
        }
    }
