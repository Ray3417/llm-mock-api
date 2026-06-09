"""工作流示例：
1. route_handler 调用 format.serialize(reply, model, options) 得到 chunks: list[SSEChunk]
2. write_sse(chunks, options) 创建 StreamingResponse（对应 TS 的 writeSSE）
3. FastAPI 返回该响应，客户端按 SSE 协议接收流
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Iterable

from fastapi.responses import StreamingResponse

from .formats.types import SSEChunk
from .types.reply import ReplyOptions


def format_sse_chunk(chunk: SSEChunk) -> str:
    """将单个 SSEChunk 格式化为 SSE 协议字符串。

    输出示例：
        event: message
        data: {"content":"Hello"}

        data: [DONE]

    """
    event_line = f"event: {chunk.event}\n" if chunk.event else ""
    return f"{event_line}data: {chunk.data}\n\n"


async def _sse_stream(
    chunks: Iterable[SSEChunk],
    latency_ms: int,
) -> AsyncGenerator[str, None]:
    """逐 chunk 生成 SSE 字符串，chunk 之间可选延迟。"""
    for chunk in chunks:
        yield format_sse_chunk(chunk)
        if latency_ms > 0:
            await asyncio.sleep(latency_ms / 1000)


def write_sse(
    chunks: Iterable[SSEChunk],
    options: ReplyOptions | None = None,
) -> StreamingResponse:
    """写入 SSE 流式响应（对应 TS 的 `writeSSE(reply, chunks, options)`）。

    与 TS 不同的是 Python/FastAPI 风格：返回响应对象而非写入参数。

    参数：
        chunks: 要流式传输的 SSEChunk 可迭代对象
        options: 含 latency（chunk 间延迟毫秒数）的选项

    返回：
        带有正确 SSE 头的 StreamingResponse
    """
    latency = options.latency if options and options.latency is not None else 0

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        _sse_stream(chunks, latency),
        media_type="text/event-stream",
        headers=headers,
    )
