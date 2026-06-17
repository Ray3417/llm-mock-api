from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from .request import MockRequest


@dataclass(frozen=True, slots=True)
class ReplyObject:
    """
    结构化回复。text、reasoning、tool calls、usage 和 errors 均为可选。

    示例：
        server.when("hello").reply({"text": "Hi!", "reasoning": "Simple greeting."})
        server.when("weather").reply({"tools": [{"name": "get_weather", "args": {"city": "London"}}]})
    """

    text: str | None = field(default=None)
    """要发送回的文本内容。"""

    reasoning: str | None = field(default=None)
    """扩展思考或思维链。支持 Anthropic 和 Responses 格式。"""

    tools: Sequence[ToolCall] | None = field(default=None)
    """模型想要进行的工具调用。"""

    usage: Usage | None = field(default=None)
    """要报告的 token 计数。"""

    error: ErrorReply | None = field(default=None)
    """当设置时，服务器返回此 HTTP 错误而非正常回复。"""


@dataclass(frozen=True, slots=True)
class ErrorReply:
    """HTTP 错误响应。服务器返回此状态码及格式特定的响应体。"""

    status: int
    """HTTP 状态码，如 `429` 或 `500`。"""

    message: str
    """响应体中的错误消息。"""

    type: str | None = field(default=None)
    """响应体中的错误类型字符串。默认为格式特定值（如 OpenAI 为 "server_error"，Anthropic 为 "api_error"）。"""


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall:
    """回复中的工具调用。"""

    id: str | None = field(default=None)
    """调用的显式 ID。如省略则自动生成。"""

    name: str
    """工具函数名称。"""

    args: dict[str, Any] = field(default_factory=dict)
    """要传递给工具的参数。"""


@dataclass(frozen=True, slots=True)
class Usage:
    """token 用量统计。"""

    input: int
    """输入 token 数量。"""

    output: int
    """输出 token 数量。"""


@dataclass(frozen=True, slots=True)
class ReplyOptions:
    """每条规则的流式选项。会与服务器级默认值合并，规则级值优先。"""

    latency: int | None = field(default=None)
    """SSE chunk 之间的毫秒延迟。"""

    chunk_size: int | None = field(default=None)
    """文本分块大小（字符数），用于更真实的流式模拟。"""


@dataclass(frozen=True, slots=True)
class ReplySequenceEntryWithOptions:
    """带每步选项的回复条目。"""

    reply: Reply
    options: ReplyOptions | None = field(default=None)


Reply: TypeAlias = str | ReplyObject
"""回复可以是纯字符串（转为 `{"text": "..."}`）或完整的回复对象。"""


Resolver: TypeAlias = (
    Reply
    | Callable[[MockRequest], Reply | Awaitable[Reply]]
)
"""
回复值或生成回复的函数。支持异步函数。

示例：
    server.when("echo").reply(lambda req: f"You said: {req.last_message}")
    
    async def slow_reply(req: MockRequest) -> Reply:
        return {"text": "Done thinking."}
    server.when("slow").reply(slow_reply)
"""


SequenceEntry: TypeAlias = Reply | ReplySequenceEntryWithOptions
"""回复序列中的单个条目。"""
