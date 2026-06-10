"""
自定义处理器示例（processor.py）
================================

此文件演示如何通过 Python 代码实现动态规则匹配与回复。

与 JSON5 规则文件相比，处理器文件的优势在于：

  - 可以访问完整的请求上下文（请求头、模型名、对话历史等
  - match() 里可以写任意 Python 表达式
  - respond() 可以返回字符串、ReplyObject、error、usage、tool calls
  - 支持异步
  - 可以实现跨请求保持状态（如计数器、缓存等）
  - 可以返回 HTTP 错误（429/500/404 等

文件结构要求：
  - default = Handler 对象或 Handler 列表
  - 可选：fallback = "..."  （无规则匹配时的默认回复

每个 Handler 接口：
    class MyHandler:
        def match(self, req: MockRequest) -> bool: ...
        def respond(self, req: MockRequest) -> Reply: ...

运行方式：
    将此文件放在 rules 目录下，MockServer 会自动识别 .py 扩展名
    （与 .json5 文件一样被递归加载）

下面展示 5 个示例处理器：
    1. EchoHandler          — 回声演示：简单字符串回复
    2. TimeHandler         — 动态内容：返回当前时间
    3. RateLimitHandler     — 返回 HTTP 429 错误（演示 error 回复
    4. StructuredHandler    — 结构化回复：text + reasoning + tools
    5. AsyncHandler         — 异步回复：演示 async respond
    6. ModelAwareHandler    — 根据请求中的模型名条件响应
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from llm_mock_api.types.reply import (
    ErrorReply,
    Reply,
    ReplyObject,
    ToolCall,
    Usage,
)
from llm_mock_api.types.request import MockRequest


# ==========================================================================
# 1. 简单回声处理器：演示最基础的 Handler 结构
# ==========================================================================


class EchoHandler:
    """
    请求中包含 "echo" 时，把用户消息原样返回并加前缀。
    """

    def match(self, req: MockRequest) -> bool:
        return "echo" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        return f"Echo: {req.last_message}"


# ==========================================================================
# 2. 动态内容处理器：演示用 Python 生成内容
# ==========================================================================


class TimeHandler:
    """
    用户问时间时，返回当前时间（带思考过程的结构化回复。
    """

    def match(self, req: MockRequest) -> bool:
        msg = req.last_message.lower()
        return any(kw in msg for kw in ["时间", "几点", "date", "time"])

    def respond(self, req: MockRequest) -> Reply:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return ReplyObject(
            text=f"当前时间是 {now}。",
            reasoning=f"用户询问时间，调用 datetime.now() 获得 {now}。",
        )


# ==========================================================================
# 3. 错误回复处理器：演示返回 HTTP 错误
# ==========================================================================


class RateLimitHandler:
    """
    匹配 "rate_limit" → 返回 HTTP 429 Too Many Requests。
    用于测试客户端的错误处理逻辑。
    """

    def __init__(self) -> None:
        self._counter = 0

    def match(self, req: MockRequest) -> bool:
        return "rate_limit" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        self._counter += 1
        return ReplyObject(
            error=ErrorReply(
                status=429,
                message=f"Rate limit exceeded. Request #{self._counter}.",
                type="rate_limit_error",
            )
        )


class ServerErrorHandler:
    """
    匹配 "server_error" → 返回 HTTP 500。
    """

    def match(self, req: MockRequest) -> bool:
        return "server_error" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        return ReplyObject(
            error=ErrorReply(
                status=500,
                message="Something went wrong on our end.",
                type="server_error",
            )
        )


# ==========================================================================
# 4. 结构化回复 + 工具调用
# ==========================================================================


class WeatherToolHandler:
    """
    匹配 "weather" → 返回结构化回复带工具调用。
    """

    def match(self, req: MockRequest) -> bool:
        return "weather" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        return ReplyObject(
            text="我将调用天气工具。",
            reasoning="用户查询天气，调用 get_weather 工具。",
            tools=[
                ToolCall(
                    id="call_weather_001",
                    name="get_weather",
                    args={"location": "Beijing", "unit": "celsius"},
                )
            ],
        )


# ==========================================================================
# 5. 异步处理器：演示 respond 支持 async
# ==========================================================================


class AsyncHandler:
    """
    匹配 "slow" → 异步返回。
    """

    def match(self, req: MockRequest) -> bool:
        return "slow" in req.last_message.lower()

    async def respond(self, req: MockRequest) -> Reply:
        await asyncio.sleep(0.1)
        return "This reply was generated asynchronously."


# ==========================================================================
# 6. 模型感知处理器：根据请求中的模型名、请求头等条件响应
# ==========================================================================


class ModelAwareHandler:
    """
    演示访问请求中的模型名、请求格式等完整上下文。
    """

    def match(self, req: MockRequest) -> bool:
        return "model" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        return (
            f"收到来自 {req.format} 格式请求，模型={req.model}，"
            f"流式={req.streaming}，消息数={len(req.messages)}。"
        )


# ==========================================================================
# 7. Usage 报告处理器：演示返回 Usage 统计
# ==========================================================================


class UsageReportHandler:
    """
    匹配 "usage" → 返回带 Usage 的结构化回复。
    """

    def match(self, req: MockRequest) -> bool:
        return "usage" in req.last_message.lower()

    def respond(self, req: MockRequest) -> Reply:
        return ReplyObject(
            text="这是一条带 token 用量报告的回复。",
            usage=Usage(input=12, output=24),
        )


# ==========================================================================
# default 导出：所有处理器按顺序匹配
# ==========================================================================

default = [
    EchoHandler(),
    TimeHandler(),
    RateLimitHandler(),
    ServerErrorHandler(),
    WeatherToolHandler(),
    AsyncHandler(),
    ModelAwareHandler(),
    UsageReportHandler(),
]

# 可选：为 processor.py 自定义 fallback（与 JSON5 fallback 二选一）
# 当请求未匹配时返回此内容。如果不需要可设为 None 或删除此行。
fallback = "（processor.py 兜底回复：未找到匹配规则。请检查您的请求内容。）"
