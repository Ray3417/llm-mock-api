from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .request import FormatName, MockRequest
from .reply import Reply, ReplyOptions, Resolver, SequenceEntry


type Match = str | re.Pattern[str] | MatchObject | Callable[[MockRequest], bool]
"""
确定规则是否匹配传入请求。

`string` 对最后一条用户消息进行大小写不敏感的字串匹配。
`re.Pattern` 对最后一条用户消息进行正则匹配。
`MatchObject` 以 AND 逻辑同时检查多个字段。
函数接收规范化请求并返回布尔值。

示例：
    server.when("hello").reply("Hi!")
    server.when(re.compile(r"explain (\w+)", re.IGNORECASE)).reply("Here's an explanation.")
    server.when({"model": "claude", "format": "anthropic"}).reply("Bonjour!")
    server.when(lambda req: len(req.messages) > 5).reply("Long conversation!")
"""


@dataclass(frozen=True, slots=True)
class MatchObject:
    """
    结构化匹配器。设置的每个字段都必须匹配规则才会触发。

    示例：
        server.when({
            "model": "/gpt/",
            "format": "openai",
            "system": "/translator/i",
            "predicate": lambda req: len(req.messages) > 2,
        }).reply("Translated output.")
    """

    message: str | re.Pattern[str] | None = field(default=None)
    """对最后一条用户消息的字串或正则匹配。"""

    model: str | re.Pattern[str] | None = field(default=None)
    """对模型名称的字串或正则匹配。"""

    system: str | re.Pattern[str] | None = field(default=None)
    """对系统提示的字串或正则匹配。"""

    format: FormatName | None = field(default=None)
    """仅匹配来自此 API 格式的请求。"""

    tool_name: str | None = field(default=None)
    """当请求包含此名称的工具定义时匹配。"""

    tool_call_id: str | None = field(default=None)
    """当最后一条工具结果消息具有此 `tool_call_id` 时匹配。"""

    predicate: Callable[[MockRequest], bool] | None = field(default=None)
    """在所有结构化字段通过后最后执行的额外谓词。"""


@runtime_checkable
class PendingRule(Protocol):
    """
    由 `when()` 返回。在此上调用 `.reply()` 或 `.reply_sequence()` 完成规则。

    示例：
        server.when("hello").reply("Hi!")
        server.when("step").reply_sequence(["First.", "Second.", "Done."])
    """

    def reply(self, response: Resolver, options: ReplyOptions | None = None) -> RuleHandle:
        """设置此规则的回复。接受静态值、对象或解析器函数。"""
        ...

    def reply_sequence(self, entries: Sequence[SequenceEntry]) -> RuleHandle:
        """设置回复序列。每次匹配推进数组。"""
        ...


@runtime_checkable
class RuleHandle(Protocol):
    """
    已注册规则的句柄。所有方法返回 `self` 以支持链式调用。

    示例：
        server.when("hello").reply("Hi!").times(3)
        server.when("urgent").reply("On it!").first()
    """

    def times(self, n: int) -> RuleHandle:
        """规则在 `n` 次匹配后自动过期。"""
        ...

    def first(self) -> RuleHandle:
        """将此规则移到列表前面，使其优先匹配。"""
        ...


@dataclass(frozen=True, slots=True)
class RuleSummary:
    """已注册规则的摘要，用于通过 `server.rules` 检查。"""

    description: str
    """规则匹配内容的人类可读描述。"""

    remaining: float
    """剩余匹配次数。默认为 `float('inf')`（无限）。"""


@runtime_checkable
class Handler(Protocol):
    """
    处理器文件的默认导出结构。可以导出单个处理器或处理器数组。

    示例：
        from llm_mock_api.types.rule import Handler
        from llm_mock_api.types.request import MockRequest

        class MyHandler:
            def match(self, req: MockRequest) -> bool:
                return "echo" in req.last_message

            def respond(self, req: MockRequest) -> Reply:
                return f"Echo: {req.last_message}"

        handler: Handler = MyHandler()
    """

    def match(self, req: MockRequest) -> bool:
        """返回 `true` 如果此处理器应响应该请求。"""
        ...

    def respond(self, req: MockRequest) -> Reply | Awaitable[Reply]:
        """为匹配的请求生成回复。可以是异步的。"""
        ...


@dataclass(slots=True)
class Rule:
    description: str
    match: Callable[[MockRequest], bool]
    resolve: Resolver
    options: ReplyOptions
    remaining: float
