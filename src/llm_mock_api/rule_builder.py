"""
规则构建器：提供流畅 API 向 RuleEngine 注册匹配规则。

用户调用 builder.when("hello").reply("Hi!")
    ↓
_PendingRule 被创建（捕获 match 和 engine 引用）
    ↓
调用 .reply("Hi!")
    ├─ engine.add("hello", "Hi!") → 创建 Rule 对象
    └─ 返回 _RuleHandle 包装该规则
    ↓
可选链式调用 .times(3).first()
    ├─ .times(3) → 修改 rule.remaining = 3
    └─ .first() → 调用 engine.move_to_front(rule)

请求到达时 engine.match(req) 遍历所有已注册规则，返回第一条匹配的 Rule。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from .types.reply import ErrorReply, ReplyObject, ReplyOptions, Resolver, SequenceEntry, ReplySequenceEntryWithOptions
from .types.rule import Match, MatchObject, PendingRule, Rule, RuleHandle
from .rule_engine import RuleEngine, _SequenceStep, create_sequence_resolver


class _RuleHandle:
    """已注册规则的句柄，支持 .times(n).first() 链式配置。"""

    def __init__(self, engine: RuleEngine, rule: Rule) -> None:
        self._engine = engine
        self._rule = rule

    def times(self, n: int) -> RuleHandle:
        """限制规则仅匹配 n 次后自动移除。"""
        self._rule.remaining = n
        return self

    def first(self) -> RuleHandle:
        """将此规则移到列表最前面，优先匹配。"""
        self._engine.move_to_front(self._rule)
        return self


class _PendingRule:
    """由 when() 返回的未完成规则，等待通过 reply() 或 reply_sequence() 完成。"""

    def __init__(self, engine: RuleEngine, match: Match) -> None:
        self._engine = engine
        self._match = match

    def reply(self, response: Resolver, options: ReplyOptions | None = None) -> RuleHandle:
        """设置回复内容并注册规则。"""
        rule = self._engine.add(self._match, response, options)
        return _RuleHandle(self._engine, rule)

    def reply_sequence(self, entries: Sequence[SequenceEntry]) -> RuleHandle:
        """按顺序回复每个条目，遍历到末尾后持续返回最后一条。"""
        steps = normalise_sequence_entries(entries)
        # 先注册一个空回复占位，随后修改 resolve 和 remaining
        rule = self._engine.add(self._match, "")
        result = create_sequence_resolver(steps, rule)
        rule.resolve = result.resolver
        rule.remaining = result.entry_count
        return _RuleHandle(self._engine, rule)


def normalise_sequence_entries(
    entries: Sequence[SequenceEntry],
) -> list[_SequenceStep]:
    """将 SequenceEntry 数组统一转换为带 reply+options 的步骤列表。"""
    result: list[_SequenceStep] = []
    for entry in entries:
        if isinstance(entry, ReplySequenceEntryWithOptions):
            result.append(_SequenceStep(reply=entry.reply, options=entry.options))
        else:
            result.append(_SequenceStep(reply=entry))
    return result


class RuleBuilder:
    """规则构建器。通过流畅的 API 向 RuleEngine 注册规则。"""

    def __init__(self, engine: RuleEngine) -> None:
        self._engine: Final = engine

    def when(self, match: Match) -> PendingRule:
        """注册匹配条件。在返回对象上调用 .reply() 完成规则。"""
        return _PendingRule(self._engine, match)

    def when_tool(self, tool_name: str) -> PendingRule:
        """快捷方式：匹配特定工具名。"""
        return self.when(MatchObject(tool_name=tool_name))

    def when_tool_result(self, tool_call_id: str) -> PendingRule:
        """快捷方式：匹配特定 tool_call_id。"""
        return self.when(MatchObject(tool_call_id=tool_call_id))

    def next_error(self, status: int, message: str, type_: str | None = None) -> RuleHandle:
        """为下一次请求注册一个一次性错误响应，匹配后自动移除。"""
        return (
            self.when(lambda req: True)
            .reply(ReplyObject(error=ErrorReply(status=status, message=message, type=type_)))
            .times(1)
            .first()
        )
