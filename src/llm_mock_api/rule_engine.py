"""

规则引擎：匹配请求并执行对应的 mock 规则。

用户调用 engine.add("hello", "Hi!")
    ↓
_create_rule 被调用
    ├─ _describe_match("hello") → '"hello"' （规则描述）
    └─ _compile_matcher("hello") → 返回一个函数：(req) -> "hello" 是否在 req.last_message 里
    ↓
组装成 Rule 对象，塞入 RuleEngine 的 rules 列表
    ↓
┌──────────────────────────────────────────────────┐
│  后来有请求来了，engine.match(req)                 │
│  遍历 rules 列表，每条规则调用 rule.match(req)      │
│  找到第一条返回 True 的规则                         │
│  rule.remaining -= 1 （消耗一次使用次数）           │
│  返回这条 rule，让调用方决定怎么回复                  │
└──────────────────────────────────────────────────┘

"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields as dataclass_fields
from typing import NamedTuple

from .types.reply import Reply, ReplyOptions, Resolver
from .types.request import MockRequest
from .types.rule import Match, MatchObject, Rule, RuleSummary


# Python 的 re.Pattern 不会在匹配间保留状态，无需对应 TS 的 safeRegex 函数
# （TS 版本需要它是因为 RegExp.prototype.test() 会受 global/sticky 标志的
# lastIndex 属性影响，Python 无此问题）


def _compile_pattern(pattern: str | re.Pattern[str]) -> Callable[[str], bool]:
    """将字符串或正则编译为值匹配函数。

    - 字符串：大小写不敏感的包含匹配
    - 正则：直接执行正则匹配
    """
    if isinstance(pattern, str):
        lower = pattern.lower()
        return lambda value: lower in value.lower()
    return lambda value: pattern.search(value) is not None


def _compile_matcher(match: Match) -> Callable[[MockRequest], bool]:
    """将任意 Match 类型编译为请求匹配函数。"""
    if isinstance(match, str):
        test = _compile_pattern(match)
        return lambda req: test(req.last_message)
    if isinstance(match, re.Pattern):
        test = _compile_pattern(match)
        return lambda req: test(req.last_message)
    if callable(match):
        return match
    # 以上三个分支排除后，剩余类型为 MatchObject
    obj: MatchObject = match
    message_test = _compile_pattern(obj.message) if obj.message is not None else None
    model_test = _compile_pattern(obj.model) if obj.model is not None else None
    system_test = _compile_pattern(obj.system) if obj.system is not None else None

    def matcher(req: MockRequest) -> bool:
        if message_test and not message_test(req.last_message):
            return False
        if model_test and not model_test(req.model):
            return False
        if system_test and not system_test(req.system_message):
            return False
        if obj.format is not None and req.format != obj.format:
            return False
        if obj.tool_name is not None and obj.tool_name not in req.tool_names:
            return False
        if obj.tool_call_id is not None and req.last_tool_call_id != obj.tool_call_id:
            return False
        if obj.predicate is not None and not obj.predicate(req):
            return False
        return True

    return matcher


def _describe_match(match: Match) -> str:
    """生成 Match 的人类可读描述，用于日志和规则列表。"""
    if isinstance(match, str):
        return f'"{match}"'
    if isinstance(match, re.Pattern):
        return f"/{match.pattern}/"
    if callable(match):
        return "(predicate)"
    # 对应 TS: Object.entries(obj) 动态遍历所有字段
    obj: MatchObject = match
    parts: list[str] = []
    for field in dataclass_fields(MatchObject):
        value = getattr(obj, field.name)
        if value is None:
            continue
        # 对应 TS: typeof v !== "function"
        if callable(value):
            parts.append(f"{field.name}=<fn>")
        else:
            parts.append(f"{field.name}={value}")
    return "{" + ", ".join(parts) + "}"


def _create_rule(
    match: Match,
    resolve: Resolver,
    options: ReplyOptions,
    description: str | None = None,
) -> Rule:
    """创建规则对象。"""
    return Rule(
        description=description if description is not None else _describe_match(match),
        match=_compile_matcher(match),
        resolve=resolve,
        options=options,
        remaining=float("inf"),
    )


# ============================================================
# 序列解析器
# ============================================================


@dataclass(slots=True)
class _SequenceStep:
    """内部使用：回复序列中的一步。"""

    reply: Reply
    options: ReplyOptions | None = None


class _SequenceResolverResult(NamedTuple):
    """序列解析器的返回值：resolver 是回复函数，entry_count 是步数。"""
    resolver: Callable[[MockRequest], Reply]
    entry_count: int


def create_sequence_resolver(
    steps: Sequence[_SequenceStep],
    rule: Rule,
) -> _SequenceResolverResult:
    """创建一个按顺序返回每条回复的解析器。

    遍历完所有步骤后，最后一次回复将被无限重复。
    每次调用会同步更新 rule.options 为当前步骤的选项。"""
    if len(steps) == 0:
        raise ValueError("Sequence requires at least one entry.")
    index: list[int] = [0]  # 用 list 作为可变容器
    last = steps[-1]

    def resolver(_req: MockRequest | None = None) -> Reply:
        step = steps[index[0]] if index[0] < len(steps) else last
        index[0] += 1
        rule.options = step.options if step.options is not None else ReplyOptions()
        return step.reply

    return _SequenceResolverResult(resolver=resolver, entry_count=len(steps))


# ============================================================
# 规则引擎
# ============================================================


class RuleEngine:
    """管理 mock 规则列表，并将传入请求与第一条匹配规则关联。"""

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def add(
        self,
        match: Match,
        resolve: Resolver,
        options: ReplyOptions | None = None,
    ) -> Rule:
        """注册一条新规则并返回其句柄。"""
        rule = _create_rule(
            match,
            resolve,
            options if options is not None else ReplyOptions(),
        )
        self._rules.append(rule)
        return rule

    def move_to_front(self, rule: Rule) -> None:
        """将指定规则移到列表最前面，提高其匹配优先级。"""
        idx = self._rules.index(rule)
        if idx > 0:
            self._rules.pop(idx)
            self._rules.insert(0, rule)

    def match(self, req: MockRequest) -> Rule | None:
        """按顺序查找匹配请求的第一条规则。

        匹配后规则的 remaining 递减，若降至 0 则从列表中移除。"""
        for i in range(len(self._rules)):
            rule = self._rules[i]
            if rule.remaining <= 0:
                continue
            if not rule.match(req):
                continue
            rule.remaining -= 1
            if rule.remaining <= 0:
                self._rules.pop(i)
            return rule
        return None

    def is_done(self) -> bool:
        """是否没有待耗尽的有限次规则。"""
        return all(
            rule.remaining == float("inf") or rule.remaining <= 0
            for rule in self._rules
        )

    @property
    def rule_count(self) -> int:
        """当前注册的规则数量。"""
        return len(self._rules)

    def describe(self) -> list[RuleSummary]:
        """所有规则的描述摘要列表。"""
        return [
            RuleSummary(description=rule.description, remaining=rule.remaining)
            for rule in self._rules
        ]

    def clear(self) -> None:
        """移除所有已注册的规则。"""
        self._rules.clear()
