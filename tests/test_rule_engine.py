"""rule_engine 模块单元测试。"""

from __future__ import annotations

import re

from llm_mock_api.rule_engine import RuleEngine, create_sequence_resolver
from llm_mock_api.types.reply import ReplyOptions
from llm_mock_api.types.request import MockRequest
from llm_mock_api.types.rule import MatchObject


def _req(last_message: str = "", model: str = "gpt-4", tool_names: tuple[str, ...] = (),
         last_tool_call_id: str | None = None, system_message: str = "",
         format: str = "openai") -> MockRequest:
    """创建一个简化的 MockRequest 用于测试。"""
    return MockRequest(
        format=format,
        model=model,
        streaming=False,
        messages=(),
        last_message=last_message,
        system_message=system_message,
        tool_names=tool_names,
        last_tool_call_id=last_tool_call_id,
    )


class TestBasicMatching:
    """测试基本的规则添加和匹配。"""

    def test_string_match(self) -> None:
        """字符串匹配：不区分大小写的包含匹配。"""
        engine = RuleEngine()
        engine.add("hello", "Hi!")
        rule = engine.match(_req(last_message="hello world"))
        assert rule is not None
        assert rule.resolve == "Hi!"

    def test_regex_match(self) -> None:
        """正则匹配。"""
        engine = RuleEngine()
        engine.add(re.compile(r"^\d+$"), "number")
        rule = engine.match(_req(last_message="123"))
        assert rule is not None
        assert rule.resolve == "number"

    def test_predicate_match(self) -> None:
        """可调用匹配函数。"""
        engine = RuleEngine()
        engine.add(lambda r: "special" in r.last_message, "matched")
        rule = engine.match(_req(last_message="this is special text"))
        assert rule is not None
        assert rule.resolve == "matched"

    def test_no_match_returns_none(self) -> None:
        """无匹配规则时返回 None。"""
        engine = RuleEngine()
        engine.add("hello", "Hi!")
        assert engine.match(_req(last_message="goodbye")) is None


class TestMatchObject:
    """测试 MatchObject 多字段匹配。"""

    def test_tool_name_match(self) -> None:
        """匹配 tool_name 字段。"""
        engine = RuleEngine()
        engine.add(MatchObject(tool_name="my_tool"), "tool_reply")
        rule = engine.match(_req(last_message="", tool_names=("my_tool",)))
        assert rule is not None
        assert rule.resolve == "tool_reply"

    def test_tool_call_id_match(self) -> None:
        """匹配 tool_call_id 字段。"""
        engine = RuleEngine()
        engine.add(MatchObject(tool_call_id="call_001"), "result_reply")
        rule = engine.match(_req(last_message="", last_tool_call_id="call_001"))
        assert rule is not None

    def test_model_and_message(self) -> None:
        """多字段联合匹配。"""
        engine = RuleEngine()
        engine.add(MatchObject(message="hi", model="gpt-4"), "reply")
        rule = engine.match(_req(last_message="hi there", model="gpt-4"))
        assert rule is not None
        assert engine.match(_req(last_message="hi", model="gpt-3")) is None


class TestRemaining:
    """测试规则匹配次数消耗。"""

    def test_times_limits_matches(self) -> None:
        """times(n) 应限制规则仅匹配 n 次。"""
        engine = RuleEngine()
        rule = engine.add("bye", "Goodbye")
        rule.remaining = 2
        r1 = engine.match(_req(last_message="bye"))
        r2 = engine.match(_req(last_message="bye"))
        r3 = engine.match(_req(last_message="bye"))
        assert r1 is not None and r2 is not None
        assert r3 is None

    def test_infinite_remaining_default(self) -> None:
        """默认次数是无限的。"""
        engine = RuleEngine()
        engine.add("always", "here")
        for _ in range(10):
            assert engine.match(_req(last_message="always")) is not None


class TestPriority:
    """测试规则优先级（move_to_front）。"""

    def test_move_to_front_changes_priority(self) -> None:
        """move_to_front 应将指定规则移到最前面优先匹配。"""
        engine = RuleEngine()
        low = engine.add("key", "low")
        high = engine.add("key", "high")
        engine.move_to_front(high)
        rule = engine.match(_req(last_message="key"))
        assert rule.resolve == "high"


class TestEngineState:
    """测试引擎状态查询和管理。"""

    def test_rule_count(self) -> None:
        """rule_count 应反映当前规则数。"""
        engine = RuleEngine()
        assert engine.rule_count == 0
        engine.add("a", "1")
        engine.add("b", "2")
        assert engine.rule_count == 2

    def test_is_done_after_consume(self) -> None:
        """所有有限次规则耗尽后 is_done 为 True。"""
        engine = RuleEngine()
        rule = engine.add("once", "only")
        rule.remaining = 1
        engine.match(_req(last_message="once"))
        assert engine.is_done()

    def test_describe_returns_summaries(self) -> None:
        """describe 应返回每条规则的摘要。"""
        engine = RuleEngine()
        engine.add("hello", "Hi")
        summaries = engine.describe()
        assert len(summaries) == 1
        assert "hello" in summaries[0].description

    def test_clear_removes_all(self) -> None:
        """clear 应清空所有规则。"""
        engine = RuleEngine()
        engine.add("a", "1")
        engine.add("b", "2")
        engine.clear()
        assert engine.rule_count == 0


class TestSequenceResolver:
    """测试序列解析器。"""

    def test_returns_in_order(self) -> None:
        """按顺序返回每条回复。"""
        engine = RuleEngine()
        from llm_mock_api.rule_engine import _SequenceStep

        steps = [_SequenceStep(reply="first"), _SequenceStep(reply="second")]
        rule = engine.add("seq", "")
        result = create_sequence_resolver(steps, rule)
        assert result.entry_count == 2
        assert result.resolver() == "first"
        assert result.resolver() == "second"

    def test_repeats_last(self) -> None:
        """超出步数后重复返回最后一条。"""
        from llm_mock_api.rule_engine import _SequenceStep

        engine = RuleEngine()
        steps = [_SequenceStep(reply="a"), _SequenceStep(reply="b")]
        rule = engine.add("seq", "")
        result = create_sequence_resolver(steps, rule)
        result.resolver()  # a
        result.resolver()  # b
        assert result.resolver() == "b"  # 继续返回最后一条

    def test_empty_raises(self) -> None:
        """空步骤列表应抛出异常。"""
        from llm_mock_api.rule_engine import _SequenceStep

        engine = RuleEngine()
        rule = engine.add("seq", "")
        import pytest
        with pytest.raises(ValueError):
            create_sequence_resolver([], rule)
