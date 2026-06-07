"""rule_builder 模块单元测试。"""

from __future__ import annotations

from llm_mock_api.rule_builder import RuleBuilder, normalise_sequence_entries
from llm_mock_api.rule_engine import RuleEngine
from llm_mock_api.types import MatchObject, MockRequest, ReplyOptions
from llm_mock_api.types.reply import ReplySequenceEntryWithOptions


def _req(last_message: str = "", tool_names: tuple[str, ...] = (),
         last_tool_call_id: str | None = None) -> MockRequest:
    """创建一个简化的 MockRequest 用于测试。"""
    return MockRequest(
        format="openai",
        model="gpt-4",
        streaming=False,
        messages=(),
        last_message=last_message,
        system_message="",
        tool_names=tool_names,
        last_tool_call_id=last_tool_call_id,
    )


class TestWhenReply:
    """测试 when + reply 基本流程。"""

    def test_basic_when_reply(self) -> None:
        """builder.when(match).reply(response) 应注册规则。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("hello").reply("Hi!")
        rule = engine.match(_req(last_message="hello world"))
        assert rule is not None
        assert rule.resolve == "Hi!"

    def test_reply_with_options(self) -> None:
        """支持传入 ReplyOptions。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("opt").reply("ok", ReplyOptions())
        rule = engine.match(_req(last_message="opt"))
        assert rule is not None


class TestTimes:
    """测试 .times() 链式调用。"""

    def test_times_limits_matches(self) -> None:
        """times(2) 应限制规则仅匹配 2 次。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("bye").reply("Goodbye").times(2)
        r1 = engine.match(_req(last_message="bye"))
        r2 = engine.match(_req(last_message="bye"))
        r3 = engine.match(_req(last_message="bye"))
        assert r1 is not None and r2 is not None
        assert r3 is None


class TestFirst:
    """测试 .first() 优先级。"""

    def test_first_moves_to_front(self) -> None:
        """first() 应将规则移到列表最前面。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("key").reply("low priority")
        builder.when("key").reply("HIGH PRIORITY").first()
        rule = engine.match(_req(last_message="key"))
        assert rule.resolve == "HIGH PRIORITY"

    def test_chaining_times_and_first(self) -> None:
        """times 和 first 应可链式调用。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("x").reply("once").times(1).first()
        r1 = engine.match(_req(last_message="x"))
        r2 = engine.match(_req(last_message="x"))
        assert r1 is not None
        assert r2 is None


class TestShortcuts:
    """测试 when_tool / when_tool_result 快捷方式。"""

    def test_when_tool(self) -> None:
        """when_tool 应匹配工具名。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when_tool("my_tool").reply("tool-ok")
        rule = engine.match(_req(last_message="", tool_names=("my_tool",)))
        assert rule is not None

    def test_when_tool_result(self) -> None:
        """when_tool_result 应匹配 tool_call_id。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when_tool_result("call_001").reply("result-ok")
        rule = engine.match(_req(last_message="", last_tool_call_id="call_001"))
        assert rule is not None


class TestReplySequence:
    """测试 reply_sequence 序列回复。"""

    def test_returns_in_order(self) -> None:
        """序列回复应按顺序返回。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.when("seq").reply_sequence(["first", "second"])
        r1 = engine.match(_req(last_message="seq test"))
        r2 = engine.match(_req(last_message="seq test"))
        assert r1.resolve() == "first"
        assert r2.resolve() == "second"


class TestNextError:
    """测试 next_error 一次性错误。"""

    def test_next_error_registers_once(self) -> None:
        """next_error 应注册一个一次性错误回复。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.next_error(400, "Bad request", "invalid_request")
        rule = engine.match(_req(last_message="anything"))
        assert rule is not None
        reply = rule.resolve
        assert reply.error is not None
        assert reply.error.status == 400
        assert reply.error.message == "Bad request"
        assert reply.error.type == "invalid_request"

    def test_next_error_auto_removed(self) -> None:
        """next_error 匹配后应自动移除。"""
        engine = RuleEngine()
        builder = RuleBuilder(engine)
        builder.next_error(500, "Error", "server_error")
        engine.match(_req(last_message="anything"))
        assert engine.match(_req(last_message="anything")) is None


class TestNormaliseSequenceEntries:
    """测试 normalise_sequence_entries。"""

    def test_mixed_entries(self) -> None:
        """混合字符串和带选项条目应都转为步骤。"""
        entries = [
            "hello",
            ReplySequenceEntryWithOptions("world", ReplyOptions()),
        ]
        steps = normalise_sequence_entries(entries)
        assert len(steps) == 2
        assert steps[0].reply == "hello"
        assert steps[0].options is None
        assert steps[1].reply == "world"
        assert steps[1].options is not None

    def test_all_string_entries(self) -> None:
        """纯字符串列表应正确转换。"""
        steps = normalise_sequence_entries(["a", "b", "c"])
        assert len(steps) == 3
        assert steps[2].reply == "c"
