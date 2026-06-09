"""loader 模块单元测试。"""

from __future__ import annotations

import asyncio
import pathlib
import tempfile

from llm_mock_api.loader import (
    LoadContext,
    _compile_match,
    _load_json5_file,
    _parse_regex_string,
    _parse_reply,
    _resolve_reply_ref,
    load_rules_from_path,
)
from llm_mock_api.rule_engine import RuleEngine
from llm_mock_api.types import MockRequest


def _req(last_message: str = "", model: str = "gpt-4", format: str = "openai") -> MockRequest:
    return MockRequest(
        format=format,
        model=model,
        streaming=False,
        messages=(),
        last_message=last_message,
        system_message="",
        tool_names=(),
        last_tool_call_id=None,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


class TestParseRegexString:
    """测试 _parse_regex_string：正则字符串解析。"""

    def test_plain_string(self) -> None:
        """普通字符串原样返回。"""
        assert _parse_regex_string("hello") == "hello"

    def test_basic_regex(self) -> None:
        """基础正则：/pattern/。"""
        result = _parse_regex_string(r"/\d+/")
        assert hasattr(result, "search")
        assert result.search("abc123") is not None

    def test_case_insensitive_flag(self) -> None:
        """i 标志：大小写不敏感。"""
        result = _parse_regex_string(r"/hello/i")
        assert result.search("HELLO") is not None

    def test_multiline_flag(self) -> None:
        """m 标志：多行模式。"""
        result = _parse_regex_string(r"/^hello/m")
        assert result.search("world\nhello") is not None


class TestCompileMatch:
    """测试 _compile_match：JSON5 匹配配置到 Match 对象。"""

    def test_string_match(self) -> None:
        """字符串匹配直接解析（可能含正则）。"""
        m = _compile_match("hello")
        assert isinstance(m, str)

    def test_object_message_field(self) -> None:
        """对象中的 message 字段。"""
        m = _compile_match({"message": "hello"})
        assert hasattr(m, "message")
        assert m.message == "hello"

    def test_object_format_field(self) -> None:
        """对象中的 format 字段。"""
        m = _compile_match({"format": "anthropic"})
        assert m.format == "anthropic"

    def test_object_multi_field(self) -> None:
        """多字段对象。"""
        m = _compile_match({"message": "hi", "model": "gpt-4"})
        assert m.message == "hi"
        assert m.model == "gpt-4"

    def test_object_regex_message(self) -> None:
        """对象中的 message 含正则。"""
        m = _compile_match({"message": r"/\d+/"})
        assert hasattr(m.message, "search")


class TestParseReply:
    """测试 _parse_reply：JSON5 reply 到 Reply 对象。"""

    def test_string_reply(self) -> None:
        """字符串回复。"""
        assert _parse_reply("hello") == "hello"

    def test_object_with_text(self) -> None:
        """对象回复：text 字段。"""
        r = _parse_reply({"text": "hello"})
        assert hasattr(r, "text")
        assert r.text == "hello"

    def test_object_with_reasoning(self) -> None:
        """对象回复：reasoning 字段。"""
        r = _parse_reply({"reasoning": "thinking..."})
        assert r.reasoning == "thinking..."

    def test_object_with_tools(self) -> None:
        """对象回复：tools 字段。"""
        r = _parse_reply({"tools": [{"name": "search", "args": {"q": "test"}}]})
        assert r.tools is not None
        assert len(r.tools) == 1
        assert r.tools[0].name == "search"
        assert r.tools[0].args == {"q": "test"}

    def test_object_full(self) -> None:
        """对象回复：完整字段。"""
        r = _parse_reply({
            "text": "hello",
            "reasoning": "thought",
            "tools": [{"name": "fn", "args": {}}],
        })
        assert r.text == "hello"
        assert r.reasoning == "thought"
        assert r.tools is not None


class TestResolveReplyRef:
    """测试 _resolve_reply_ref：模板引用解析。"""

    def test_plain_string(self) -> None:
        """非模板引用原样返回。"""
        assert _resolve_reply_ref("hello", None, "test.json5") == "hello"

    def test_plain_object(self) -> None:
        """对象原样返回。"""
        assert _resolve_reply_ref({"text": "hi"}, None, "test.json5") == {"text": "hi"}

    def test_template_lookup(self) -> None:
        """模板引用：$name 从 templates 查。"""
        templates = {"greet": "Hello there!"}
        resolved = _resolve_reply_ref("$greet", templates, "test.json5")
        assert resolved == "Hello there!"

    def test_template_object(self) -> None:
        """模板引用：值为对象。"""
        templates = {"greet": {"text": "Hello!", "reasoning": "say hi"}}
        resolved = _resolve_reply_ref("$greet", templates, "test.json5")
        assert resolved == {"text": "Hello!", "reasoning": "say hi"}

    def test_missing_template_raises(self) -> None:
        """模板不存在抛错。"""
        try:
            _resolve_reply_ref("$unknown", {}, "test.json5")
            raise AssertionError("should have raised")
        except ValueError as e:
            assert "unknown" in str(e)


# ---------------------------------------------------------------------------
# JSON5 文件加载
# ---------------------------------------------------------------------------


class TestLoadJson5File:
    """测试 _load_json5_file：JSON5 规则文件加载。"""

    def test_array_form(self) -> None:
        """数组形式：简单规则列表。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[
  { when: "hello", reply: "Hi!" },
  { when: "bye", reply: "Goodbye!" }
]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        assert engine.match(_req(last_message="hello")) is not None
        assert engine.match(_req(last_message="bye")) is not None
        assert engine.match(_req(last_message="nope")) is None

    def test_object_form_with_templates(self) -> None:
        """对象形式：含 templates 和 rules。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """{
  templates: { greet: "Hello from template!" },
  rules: [ { when: "hi", reply: "$greet" } ]
}""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        rule = engine.match(_req(last_message="hi there"))
        assert rule is not None
        assert rule.resolve == "Hello from template!"

    def test_times_limit(self) -> None:
        """规则带 times 限制。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{ when: "once", reply: "only once", times: 1 }]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        assert engine.match(_req(last_message="once")) is not None
        # 第二次不应匹配（已消耗一次）
        assert engine.match(_req(last_message="once")) is None

    def test_sequence_rule(self) -> None:
        """序列规则：多回复依次返回。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{
  when: "step",
  replies: [{ reply: "First" }, { reply: "Second", latency: 100 }]
}]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        rule1 = engine.match(_req(last_message="step"))
        assert rule1 is not None
        # 第一次返回 "First"（resolve 是 callable，调用后得到 reply）
        # 但 RuleEngine.match 已消耗一次 remaining
        rule2 = engine.match(_req(last_message="step"))
        assert rule2 is not None

    def test_object_match_in_json5(self) -> None:
        """JSON5 中使用对象格式的 match。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{
  when: { model: "gpt-4", format: "openai" },
  reply: "gpt-4 response"
}]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        assert engine.match(_req(last_message="anything", model="gpt-4")) is not None
        assert engine.match(_req(last_message="anything", model="claude")) is None

    def test_regex_in_when(self) -> None:
        """JSON5 中使用正则匹配。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{ when: "/^test/", reply: "matched regex" }]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        assert engine.match(_req(last_message="testing")) is not None
        assert engine.match(_req(last_message="no match")) is None


# ---------------------------------------------------------------------------
# 目录遍历
# ---------------------------------------------------------------------------


class TestLoadRulesFromPath:
    """测试 load_rules_from_path：递归目录加载。"""

    def test_load_single_file(self) -> None:
        """加载单个 JSON5 文件。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{ when: "hello", reply: "Hi!" }]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(load_rules_from_path(str(fp), ctx))

        assert engine.match(_req(last_message="hello")) is not None

    def test_load_directory(self) -> None:
        """加载整个目录。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            (base / "a.json5").write_text(
                """[{ when: "alpha", reply: "A" }]""",
                encoding="utf-8",
            )
            (base / "b.json5").write_text(
                """[{ when: "beta", reply: "B" }]""",
                encoding="utf-8",
            )

            ctx = LoadContext(engine=engine)
            asyncio.run(load_rules_from_path(str(base), ctx))

        assert engine.match(_req(last_message="alpha")) is not None
        assert engine.match(_req(last_message="beta")) is not None

    def test_load_nested_directory(self) -> None:
        """加载嵌套目录。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            sub = base / "sub"
            sub.mkdir()
            (sub / "nested.json5").write_text(
                """[{ when: "deep", reply: "found" }]""",
                encoding="utf-8",
            )

            ctx = LoadContext(engine=engine)
            asyncio.run(load_rules_from_path(str(base), ctx))

        assert engine.match(_req(last_message="deep")) is not None

    def test_unsupported_extension_raises(self) -> None:
        """不支持的扩展名报错。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.txt"
            fp.write_text("hello", encoding="utf-8")
            ctx = LoadContext(engine=engine)
            try:
                asyncio.run(load_rules_from_path(str(fp), ctx))
                raise AssertionError("should have raised")
            except ValueError as e:
                assert "extension" in str(e).lower()


# ---------------------------------------------------------------------------
# Python 处理器文件
# ---------------------------------------------------------------------------


class TestLoadHandlerFile:
    """测试 _load_handler_file：Python 处理器文件加载。"""

    def test_single_handler(self) -> None:
        """单个 Handler 对象。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "handler.py"
            fp.write_text(
                '''from llm_mock_api.types import MockRequest

class _Handler:
    def match(self, req: MockRequest) -> bool:
        return "echo" in req.last_message

    def respond(self, req: MockRequest) -> str:
        return f"Echo: {req.last_message}"

default = _Handler()
''',
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(load_rules_from_path(str(fp), ctx))

        rule = engine.match(_req(last_message="echo hello"))
        assert rule is not None

    def test_handler_list(self) -> None:
        """Handler 列表。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "handlers.py"
            fp.write_text(
                '''from llm_mock_api.types import MockRequest

class _HandlerA:
    def match(self, req: MockRequest) -> bool:
        return "aaa" in req.last_message

    def respond(self, req: MockRequest) -> str:
        return "got aaa"

class _HandlerB:
    def match(self, req: MockRequest) -> bool:
        return "bbb" in req.last_message

    def respond(self, req: MockRequest) -> str:
        return "got bbb"

default = [_HandlerA(), _HandlerB()]
''',
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(load_rules_from_path(str(fp), ctx))

        assert engine.match(_req(last_message="xxx aaa xxx")) is not None
        assert engine.match(_req(last_message="yyy bbb yyy")) is not None

    def test_handler_with_fallback(self) -> None:
        """处理器文件带 fallback。"""
        engine = RuleEngine()
        fallback_replies: list = []

        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "handler.py"
            fp.write_text(
                '''class _Handler:
    def match(self, req):
        return False
    def respond(self, req):
        return "nope"

default = _Handler()
fallback = "Default response"
''',
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine, set_fallback=lambda r: fallback_replies.append(r))
            asyncio.run(load_rules_from_path(str(fp), ctx))

        assert len(fallback_replies) == 1
        assert fallback_replies[0] == "Default response"


# ---------------------------------------------------------------------------
# JSON5 中的 Object Reply
# ---------------------------------------------------------------------------


class TestJson5ObjectReply:
    """测试 JSON5 中的对象 reply。"""

    def test_object_reply_text_only(self) -> None:
        """reply 是对象时，解析为 ReplyObject。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """[{ when: "hello", reply: { text: "Hello world!" } }]""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        rule = engine.match(_req(last_message="hello"))
        assert rule is not None
        assert hasattr(rule.resolve, "text")
        assert rule.resolve.text == "Hello world!"

    def test_object_reply_with_template(self) -> None:
        """模板引用的对象 reply。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """{
  templates: {
    resp: { text: "response text", reasoning: "thinking" }
  },
  rules: [{ when: "ask", reply: "$resp" }]
}""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)
            asyncio.run(_load_json5_file(str(fp), ctx))

        rule = engine.match(_req(last_message="ask me"))
        assert rule is not None
        assert rule.resolve.text == "response text"
        assert rule.resolve.reasoning == "thinking"


class TestFallbackRegistration:
    """测试 fallback 注册。"""

    def test_json5_fallback(self) -> None:
        """JSON5 文件中的 fallback 字段。"""
        engine = RuleEngine()
        fallbacks: list = []

        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """{
  fallback: { text: "I don't understand." },
  rules: [{ when: "hello", reply: "Hi!" }]
}""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine, set_fallback=lambda r: fallbacks.append(r))
            asyncio.run(_load_json5_file(str(fp), ctx))

        assert len(fallbacks) == 1
        assert hasattr(fallbacks[0], "text")
        assert fallbacks[0].text == "I don't understand."

    def test_no_set_fallback_callback(self) -> None:
        """没有 set_fallback 回调时不报错。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "rules.json5"
            fp.write_text(
                """{
  fallback: "oops",
  rules: [{ when: "hi", reply: "hello" }]
}""",
                encoding="utf-8",
            )
            ctx = LoadContext(engine=engine)  # 无 set_fallback
            asyncio.run(_load_json5_file(str(fp), ctx))  # 不应抛错

        assert engine.match(_req(last_message="hi")) is not None


# ---------------------------------------------------------------------------
# 错误路径
# ---------------------------------------------------------------------------


class TestErrorCases:
    """测试错误输入。"""

    def test_invalid_top_level_type(self) -> None:
        """JSON5 顶层既不是数组也不是对象。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "bad.json5"
            fp.write_text("""42""", encoding="utf-8")
            ctx = LoadContext(engine=engine)
            try:
                asyncio.run(_load_json5_file(str(fp), ctx))
                raise AssertionError("should have raised")
            except ValueError:
                pass

    def test_missing_when(self) -> None:
        """规则缺少 when 字段。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "bad.json5"
            fp.write_text("""[{ reply: "hi" }]""", encoding="utf-8")
            ctx = LoadContext(engine=engine)
            try:
                asyncio.run(_load_json5_file(str(fp), ctx))
                raise AssertionError("should have raised")
            except ValueError as e:
                assert "when" in str(e).lower()

    def test_missing_reply(self) -> None:
        """规则缺少 reply 字段。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "bad.json5"
            fp.write_text("""[{ when: "hi" }]""", encoding="utf-8")
            ctx = LoadContext(engine=engine)
            try:
                asyncio.run(_load_json5_file(str(fp), ctx))
                raise AssertionError("should have raised")
            except ValueError as e:
                assert "reply" in str(e).lower()

    def test_handler_without_default_raises(self) -> None:
        """处理器文件缺少 default 抛错。"""
        engine = RuleEngine()
        with tempfile.TemporaryDirectory() as td:
            fp = pathlib.Path(td) / "bad.py"
            fp.write_text("""x = 1\n""", encoding="utf-8")
            ctx = LoadContext(engine=engine)
            try:
                asyncio.run(load_rules_from_path(str(fp), ctx))
                raise AssertionError("should have raised")
            except ValueError:
                pass
