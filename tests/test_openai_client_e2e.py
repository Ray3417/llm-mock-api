"""面向用户的端到端集成测试。

使用 openai 官方 SDK（AsyncOpenAI）向 llm-mock-api 发送真实 HTTP 请求，
验证 Chat Completions 和 Responses 两个端点的核心功能。

特点：
- 只启动一次 MockServer（session 级），所有测试共享
- 每个测试类/方法前重新 load rules，清除序列回复与次数限制的状态
- 不修改任何源码，仅调用公共 API

使用方式::

    pytest tests/test_openai_client_e2e.py -v -s
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest

from llm_mock_api import cli
from llm_mock_api.mock_server import MockServer, MockServerOptions

# ---------------------------------------------------------------------------
# Session-scoped async fixtures: 启动一次 mock server + openai client
# 必须使用 pytest-asyncio 的 async session fixture，确保在同一个事件循环中管理 server
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 注意：pytest-asyncio 默认使用 Mode.STRICT，async test/method 需加 @pytest.mark.asyncio
# 本文件所有 async 测试方法已装饰；session-scoped async fixture 由 pytest-asyncio 自动管理
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def e2e_tmp_path(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """临时目录，存放 init 生成的默认规则。"""
    return tmp_path_factory.mktemp("e2e_rules")


@pytest.fixture(scope="session")
def e2e_config_path(e2e_tmp_path: pathlib.Path) -> pathlib.Path:
    """用 llm-mock-api init 生成默认 config.json + rules.json5。

    from_json_config 现在会以 config 文件所在目录解析 rules 的相对路径，
    因此无需手动覆写为绝对路径。
    """
    args = type("Args", (), {"dir": str(e2e_tmp_path), "force": False})()
    asyncio.run(cli._cmd_init(args))
    config_file = e2e_tmp_path / "config.json"
    assert config_file.exists(), f"init 未生成 {config_file}"
    return config_file


@pytest.fixture(scope="session")
async def e2e_server(e2e_config_path: pathlib.Path) -> MockServer:
    """从 config.json 启动一次 MockServer（带 default_chunk_size / default_latency / fallback），
    session 结束时停止。"""
    server = await MockServer.from_json_config(str(e2e_config_path))
    await server.start()
    yield server
    await server.stop()


@pytest.fixture(scope="session")
async def openai_client(e2e_server: MockServer):
    """AsyncOpenAI 客户端，base_url 指向 mock server。"""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=f"{e2e_server.url}/v1",
        api_key="sk-test",
    )
    yield client
    await client.close()


@pytest.fixture(autouse=True)
async def _reload_rules(e2e_server: MockServer, e2e_config_path: pathlib.Path) -> None:
    """每个测试方法前 reset 并重新加载 rules + fallback，
    清除序列回复进度与次数限制状态。

    复用 MockServerOptions.from_json_config 的路径解析逻辑（相对路径以 config 目录为基准）。
    """
    e2e_server.reset()
    opts = MockServerOptions.from_json_file(str(e2e_config_path))
    if opts.fallback is not None:
        e2e_server.fallback(opts.fallback)
    if opts.rules:
        await e2e_server.load(str(opts.rules))


# ===========================================================================
# 第一组：/v1/chat/completions
# ===========================================================================


class TestChatCompletionsBasic:
    """基础对话：简单问答、多模型、中文消息。"""

    @pytest.mark.asyncio
    async def test_hello_returns_content(self, openai_client: Any) -> None:
        """发送 'hi' → 返回非空 content。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hi"}],
        )
        content = resp.choices[0].message.content
        assert content is not None
        assert len(content) > 0
        assert resp.choices[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_hello_has_usage(self, openai_client: Any) -> None:
        """响应应包含 usage 字段（即使是 mock 值）。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert resp.usage is not None
        assert isinstance(resp.usage.total_tokens, int)
        assert resp.usage.total_tokens > 0

    @pytest.mark.asyncio
    async def test_chinese_message(self, openai_client: Any) -> None:
        """中文消息 '你好' 应匹配中文规则，返回中文回复。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "你好"}],
        )
        content = resp.choices[0].message.content
        assert content is not None
        # 中文回复包含中文字符
        assert any("\u4e00" <= ch <= "\u9fff" for ch in content), f"expected chinese, got: {content}"

    @pytest.mark.asyncio
    async def test_echo_regex_match(self, openai_client: Any) -> None:
        """'echo 12345'（不含其他关键词的触发字）匹配正则规则，回复包含 echo 提示。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "echo 12345"}],
        )
        content = resp.choices[0].message.content
        assert content is not None
        assert "echo" in content.lower() or "Echo" in content


class TestChatCompletionsStream:
    """流式输出：验证 SSE chunk 能被 SDK 正确解析。"""

    @pytest.mark.asyncio
    async def test_stream_receives_chunks(self, openai_client: Any) -> None:
        """stream=True: 异步迭代接收多个 chunk，合并后有非空文本。"""
        stream = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "long"}],
            stream=True,
        )
        chunks: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                chunks.append(chunk.choices[0].delta.content)
        assert len(chunks) > 1, f"expected multiple chunks, got {len(chunks)}"
        assert len("".join(chunks)) > 50, "merged text should be long enough"

    @pytest.mark.asyncio
    async def test_stream_order_is_sequential(self, openai_client: Any) -> None:
        """流式 chunk 顺序拼接应得到完整文本（与非流式一致）。"""
        # 非流式
        non_stream = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "long"}],
            stream=False,
        )
        non_stream_text = non_stream.choices[0].message.content or ""

        # 流式 - 重新 load 避免状态干扰（autouse fixture 已 reload）
        stream = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "long"}],
            stream=True,
        )
        merged: list[str] = []
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                merged.append(chunk.choices[0].delta.content)
        stream_text = "".join(merged)
        assert stream_text == non_stream_text, (
            f"stream text mismatch\n"
            f"  non-stream: {non_stream_text[:100]}...\n"
            f"  stream:     {stream_text[:100]}..."
        )


class TestChatCompletionsToolCalls:
    """工具调用：验证 mock 返回的 tool_calls 被 SDK 正确解析。"""

    @pytest.mark.asyncio
    async def test_weather_returns_tool_call(self, openai_client: Any) -> None:
        """'weather' 触发工具调用 → tool_calls 非空，函数名为 get_weather。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "weather"}],
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None, "expected tool_calls, got None"
        assert len(msg.tool_calls) > 0
        assert msg.tool_calls[0].function.name == "get_weather"
        # arguments 是 JSON 字符串，应能被解析
        import json as _json

        args = _json.loads(msg.tool_calls[0].function.arguments)
        assert "location" in args or "unit" in args

    @pytest.mark.asyncio
    async def test_tool_call_finish_reason(self, openai_client: Any) -> None:
        """有 tool_calls 的回复，finish_reason 应为 'tool_calls'。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "weather"}],
        )
        assert resp.choices[0].finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_multi_tool_calls(self, openai_client: Any) -> None:
        """'multi tool' 触发多个工具调用。"""
        resp = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "multi tool"}],
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) >= 2, f"expected >=2 tool_calls, got {len(msg.tool_calls or [])}"


class TestChatCompletionsSequence:
    """序列回复与次数限制：step 推进，once 消耗后回退。"""

    @pytest.mark.asyncio
    async def test_step_advances_on_each_request(self, openai_client: Any) -> None:
        """连续发送 'step'，每次得到不同回复（序列推进）。"""
        replies: list[str] = []
        for _ in range(3):
            resp = await openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "step"}],
            )
            content = resp.choices[0].message.content
            assert content is not None
            replies.append(content)
        # 三条回复应不全相同（序列推进）
        assert len(set(replies)) >= 2, f"expected advancing sequence, got: {replies}"

    @pytest.mark.asyncio
    async def test_once_matches_first_then_falls_through(self, openai_client: Any) -> None:
        """'once' 第一次匹配，第二次应落入 fallback（或其他规则）。"""
        # 第一次
        r1 = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "once"}],
        )
        first = r1.choices[0].message.content
        assert first is not None and len(first) > 0

        # 第二次 - times 已耗尽，应落入 fallback
        r2 = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "once"}],
        )
        second = r2.choices[0].message.content
        # fallback 不同于第一条规则的回复
        assert second != first, f"expected fallback to differ from first reply, both: {second}"


# ===========================================================================
# 第二组：/v1/responses（同样的场景，换一个 API 端点）
# ===========================================================================


class TestResponsesBasic:
    """/v1/responses 基础对话（含 reasoning 字段验证）。"""

    @pytest.mark.asyncio
    async def test_hello_returns_output(self, openai_client: Any) -> None:
        """'hi' → output 数组非空，包含文本。"""
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "hi"}],
        )
        assert resp.output is not None
        assert len(resp.output) > 0

        # 提取文本（Responses API: output[0].content[0].text）
        first_output = resp.output[0]
        assert hasattr(first_output, "content")
        content_items = first_output.content
        assert len(content_items) > 0
        text = getattr(content_items[0], "text", "")
        assert isinstance(text, str)
        assert len(text) > 0

    @pytest.mark.asyncio
    async def test_responses_has_usage(self, openai_client: Any) -> None:
        """Responses 也应返回 usage。"""
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "hello"}],
        )
        assert resp.usage is not None
        assert resp.usage.output_tokens is not None and resp.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_chinese_responses(self, openai_client: Any) -> None:
        """中文消息 → Responses 返回中文文本。"""
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "你好"}],
        )
        first_output = resp.output[0]
        text = getattr(first_output.content[0], "text", "")
        assert any("\u4e00" <= ch <= "\u9fff" for ch in text), f"expected chinese, got: {text}"

    @pytest.mark.asyncio
    async def test_joke_returns_reasoning_and_text(self, openai_client: Any) -> None:
        """'joke' 触发带 reasoning 的结构化回复 → output 含 reasoning item + message item。

        Responses API 将 reasoning 作为独立的 output item（type='reasoning'），
        reasoning_text 通过 summary[0].text 访问；正文在 type='message' item 中。
        """
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "joke"}],
        )
        assert resp.output is not None and len(resp.output) >= 2, (
            f"expected at least 2 output items (reasoning + message), got: {len(resp.output or [])}"
        )

        # 第一个 output 应为 reasoning
        reasoning_item = resp.output[0]
        assert getattr(reasoning_item, "type", None) == "reasoning", (
            f"expected first output type='reasoning', got: {getattr(reasoning_item, 'type', None)}"
        )
        summary = getattr(reasoning_item, "summary", None)
        assert isinstance(summary, list) and len(summary) > 0
        assert hasattr(summary[0], "text")
        reasoning_text = getattr(summary[0], "text", "")
        assert isinstance(reasoning_text, str) and len(reasoning_text) > 0

        # 第二个 output 应为 message，包含正文
        message_item = resp.output[1]
        assert getattr(message_item, "type", None) == "message"
        content_list = getattr(message_item, "content", None)
        assert isinstance(content_list, list) and len(content_list) > 0
        main_text = getattr(content_list[0], "text", "")
        assert isinstance(main_text, str) and len(main_text) > 0

        # 验证内容合理：reasoning 与正文不同
        assert reasoning_text.lower() != main_text.lower()

    @pytest.mark.asyncio
    async def test_unknown_template_returns_reasoning(self, openai_client: Any) -> None:
        """'unknown' 引用 templates.cant_answer（内含 reasoning）→ reasoning 字段正确传递。"""
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "unknown"}],
        )
        # 在 output 中找到 reasoning item
        reasoning_items = [
            item for item in (resp.output or [])
            if getattr(item, "type", None) == "reasoning"
        ]
        assert len(reasoning_items) > 0, "template with reasoning should produce reasoning item"
        summary = getattr(reasoning_items[0], "summary", None)
        assert isinstance(summary, list) and len(summary) > 0
        reasoning_text = getattr(summary[0], "text", "")
        # 模板中预设的推理文本
        assert "outside my scope" in reasoning_text


class TestResponsesStream:
    """/v1/responses 流式输出。"""

    @pytest.mark.asyncio
    async def test_stream_responses_receives_events(self, openai_client: Any) -> None:
        """stream=True → 异步迭代收到多个事件，合并后有文本。"""
        chunks: list[str] = []
        async for event in await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "long"}],
            stream=True,
        ):
            # Responses API 的事件对象；尝试从各种可能的属性提取文本
            text_piece = self._extract_text_from_event(event)
            if text_piece:
                chunks.append(text_piece)
        assert len(chunks) >= 1, f"expected stream chunks, got {len(chunks)}"
        assert len("".join(chunks)) > 50

    @staticmethod
    def _extract_text_from_event(event: Any) -> str:
        """从 Responses stream event 中提取文本片段（尽力而为）。"""
        # 尝试多种常见字段
        for attr in ("delta", "text"):
            val = getattr(event, attr, None)
            if isinstance(val, str) and val:
                return val
        # 可能是 output_text 事件
        output = getattr(event, "output", None)
        if output is not None:
            # output 可能是单条或列表
            items = output if isinstance(output, list) else [output]
            for item in items:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        text = getattr(c, "text", None)
                        if isinstance(text, str) and text:
                            return text
        return ""


class TestResponsesToolCalls:
    """/v1/responses 工具调用。"""

    @pytest.mark.asyncio
    async def test_weather_returns_function_call(self, openai_client: Any) -> None:
        """'weather' → Responses output 包含 function_call 类型的 item。"""
        resp = await openai_client.responses.create(
            model="gpt-4",
            input=[{"role": "user", "content": "weather"}],
        )
        assert resp.output is not None and len(resp.output) > 0
        # 在 Responses API 中，工具调用以 type="function_call" 的 item 直接出现在 output 数组
        function_calls = [
            item for item in resp.output
            if getattr(item, "type", None) == "function_call"
        ]
        assert len(function_calls) > 0, (
            f"expected function_call in output, "
            f"got types: {[getattr(o, 'type', type(o).__name__) for o in resp.output]}"
        )
        # 验证函数名正确
        assert function_calls[0].name == "get_weather"
