"""mock_server.py 的单元测试（不含真 HTTP 请求）。

本文件包含四类非网络测试，按复杂度递增：

1. **配置文件 / MockServerOptions** — from_json_file 行为、字段默认值。
2. **纯单元测试** — 不启动任何服务器，只测 Python 对象逻辑
   （规则注册、fallback、reset、is_done、history 计数等）。
3. **TestClient 集成测试** — 使用 FastAPI 的 TestClient
   把路由处理逻辑在同一进程里跑一遍；覆盖 JSON 响应、SSE 流、
   规则匹配优先级、`.times(N)` 消耗、`next_error` 单次触发后消失、
   `/v1/responses` 路由、无效请求体、ReplyObject 结构化回复。
4. **load() 与文件系统快照** — rules json5 加载、`_fs_snapshot` 变更检测。

真 HTTP 集成测试（启动 uvicorn + httpx2 请求）在 `test_mock_server_integration.py`。
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from llm_mock_api.mock_server import (
    MockServer,
    MockServerOptions,
    _fs_snapshot,
)
from llm_mock_api.types.reply import ReplyObject


# ======================================================================
# 1. 配置文件 / MockServerOptions — from_json_file 行为
# ======================================================================


class TestMockServerOptions:
    """验证 MockServerOptions 的语义：字段默认值、JSON 加载行为。"""

    def test_mock_server_applies_sensible_defaults(self) -> None:
        """未传选项时，MockServer 应使用 127.0.0.1 + 随机端口（测试友好）。"""
        server = MockServer()
        assert server._host == "127.0.0.1"
        assert server._default_port == 0

    def test_host_0_0_0_0_is_respected(self) -> None:
        """显式传 host="0.0.0.0" 时，MockServer 应绑定到所有接口。"""
        server = MockServer(MockServerOptions(host="0.0.0.0"))
        assert server._host == "0.0.0.0"

    def test_load_from_json_file(self, tmp_path) -> None:
        """从 JSON 配置文件加载选项；未写的字段使用 MockServerOptions 的 dataclass 默认值。"""
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"port": 8002, "host": "127.0.0.1"}), encoding="utf-8")

        opts = MockServerOptions.from_json_file(str(cfg))
        assert opts.port == 8002
        assert opts.host == "127.0.0.1"
        # 未写字段 → 使用 MockServerOptions 的 dataclass 默认值
        assert opts.log_level == "none"
        assert opts.default_latency == 0
        assert opts.default_chunk_size == 0

    def test_json_missing_file_raises_runtime_error(self, tmp_path) -> None:
        """不存在的文件 → RuntimeError（带路径提示）。"""
        missing = tmp_path / "does-not-exist.json"
        with pytest.raises(RuntimeError, match="not found"):
            MockServerOptions.from_json_file(str(missing))

    def test_json_invalid_syntax_raises_runtime_error(self, tmp_path) -> None:
        """内容不是合法 JSON → RuntimeError（包含 "Invalid JSON" 提示）。"""
        cfg = tmp_path / "config.json"
        cfg.write_text("{ not valid json", encoding="utf-8")
        with pytest.raises(RuntimeError, match="Invalid JSON"):
            MockServerOptions.from_json_file(str(cfg))

    def test_json_full_fields_roundtrip(self, tmp_path) -> None:
        """8 个字段（port/host/log_level/default_latency/default_chunk_size/
        fallback/rules/watch）都能从 JSON 正确加载。"""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "port": 8002,
                    "host": "0.0.0.0",
                    "log_level": "info",
                    "default_latency": 5,
                    "default_chunk_size": 50,
                    "fallback": "sorry, no match",
                    "rules": "./examples/rules.json5",
                    "watch": True,
                },
            ),
            encoding="utf-8",
        )
        parsed = MockServerOptions.from_json_file(str(cfg))
        assert parsed.port == 8002
        assert parsed.host == "0.0.0.0"
        assert parsed.log_level == "info"
        assert parsed.default_latency == 5
        assert parsed.default_chunk_size == 50
        assert parsed.fallback == "sorry, no match"
        # rules 相对路径以 config 所在目录解析（而不是当前工作目录）
        assert pathlib.Path(parsed.rules).is_absolute()
        assert pathlib.Path(parsed.rules) == cfg.parent / "examples" / "rules.json5"
        assert parsed.watch is True


# ======================================================================
# 2. from_json_config 端到端 — 构造 MockServer + 应用 rules/fallback
# ======================================================================


class TestFromJsonConfig:
    """end-to-end 验证 MockServer.from_json_config 正确初始化了 fallback 和 rules。"""

    def test_fallback_only(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"fallback": "sorry, I don't know"}), encoding="utf-8")
        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server._fallback_reply == "sorry, I don't know"

    def test_load_rules_from_path(self, tmp_path) -> None:
        rules_file = tmp_path / "rules.json5"
        rules_file.write_text(
            """
        [
          { when: "ping", reply: "pong" }
        ]
        """,
            encoding="utf-8",
        )
        cfg = tmp_path / "config.json"
        cfg.write_text(json.dumps({"rules": str(rules_file)}), encoding="utf-8")

        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server.rule_count >= 1

    def test_missing_rules_path_is_handled(self, tmp_path) -> None:
        """rules 指向不存在的路径 → loader 跳过；规则数保持 0，不抛异常。"""
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"rules": str(tmp_path / "does-not-exist.json5")}),
            encoding="utf-8",
        )
        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server.rule_count == 0

    def test_empty_json_is_legitimate(self, tmp_path) -> None:
        cfg = tmp_path / "config.json"
        cfg.write_text("{}", encoding="utf-8")
        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server._host == "127.0.0.1"
        assert server._default_port == 0

    def test_watch_path_recorded_when_watch_true_and_rules_present(self, tmp_path) -> None:
        """watch: true + rules 存在 → server._watch_path 应被设置，供 run_until_shutdown 使用。"""
        rules_file = tmp_path / "rules.json5"
        rules_file.write_text('[{ when: "ping", reply: "pong" }]', encoding="utf-8")
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"rules": str(rules_file), "watch": True}),
            encoding="utf-8",
        )
        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server._watch_path == str(rules_file)

    def test_watch_path_not_recorded_when_watch_false(self, tmp_path) -> None:
        rules_file = tmp_path / "rules.json5"
        rules_file.write_text('[{ when: "ping", reply: "pong" }]', encoding="utf-8")
        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps({"rules": str(rules_file)}),  # 缺省 watch=false
            encoding="utf-8",
        )
        server = asyncio.run(MockServer.from_json_config(str(cfg)))
        assert server._watch_path is None


# ======================================================================
# 3. 文件系统快照（watch 模式依赖）
# ======================================================================


class TestFsSnapshot:
    """_fs_snapshot 用于 watch 模式的变更检测；测其可区分文件内容变化。"""

    def test_file_snapshot_changes_on_write(self, tmp_path) -> None:
        f = tmp_path / "x.json5"
        f.write_text('{"a":1}', encoding="utf-8")
        before = _fs_snapshot(str(f))
        import time

        time.sleep(0.01)
        f.write_text('{"a":2}', encoding="utf-8")
        after = _fs_snapshot(str(f))
        assert before != after

    def test_dir_snapshot_changes_when_file_added(self, tmp_path) -> None:
        d = tmp_path / "rules"
        d.mkdir()
        before = _fs_snapshot(str(d))
        (d / "a.json5").write_text("[]", encoding="utf-8")
        after = _fs_snapshot(str(d))
        assert before != after


# ======================================================================
# 4. 纯单元测试 — 对象行为、状态机
# ======================================================================


class TestMockServerUnit:
    """测试不涉及网络的部分。"""

    def test_initial_state(self) -> None:
        server = MockServer()
        assert server.rule_count == 0
        assert server.is_done() is True  # 0 条规则 → 视为"全部消耗完"
        assert server.routes == [
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/messages",
        ]
        assert server._listening is False

    def test_when_delegates_to_rule_builder(self) -> None:
        server = MockServer()
        pending = server.when("hello")
        handle = pending.reply("Hi there!")
        assert server.rule_count == 1
        handle.times(5)
        assert server.is_done() is False
        assert hasattr(handle, "times")
        assert hasattr(handle, "first")

    def test_reset_clears_everything(self) -> None:
        server = MockServer()
        server.when("a").reply("b")
        server.when("x").reply("y")
        server.fallback("new-fallback")
        assert server.rule_count == 2
        server.reset()
        assert server.rule_count == 0
        assert server.is_done() is True
        assert server._fallback_reply == "Mock server: no matching rule."

    def test_url_before_start_raises(self) -> None:
        server = MockServer()
        with pytest.raises(RuntimeError, match="Server is not running"):
            _ = server.url

    def test_rules_summary_is_list_of_descriptions(self) -> None:
        server = MockServer()
        server.when("ping").reply("pong")
        summaries = server.rules
        assert isinstance(summaries, list)
        assert len(summaries) == 1
        assert summaries[0].description
        assert isinstance(summaries[0].remaining, float)


# ======================================================================
# 5. TestClient 集成测试 — FastAPI 内置 in-process 请求器
#
# 覆盖：JSON 响应、SSE 流、规则匹配顺序、times 消耗、next_error 单次触发、
# /v1/responses 路由、无效请求体、ReplyObject 结构化回复。
# ======================================================================


class TestWithTestClient:
    """通过 TestClient 测路由层逻辑。"""

    def test_matched_rule_returns_reply(self) -> None:
        server = MockServer()
        server.when("hello").reply("Hi!")
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            assert "Hi!" in resp.json()["choices"][0]["message"]["content"]

    def test_no_match_returns_fallback(self) -> None:
        server = MockServer()
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "no-match-plz"}],
                    "stream": False,
                },
            )
            assert resp.status_code == 200
            assert "Mock server: no matching rule." in resp.json()["choices"][0]["message"]["content"]

    def test_custom_fallback_applied(self) -> None:
        server = MockServer()
        server.fallback("my-custom-fallback")
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "no match"}],
                    "stream": False,
                },
            )
            assert "my-custom-fallback" in resp.json()["choices"][0]["message"]["content"]

    def test_invalid_request_body_returns_400(self) -> None:
        server = MockServer()
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={"garbage": "payload"},  # 缺 model / messages
            )
        assert resp.status_code == 400

    def test_responses_format_route_works(self) -> None:
        server = MockServer()
        server.when("hello").reply("Hi from responses!")
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/responses",
                json={"model": "gpt-4", "input": "hello"},
            )
        assert resp.status_code == 200

    # --- 新增的关键测试 ---

    def test_stream_true_returns_sse(self) -> None:
        """stream=True 应返回 SSE 流。

        关键点：Content-Type 包含 text/event-stream；body 包含
        'data: {...}\n\ndata: [... DONE ...]' 这种标准 SSE 载荷。

        注意：不传递 stream 字段默认为非流式（与主流 LLM API 一致）。
        """
        server = MockServer()
        server.when("hi").reply("hello world")
        with TestClient(server._app) as client:
            # 显式 stream=True → SSE 流式
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"].lower()

            body = resp.text
            # SSE 以 data: 开头的行构成消息；断言至少有一个 delta 和一个 [DONE]
            assert "data:" in body
            # hello world 至少在某条 delta.content 里
            assert "hello world" in body or '"content":' in body
            # 收尾必须有 [DONE]
            assert "[DONE]" in body

    def test_rule_times_2_consumed_after_two_requests(self) -> None:
        """times(2) 注册的规则，只应在前两次命中时生效；第三次走 fallback。"""
        server = MockServer()
        server.when("ping").reply("pong").times(2)
        with TestClient(server._app) as client:
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            }
            r1 = client.post("/v1/chat/completions", json=payload)
            r2 = client.post("/v1/chat/completions", json=payload)
            r3 = client.post("/v1/chat/completions", json=payload)
            assert "pong" in r1.json()["choices"][0]["message"]["content"]
            assert "pong" in r2.json()["choices"][0]["message"]["content"]
            # 第三次：fallback
            assert "Mock server: no matching rule." in r3.json()["choices"][0]["message"]["content"]

    def test_first_registered_rule_wins_when_both_match(self) -> None:
        """两条规则都可能命中时，按注册顺序优先走第一条。"""
        server = MockServer()
        server.when("hello").reply("first-reply")
        server.when("hello world").reply("second-reply")
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "hello world"}],
                    "stream": False,
                },
            )
            # "hello" 先注册，字符串匹配器"hello"能同时命中 "hello world"
            # 按 engine 的顺序匹配策略应返回 first-reply
            assert "first-reply" in resp.json()["choices"][0]["message"]["content"]

    def test_next_error_fires_once_then_removed(self) -> None:
        """next_error(500, ...) 只触发一次；第二次请求走正常规则或 fallback。"""
        server = MockServer()
        server.next_error(500, "boom", "server_error")
        server.when("ping").reply("pong")  # 再加一条兜底规则
        with TestClient(server._app) as client:
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
            }
            r1 = client.post("/v1/chat/completions", json=payload)
            # next_error 注册的是 error 规则，应返回非 200
            assert r1.status_code == 500
            r2 = client.post("/v1/chat/completions", json=payload)
            # 第二次：next_error 已消耗，走正常规则
            assert r2.status_code == 200
            assert "pong" in r2.json()["choices"][0]["message"]["content"]

    def test_history_records_every_request(self) -> None:
        """每一次 HTTP 请求都应被记入 server.history。"""
        server = MockServer()
        server.when("ping").reply("pong")
        with TestClient(server._app) as client:
            for _ in range(3):
                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": False,
                    },
                )
        assert server.history.count() >= 3

    def test_reply_object_structured(self) -> None:
        """ReplyObject(text=...) 走结构化回复路径，仍应返回 text 到 content。"""
        server = MockServer()
        server.when("info").reply(ReplyObject(text="structured reply"))
        with TestClient(server._app) as client:
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "info"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        assert "structured reply" in resp.json()["choices"][0]["message"]["content"]


# ======================================================================
# 6. load() 规则文件加载
# ======================================================================


class TestLoadRules:
    @pytest.fixture
    def tmp_json5(self, tmp_path: pathlib.Path) -> pathlib.Path:
        f = tmp_path / "rules.json5"
        f.write_text(
            """
            [
              { when: "greet", reply: "hello!" },
              { when: "bye", reply: "see you" }
            ]
            """,
            encoding="utf-8",
        )
        return f

    def test_load_from_file_registers_rules(self, tmp_json5: pathlib.Path) -> None:
        server = MockServer()
        asyncio.run(server.load(str(tmp_json5)))
        assert server.rule_count == 2
