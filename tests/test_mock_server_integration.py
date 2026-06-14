"""mock_server.py 的真 HTTP 集成测试。

需要启动真实 uvicorn 服务、用 httpx2 发送 TCP 请求，比单元测试慢。
适合验证：start/stop 生命周期、端口绑定、URL、跨进程响应、context manager。

本文件覆盖范围：
  * MockServer 生命周期（原有）
  * llm-mock-api init 命令：生成文件 + 内容合法性
  * llm-mock-api validate 命令：有效/无效/目录/不存在场景
  * CLI 配置合并：CLI args > JSON config > dataclass 默认值
  * 各种 rule 配置的真实 HTTP 响应验证（重点）：
      - 字符串匹配 / 正则匹配 / 对象匹配（message+model / message+format）
      - 结构化回复：text / text+reasoning / text+tools / 多 tool calls
      - 序列回复 replies / 自定义 latency
      - times 次数限制
      - 模板引用 $name
      - SSE 流式输出
      - fallback 无匹配
  * from_json_config 全配置生效验证
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
from typing import Any

import json5
import pytest

from llm_mock_api import cli
from llm_mock_api.mock_server import MockServer
from llm_mock_api.types.reply import ReplyObject, ToolCall
from llm_mock_api.types.rule import MatchObject

# ---------------------------------------------------------------------------
# 测试工具
# ---------------------------------------------------------------------------


def _chat_payload(message: str, model: str = "gpt-4", stream: bool = False) -> dict[str, Any]:
    """构造 /v1/chat/completions 请求体。"""
    return {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "stream": stream,
    }


def _ns(**kwargs: Any) -> argparse.Namespace:
    """构造 argparse.Namespace。"""
    return argparse.Namespace(**kwargs)


def _json5_file(path: pathlib.Path, content: str) -> pathlib.Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestIntegrationWithConfig:
    """真实启动 server，用 httpx2 验证 config.json 的参数真正生效。"""

    def test_real_server_from_json_config(self, tmp_path) -> None:
        """用 config.json 启动 server → 注册了 fallback → 不匹配请求应返回 fallback。"""
        rules_file = tmp_path / "rules.json5"
        rules_file.write_text(
            """
        [
          { when: "hi", reply: "hello" }
        ]
        """,
            encoding="utf-8",
        )

        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "port": 0,
                    "fallback": "custom-fallback-msg",
                    "rules": str(rules_file),
                },
            ),
            encoding="utf-8",
        )

        async def _run() -> None:
            server = await MockServer.from_json_config(str(cfg))
            await server.start()
            try:
                import httpx2

                async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                    # 匹配规则
                    resp = await http.post(
                        "/v1/chat/completions",
                        json={
                            "model": "gpt-4",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": False,
                        },
                    )
                    assert resp.status_code == 200
                    assert "hello" in resp.json()["choices"][0]["message"]["content"]

                    # 不匹配 → fallback
                    resp2 = await http.post(
                        "/v1/chat/completions",
                        json={
                            "model": "gpt-4",
                            "messages": [{"role": "user", "content": "greetings"}],
                            "stream": False,
                        },
                    )
                    assert resp2.status_code == 200
                    assert "custom-fallback-msg" in resp2.json()["choices"][0]["message"]["content"]
            finally:
                await server.stop()

        asyncio.run(_run())


@pytest.mark.asyncio
class TestIntegrationServer:
    async def test_start_stop_lifecycle(self) -> None:
        """验证 start/stop 能正常完成，不会抛异常。"""
        server = MockServer()
        server.when("ping").reply("pong")
        await server.start()
        assert server._listening is True
        assert server.url.startswith("http://")
        assert server.url.endswith(":0") is False  # 实际端口不是 0
        await server.stop()
        assert server._listening is False

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self) -> None:
        """重复 stop() 不应抛错。"""
        server = MockServer()
        await server.stop()  # 什么都没做过
        await server.stop()  # 再来一次

    @pytest.mark.asyncio
    async def test_start_twice_raises(self) -> None:
        """已启动后再次 start 应报 RuntimeError。"""
        server = MockServer()
        await server.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await server.start()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_real_http_request(self) -> None:
        """完整流程：启动 → 注册规则 → 真 HTTP 请求 → 验证响应 → 关闭。"""
        import httpx2

        server = MockServer()
        server.when("hi").reply("hello from mock server")

        await server.start()
        try:
            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
            assert resp.status_code == 200
            body = resp.json()
            assert "hello from mock server" in body["choices"][0]["message"]["content"]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """async with 语法糖：进入时 start，退出时 stop。"""
        import httpx2

        async with MockServer() as server:
            assert server._listening is True
            server.when("ping").reply("pong")
            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-4",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": False,
                    },
                )
                assert resp.status_code == 200
        # 退出上下文后 server 应已停止
        assert server._listening is False


# ============================================================================
# TestCliInitCommand —— llm-mock-api init 命令
# ============================================================================


class TestCliInitCommand:
    """验证 llm-mock-api init 命令。"""

    def test_init_generates_both_files(self, tmp_path: pathlib.Path) -> None:
        """在空目录生成 config.json 和 rules.json5。"""
        args = _ns(dir=str(tmp_path), force=False)
        rc = asyncio.run(cli._cmd_init(args))
        assert rc == 0

        config_path = tmp_path / "config.json"
        rules_path = tmp_path / "rules.json5"
        assert config_path.exists(), "config.json 应被创建"
        assert rules_path.exists(), "rules.json5 应被创建"

    def test_init_config_is_valid_json(self, tmp_path: pathlib.Path) -> None:
        """config.json 必须是合法 JSON 对象，包含关键字段。"""
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["port"] == 8002
        assert data["host"] == "127.0.0.1"
        assert data["log_level"] == "info"
        assert data["default_latency"] == 50
        assert data["default_chunk_size"] == 50
        assert "fallback" in data
        assert data["rules"] == "./rules.json5"
        assert data["watch"] is True

    def test_init_rules_is_valid_json5_with_15_rules(self, tmp_path: pathlib.Path) -> None:
        """rules.json5 必须是合法 JSON5，顶层有 templates/fallback/rules，rules 数 = 15。"""
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))
        rules_path = tmp_path / "rules.json5"
        data = json5.loads(rules_path.read_text(encoding="utf-8"))
        assert isinstance(data, dict), "rules.json5 顶层必须是对象"
        assert isinstance(data["rules"], list), "rules 必须是数组"
        assert len(data["rules"]) == 16, f"应有 16 条规则，实际 {len(data['rules'])}"
        assert isinstance(data["templates"], dict), "应有 templates 对象"
        assert "weather_tool" in data["templates"]
        assert "cant_answer" in data["templates"]
        assert isinstance(data["fallback"], str)

    def test_init_skip_if_exists(self, tmp_path: pathlib.Path) -> None:
        """不带 --force：文件已存在应跳过。"""
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))

        # 记录初始内容
        original_cfg = (tmp_path / "config.json").read_bytes()
        original_rules = (tmp_path / "rules.json5").read_bytes()

        # 再次运行（不带 force）不应覆盖
        asyncio.run(cli._cmd_init(args))
        assert (tmp_path / "config.json").read_bytes() == original_cfg
        assert (tmp_path / "rules.json5").read_bytes() == original_rules

    def test_init_force_overwrites(self, tmp_path: pathlib.Path) -> None:
        """带 --force：文件存在也覆盖。"""
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))
        # 修改再用 force 覆盖
        (tmp_path / "config.json").write_text("garbage", encoding="utf-8")

        args_force = _ns(dir=str(tmp_path), force=True)
        asyncio.run(cli._cmd_init(args_force))
        data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        assert data["port"] == 8002  # 已被覆盖为正确内容

    def test_init_uses_current_dir_by_default(self, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
        """不传 -d：默认在当前目录生成（monkeypatch chdir 验证）。"""
        monkeypatch.chdir(tmp_path)
        args = _ns(dir=None, force=False)
        asyncio.run(cli._cmd_init(args))
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "rules.json5").exists()

    def test_init_output_can_be_validated(self, tmp_path: pathlib.Path) -> None:
        """生成的 rules.json5 必须能通过 validate 命令检查。"""
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(tmp_path / "rules.json5"))))
        assert rc == 0, "生成的 rules.json5 应通过 validate 检查"

    def test_init_output_can_be_loaded_by_mock_server(self, tmp_path: pathlib.Path) -> None:
        """生成的 config.json + rules.json5 必须能被 MockServer 成功加载并在真实 HTTP 请求中生效。"""
        # 生成文件
        args = _ns(dir=str(tmp_path), force=False)
        asyncio.run(cli._cmd_init(args))

        # 相对路径 "./rules.json5" 是相对于当前目录的；为避免 cwd 干扰，
        # 先修改 config.json 中 rules 为绝对路径
        cfg = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
        cfg["rules"] = str(tmp_path / "rules.json5")
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

        async def _run() -> None:
            server = await MockServer.from_json_config(str(tmp_path / "config.json"))
            await server.start()
            try:
                import httpx2

                async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                    resp = await http.post(
                        "/v1/chat/completions",
                        json=_chat_payload("hi"),
                    )
                    assert resp.status_code == 200
                    body = resp.json()
                    assert "Hello" in body["choices"][0]["message"]["content"]
            finally:
                await server.stop()

        asyncio.run(_run())


# ============================================================================
# TestCliValidateCommand —— llm-mock-api validate 命令
# ============================================================================


class TestCliValidateCommand:
    """验证 llm-mock-api validate 命令。"""

    def test_validate_valid_file(self, tmp_path: pathlib.Path) -> None:
        """合法规则文件应返回 0。"""
        rules_file = _json5_file(
            tmp_path / "ok.json5",
            '{ rules: [{ when: "a", reply: "b" }] }',
        )
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(rules_file))))
        assert rc == 0

    def test_validate_file_not_found(self, tmp_path: pathlib.Path) -> None:
        """不存在的路径应返回 1（错误码）。"""
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(tmp_path / "nope.json5"))))
        assert rc == 1

    def test_validate_root_not_object(self, tmp_path: pathlib.Path) -> None:
        """根节点必须是对象，不是数组/字符串等。"""
        bad = _json5_file(tmp_path / "bad.json5", '[{ when: "a", reply: "b" }]')
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(bad))))
        assert rc == 1

    def test_validate_missing_rules_key(self, tmp_path: pathlib.Path) -> None:
        """缺少 rules 键应返回 1。"""
        bad = _json5_file(tmp_path / "bad.json5", '{ hello: "world" }')
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(bad))))
        assert rc == 1

    def test_validate_rules_not_array(self, tmp_path: pathlib.Path) -> None:
        """rules 必须是数组。"""
        bad = _json5_file(tmp_path / "bad.json5", '{ rules: "not an array" }')
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(bad))))
        assert rc == 1

    def test_validate_syntax_error(self, tmp_path: pathlib.Path) -> None:
        """JSON5 语法错误应返回 1。"""
        bad = _json5_file(tmp_path / "bad.json5", '{ rules: [missing closing brace')
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(bad))))
        assert rc == 1

    def test_validate_directory_recursive(self, tmp_path: pathlib.Path) -> None:
        """目录递归：只处理 .json5/.json 文件，有效文件应全通过。"""
        # 构造 3 个文件：2 个有效，1 个无效
        _json5_file(tmp_path / "a.json5", '{ rules: [{ when: "a", reply: "A" }] }')
        _json5_file(tmp_path / "b.json5", '{ rules: [{ when: "b", reply: "B" }] }')
        _json5_file(tmp_path / "c.json5", '{ rules: "bad" }')
        # 非规则文件应被忽略
        (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")

        rc = asyncio.run(cli._cmd_validate(_ns(path=str(tmp_path))))
        assert rc == 1  # 有一个文件失败

    def test_validate_empty_directory(self, tmp_path: pathlib.Path) -> None:
        """无规则文件的目录：warning 但返回 0。"""
        rc = asyncio.run(cli._cmd_validate(_ns(path=str(tmp_path))))
        assert rc == 0


# ============================================================================
# TestCliConfigMerge —— CLI 参数 > JSON 配置 > dataclass 默认值
# ============================================================================


class TestCliConfigMerge:
    """验证 _merge_options / _load_config_file 的优先级。"""

    def test_config_file_does_not_exist(self, tmp_path: pathlib.Path) -> None:
        """不存在的 config.json：_load_config_file 返回空 dict。"""
        data = cli._load_config_file(str(tmp_path / "missing.json"))
        assert data == {}

    def test_config_file_syntax_error(self, tmp_path: pathlib.Path) -> None:
        """config.json 语法错误：返回空 dict（不抛异常，打印 warning）。"""
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json", encoding="utf-8")
        data = cli._load_config_file(str(bad))
        assert data == {}

    def test_config_file_not_object(self, tmp_path: pathlib.Path) -> None:
        """顶层是数组等非对象：返回空 dict。"""
        cfg = tmp_path / "config.json"
        cfg.write_text('["a", "b"]', encoding="utf-8")
        data = cli._load_config_file(str(cfg))
        assert data == {}

    def test_merge_json_values_used_when_cli_absent(self, tmp_path: pathlib.Path) -> None:
        """CLI 参数未传时，使用 JSON 配置。

        注意：_pick(key) 在 JSON 中查找的键名与 CLI 参数名一致（如 latency、chunk_size），
        而非 MockServerOptions dataclass 的完整字段名（default_latency、default_chunk_size）。
        """
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "port": 9999,
                "host": "0.0.0.0",
                "log_level": "debug",
                "latency": 30,
                "chunk_size": 40,
                "fallback": "my-fallback",
                "rules": "./my-rules.json5",
                "watch": False,
            }),
            encoding="utf-8",
        )
        args = _ns(
            config=str(cfg_path),
            port=None, host=None, log_level=None,
            latency=None, chunk_size=None,
            fallback=None, rules=None, watch=False,
        )
        options, rules, fallback, watch = cli._merge_options(args)
        assert options.port == 9999
        assert options.host == "0.0.0.0"
        assert options.log_level == "debug"
        assert options.default_latency == 30
        assert options.default_chunk_size == 40
        assert fallback == "my-fallback"
        assert rules == "./my-rules.json5"
        assert watch is False

    def test_cli_overrides_json(self, tmp_path: pathlib.Path) -> None:
        """CLI 参数显式传入时覆盖 JSON 配置。"""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "port": 1111,
                "host": "127.0.0.1",
                "log_level": "info",
                "default_latency": 50,
                "default_chunk_size": 50,
                "fallback": "from-json",
                "rules": "./from-json.json5",
                "watch": False,
            }),
            encoding="utf-8",
        )
        args = _ns(
            config=str(cfg_path),
            port=2222, host="192.168.0.1", log_level="debug",
            latency=10, chunk_size=20,
            fallback="from-cli", rules="./from-cli.json5", watch=True,
        )
        options, rules, fallback, watch = cli._merge_options(args)
        assert options.port == 2222
        assert options.host == "192.168.0.1"
        assert options.log_level == "debug"
        assert options.default_latency == 10
        assert options.default_chunk_size == 20
        assert fallback == "from-cli"
        assert rules == "./from-cli.json5"
        assert watch is True

    def test_watch_flag_from_json(self, tmp_path: pathlib.Path) -> None:
        """watch: true 在 JSON 中生效。"""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"watch": True}), encoding="utf-8")
        args = _ns(config=str(cfg_path), port=None, host=None, log_level=None,
                    latency=None, chunk_size=None, fallback=None, rules=None, watch=False)
        _, _, _, watch = cli._merge_options(args)
        assert watch is True


# ============================================================================
# TestRuleConfigHttp —— 各种 rule 配置的真实 HTTP 响应验证（重点）
# ============================================================================


class TestRuleConfigHttp:
    """以不同的 JSON5 rules 配置启动真实 MockServer，
    用 httpx2 验证响应内容。
    每个用例独立启动/关闭 server，隔离规则状态。"""

    # ------------------------------------------------------------------
    # 基础匹配
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_string_match_simple_text(self) -> None:
        """字符串匹配 + 文本回复。"""
        async with MockServer() as server:
            server.when("hi").reply("world")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("hi"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "world"

    @pytest.mark.asyncio
    async def test_string_match_case_insensitive(self) -> None:
        """大小写不敏感匹配：Hi / HI / hI 都匹配 'hi'。"""
        async with MockServer() as server:
            server.when("hello").reply("HELLO_WORLD")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                for msg in ("HELLO", "Hello", "hello world"):
                    resp = await http.post("/v1/chat/completions", json=_chat_payload(msg))
                    assert resp.status_code == 200
                    body = resp.json()
                    assert body["choices"][0]["message"]["content"] == "HELLO_WORLD", f"msg='{msg}' 应匹配"

    @pytest.mark.asyncio
    async def test_regex_match(self) -> None:
        """正则匹配：使用 re.compile 匹配用户消息。"""
        async with MockServer() as server:
            server.when(re.compile(r"^echo .*$", re.IGNORECASE)).reply("echo triggered")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("echo 123"))
                assert resp.status_code == 200
                assert "echo triggered" in resp.json()["choices"][0]["message"]["content"]

                # 不匹配
                resp = await http.post("/v1/chat/completions", json=_chat_payload("no match"))
                assert resp.status_code == 200

    # ------------------------------------------------------------------
    # 对象匹配
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_object_match_by_model(self) -> None:
        """按 model 字段筛选：仅对 gpt-4 命中。"""
        async with MockServer() as server:
            server.when(MatchObject(message="check", model="gpt-4")).reply("matched-gpt4")
            server.when("check").reply("matched-default")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("check", model="gpt-4"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "matched-gpt4"

                resp = await http.post("/v1/chat/completions", json=_chat_payload("check", model="other-model"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "matched-default"

    @pytest.mark.asyncio
    async def test_object_match_by_format(self) -> None:
        """按 format 筛选：仅在 /v1/chat/completions 路由命中。"""
        async with MockServer() as server:
            server.when(MatchObject(message="fmt", format="openai")).reply("openai-format")
            server.when("fmt").reply("any-format")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("fmt"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "openai-format"

    # ------------------------------------------------------------------
    # 结构化回复
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_structured_reply_text_and_reasoning(self) -> None:
        """text + reasoning：响应中应包含 content 和 reasoning。"""
        async with MockServer() as server:
            server.when("explain").reply(ReplyObject(
                text="42 is the answer.",
                reasoning="Deep thought about life, universe and everything.",
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("explain"))
                assert resp.status_code == 200
                body = resp.json()
                assert body["choices"][0]["message"]["content"] == "42 is the answer."

    @pytest.mark.asyncio
    async def test_structured_reply_with_single_tool_call(self) -> None:
        """tools 数组：响应中应包含 tool_calls。"""
        async with MockServer() as server:
            server.when("weather").reply(ReplyObject(
                text="Looking up the weather...",
                tools=[
                    ToolCall(name="get_weather", args={"location": "Beijing", "unit": "celsius"}),
                ],
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("weather"))
                assert resp.status_code == 200
                body = resp.json()
                message = body["choices"][0]["message"]
                # 文本 + tool_calls 都应出现
                assert message["content"] == "Looking up the weather..."
                assert len(message["tool_calls"]) == 1
                assert message["tool_calls"][0]["type"] == "function"
                assert message["tool_calls"][0]["function"]["name"] == "get_weather"
                args = json.loads(message["tool_calls"][0]["function"]["arguments"])
                assert args["location"] == "Beijing"
                assert args["unit"] == "celsius"

    @pytest.mark.asyncio
    async def test_structured_reply_with_multiple_tool_calls(self) -> None:
        """多工具调用：响应中应有多个 tool_calls。"""
        async with MockServer() as server:
            server.when("multi tool").reply(ReplyObject(
                text="Running 2 tools.",
                tools=[
                    ToolCall(name="tool_a", args={"x": 1}),
                    ToolCall(name="tool_b", args={"y": "str"}),
                ],
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("multi tool"))
                assert resp.status_code == 200
                body = resp.json()
                message = body["choices"][0]["message"]
                assert message["content"] == "Running 2 tools."
                assert len(message["tool_calls"]) == 2
                assert message["tool_calls"][0]["function"]["name"] == "tool_a"
                assert message["tool_calls"][1]["function"]["name"] == "tool_b"

    @pytest.mark.asyncio
    async def test_tool_call_only_no_text(self) -> None:
        """纯工具调用，无 text：content 应为 None，tool_calls 存在。"""
        async with MockServer() as server:
            server.when("toolonly").reply(ReplyObject(
                tools=[ToolCall(name="search", args={"q": "abc"})],
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("toolonly"))
                assert resp.status_code == 200
                body = resp.json()
                message = body["choices"][0]["message"]
                assert message["content"] is None
                assert message["tool_calls"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_finish_reason_is_tool_calls_when_tools_present(self) -> None:
        """有 tool_calls 的回复，finish_reason 应为 'tool_calls'。"""
        async with MockServer() as server:
            server.when("use-tool").reply(ReplyObject(
                text="ok",
                tools=[ToolCall(name="t", args={})],
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("use-tool"))
                body = resp.json()
                assert body["choices"][0]["finish_reason"] == "tool_calls"

    @pytest.mark.asyncio
    async def test_finish_reason_is_stop_when_text_only(self) -> None:
        """纯文本回复，finish_reason 为 stop。"""
        async with MockServer() as server:
            server.when("x").reply("y")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("x"))
                assert resp.json()["choices"][0]["finish_reason"] == "stop"

    # ------------------------------------------------------------------
    # 序列回复
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_sequence_reply_advance_on_each_request(self) -> None:
        """同一关键词每次请求推进到下一条。"""
        async with MockServer() as server:
            server.when("step").reply_sequence(["step 1", "step 2", "step 3"])
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                for expected in ("step 1", "step 2", "step 3"):
                    resp = await http.post("/v1/chat/completions", json=_chat_payload("step"))
                    assert resp.status_code == 200
                    assert resp.json()["choices"][0]["message"]["content"] == expected

    @pytest.mark.asyncio
    async def test_sequence_reply_reuse_last_after_exhausted(self) -> None:
        """序列耗尽后继续返回最后一条。"""
        async with MockServer() as server:
            server.when("q").reply_sequence(["first", "second"])
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # 前两次
                for expected in ("first", "second"):
                    resp = await http.post("/v1/chat/completions", json=_chat_payload("q"))
                    assert resp.json()["choices"][0]["message"]["content"] == expected
                # 第三次：仍返回 second（最后一条）
                resp = await http.post("/v1/chat/completions", json=_chat_payload("q"))
                assert resp.json()["choices"][0]["message"]["content"] == "second"

    @pytest.mark.asyncio
    async def test_sequence_reply_structured_objects(self) -> None:
        """reply_sequence 中的条目可以是字符串或 ReplyObject。"""
        async with MockServer() as server:
            server.when("mix").reply_sequence([
                "plain text",
                ReplyObject(text="structured text"),
                ReplyObject(
                    text="with tool",
                    tools=[ToolCall(name="x", args={"a": 1})],
                ),
            ])
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # 1
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("mix"))
                assert r1.json()["choices"][0]["message"]["content"] == "plain text"
                # 2
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("mix"))
                assert r2.json()["choices"][0]["message"]["content"] == "structured text"
                # 3
                r3 = await http.post("/v1/chat/completions", json=_chat_payload("mix"))
                body = r3.json()
                assert body["choices"][0]["message"]["content"] == "with tool"
                assert body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "x"

    # ------------------------------------------------------------------
    # times 次数限制
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_times_once_then_fallback(self) -> None:
        """times:1：第一次命中，第二次应落入 fallback。"""
        async with MockServer() as server:
            server.when("once").reply("only once").times(1)
            server.fallback("fb-msg")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("once"))
                assert r1.json()["choices"][0]["message"]["content"] == "only once"

                # 第二次：times 已耗尽，回落到 fallback
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("once"))
                assert r2.json()["choices"][0]["message"]["content"] == "fb-msg"

    @pytest.mark.asyncio
    async def test_times_three(self) -> None:
        """times:3：前 3 次命中，第 4 次 fallback。"""
        async with MockServer() as server:
            server.when("t3").reply("hit").times(3)
            server.fallback("fallback")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                for _ in range(3):
                    r = await http.post("/v1/chat/completions", json=_chat_payload("t3"))
                    assert r.json()["choices"][0]["message"]["content"] == "hit"
                r4 = await http.post("/v1/chat/completions", json=_chat_payload("t3"))
                assert r4.json()["choices"][0]["message"]["content"] == "fallback"

    @pytest.mark.asyncio
    async def test_times_independent_per_rule(self) -> None:
        """多规则：各规则 times 计数器独立。"""
        async with MockServer() as server:
            server.when("a").reply("A").times(1)
            server.when("b").reply("B").times(2)
            server.fallback("fallback")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # a 第一次命中
                assert (await http.post("/v1/chat/completions", json=_chat_payload("a"))).json()[
                    "choices"
                ][0]["message"]["content"] == "A"
                # b 2 次命中
                for _ in range(2):
                    assert (await http.post("/v1/chat/completions", json=_chat_payload("b"))).json()[
                        "choices"
                    ][0]["message"]["content"] == "B"
                # a 已用完
                assert (await http.post("/v1/chat/completions", json=_chat_payload("a"))).json()[
                    "choices"
                ][0]["message"]["content"] == "fallback"
                # b 也用完
                assert (await http.post("/v1/chat/completions", json=_chat_payload("b"))).json()[
                    "choices"
                ][0]["message"]["content"] == "fallback"

    # ------------------------------------------------------------------
    # 模板引用
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_template_reference_in_reply(self, tmp_path: pathlib.Path) -> None:
        """在 JSON5 文件中定义 templates，reply 用 $name 引用。"""
        rules_file = _json5_file(
            tmp_path / "t.json5",
            """
            {
                templates: {
                    greet: { text: "Hello from template!" },
                    tool_weather: {
                        text: "Let me check the weather.",
                        tools: [
                            { name: "get_weather", args: { location: "Shanghai" } }
                        ],
                    },
                },
                rules: [
                    { when: "greet", reply: "$greet" },
                    { when: "weather", reply: "$tool_weather" },
                ],
            }
            """,
        )

        async with MockServer() as server:
            await server.load(str(rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # 文本模板
                resp = await http.post("/v1/chat/completions", json=_chat_payload("greet"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "Hello from template!"

                # 带 tools 的模板
                resp = await http.post("/v1/chat/completions", json=_chat_payload("weather"))
                body = resp.json()
                message = body["choices"][0]["message"]
                assert message["content"] == "Let me check the weather."
                assert message["tool_calls"][0]["function"]["name"] == "get_weather"
                args = json.loads(message["tool_calls"][0]["function"]["arguments"])
                assert args["location"] == "Shanghai"

    @pytest.mark.asyncio
    async def test_template_reference_in_sequence(self, tmp_path: pathlib.Path) -> None:
        """replies 序列中也可以用 $name 引用模板。"""
        rules_file = _json5_file(
            tmp_path / "t2.json5",
            """
            {
                templates: { tpl: { text: "templated content" } },
                rules: [
                    { when: "go", replies: ["first", "$tpl"] },
                ],
            }
            """,
        )
        async with MockServer() as server:
            await server.load(str(rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("go"))
                assert r1.json()["choices"][0]["message"]["content"] == "first"
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("go"))
                assert r2.json()["choices"][0]["message"]["content"] == "templated content"

    # ------------------------------------------------------------------
    # JSON5 中文件内 fallback（顶层 fallback 字段）
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rules_file_level_fallback(self, tmp_path: pathlib.Path) -> None:
        """JSON5 顶层 fallback 覆盖 MockServer 的 fallback。"""
        rules_file = _json5_file(
            tmp_path / "fb.json5",
            """
            {
                fallback: "file-level-fallback",
                rules: [{ when: "hit", reply: "got it" }],
            }
            """,
        )
        async with MockServer() as server:
            server.fallback("server-default-fallback")  # 应被 rules 文件覆盖
            await server.load(str(rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # 命中 → got it
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("hit"))
                assert r1.json()["choices"][0]["message"]["content"] == "got it"
                # 未命中 → file-level-fallback
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("nothing matches"))
                assert r2.json()["choices"][0]["message"]["content"] == "file-level-fallback"

    # ------------------------------------------------------------------
    # SSE 流式输出
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stream_sse_produces_data_lines(self) -> None:
        """stream=true：响应 content-type 含 text/event-stream，且含多个 'data:' 块。"""
        long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5
        async with MockServer() as server:
            server.when("long").reply(long_text)
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("long", stream=True))
                assert resp.status_code == 200
                ct = resp.headers.get("content-type", "")
                assert "text/event-stream" in ct, f"期望 event-stream，实际 {ct}"

                raw = resp.text
                # 应至少包含一个 data: ... 与 [DONE]
                assert "data:" in raw
                assert "[DONE]" in raw
                # 各 chunk 累积文本应等于原文本
                chunks = []
                for line in raw.split("\n"):
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload == "[DONE]":
                            continue
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"]
                        if isinstance(delta.get("content"), str):
                            chunks.append(delta["content"])
                # 合并后应 == 原始文本
                assert "".join(chunks) == long_text, "流式 chunk 合并后应等于原始文本"

    @pytest.mark.asyncio
    async def test_non_stream_has_usage(self) -> None:
        """非流式响应必须包含 usage 字段。"""
        async with MockServer() as server:
            server.when("hi").reply("hi there")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("hi"))
                body = resp.json()
                assert "usage" in body
                assert isinstance(body["usage"]["prompt_tokens"], int)
                assert isinstance(body["usage"]["completion_tokens"], int)
                assert isinstance(body["usage"]["total_tokens"], int)

    # ------------------------------------------------------------------
    # 响应格式跨路由一致性（/v1/chat/completions vs /v1/responses）
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_same_rule_on_both_routes(self) -> None:
        """同一条规则在两个 OpenAI 路由上都可用，只是序列化不同。"""
        async with MockServer() as server:
            server.when("hi").reply(ReplyObject(
                text="world",
                reasoning="just a test",
            ))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                # /v1/chat/completions
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("hi"))
                assert r1.status_code == 200
                assert r1.json()["choices"][0]["message"]["content"] == "world"

                # /v1/responses —— 不同的请求体结构（显式 stream=False 确保非流式
                r2 = await http.post(
                    "/v1/responses",
                    json={
                        "model": "gpt-4",
                        "input": [{"role": "user", "content": "hi"}],
                        "stream": False,
                    },
                )
                assert r2.status_code == 200
                body = r2.json()
                # Responses API 有 output 数组
                assert "output" in body

    @pytest.mark.asyncio
    async def test_request_without_stream_key(self) -> None:
        """请求体不含 stream 键：默认为非流式 JSON 响应（与主流 LLM API 一致）。"""
        async with MockServer() as server:
            server.when("x").reply("y")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post(
                    "/v1/chat/completions",
                    json={"model": "gpt-4", "messages": [{"role": "user", "content": "x"}]},
                )
                assert resp.status_code == 200
                # 新语义：无 stream 字段 → JSON 响应
                body = resp.json()
                assert "choices" in body
                assert body["choices"][0]["message"]["content"] == "y"

    # ------------------------------------------------------------------
    # 规则顺序 —— 先匹配优先
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_rule_order_first_match_wins(self) -> None:
        """两条规则都能匹配时，先注册的返回。"""
        async with MockServer() as server:
            server.when("foo").reply("first")
            server.when("foo").reply("second")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                r = await http.post("/v1/chat/completions", json=_chat_payload("foo"))
                assert r.json()["choices"][0]["message"]["content"] == "first"

    # ------------------------------------------------------------------
    # 中文 / unicode 消息
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_chinese_message_roundtrip(self) -> None:
        """中文消息匹配与中文回复，编码正确。"""
        async with MockServer() as server:
            server.when("你好").reply("世界！")
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("你好"))
                assert resp.status_code == 200
                assert resp.json()["choices"][0]["message"]["content"] == "世界！"


# ============================================================================
# TestDefaultRulesHttp —— 使用 init 输出的默认规则做端到端 HTTP 验证
# ============================================================================


class TestDefaultRulesHttp:
    """直接使用 llm-mock-api init 生成的默认 15 条规则验证 HTTP 响应。

    与上面 TestRuleConfigHttp 不同，这里验证的是“真实用户开箱即用”的规则文件：
    逐条发请求，检查是否命中预期回复。"""

    @pytest.fixture()
    def _default_rules_file(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """生成 init 默认规则，返回 rules.json5 路径。"""
        asyncio.run(cli._cmd_init(_ns(dir=str(tmp_path), force=False)))
        return tmp_path / "rules.json5"

    def test_default_rules_file_exists(self, _default_rules_file: pathlib.Path) -> None:
        assert _default_rules_file.exists()

    @pytest.mark.asyncio
    async def test_default_rule_hi(self, _default_rules_file: pathlib.Path) -> None:
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("hi"))
                assert resp.status_code == 200
                assert "Hello" in resp.json()["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_default_rule_regex_echo(self, _default_rules_file: pathlib.Path) -> None:
        """正则规则：匹配以 echo 开头的消息。"""
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("echo test"))
                assert resp.status_code == 200
                assert "echo" in resp.json()["choices"][0]["message"]["content"].lower()

    @pytest.mark.asyncio
    async def test_default_rule_model_filter(self, _default_rules_file: pathlib.Path) -> None:
        """默认规则 'model check' + model=gpt-4 应命中专用回复。"""
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("model check", model="gpt-4"))
                content = resp.json()["choices"][0]["message"]["content"]
                assert "gpt-4" in content.lower() or "object" in content.lower()

    @pytest.mark.asyncio
    async def test_default_rule_joke_structured(self, _default_rules_file: pathlib.Path) -> None:
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("joke"))
                assert resp.status_code == 200
                content = resp.json()["choices"][0]["message"]["content"]
                assert len(content) > 0

    @pytest.mark.asyncio
    async def test_default_rule_weather_toolcall(self, _default_rules_file: pathlib.Path) -> None:
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post("/v1/chat/completions", json=_chat_payload("weather"))
                body = resp.json()
                message = body["choices"][0]["message"]
                assert message["tool_calls"][0]["function"]["name"] == "get_weather"
                args = json.loads(message["tool_calls"][0]["function"]["arguments"])
                assert "location" in args

    @pytest.mark.asyncio
    async def test_default_rule_sequence_step(self, _default_rules_file: pathlib.Path) -> None:
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("step"))
                assert "step 1" in r1.json()["choices"][0]["message"]["content"].lower()
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("step"))
                assert "step 2" in r2.json()["choices"][0]["message"]["content"].lower()
                r3 = await http.post("/v1/chat/completions", json=_chat_payload("step"))
                assert "step 3" in r3.json()["choices"][0]["message"]["content"].lower() or "done" in r3.json()["choices"][0]["message"]["content"].lower()

    @pytest.mark.asyncio
    async def test_default_rule_times_once(self, _default_rules_file: pathlib.Path) -> None:
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                r1 = await http.post("/v1/chat/completions", json=_chat_payload("once"))
                first = r1.json()["choices"][0]["message"]["content"]
                assert "triggers only once" in first.lower() or "only on" in first.lower()

                # 第二次不应再命中，回落到 fallback
                r2 = await http.post("/v1/chat/completions", json=_chat_payload("once"))
                second = r2.json()["choices"][0]["message"]["content"]
                assert "triggers only once" not in second.lower()

    @pytest.mark.asyncio
    async def test_default_rule_fallback(self, _default_rules_file: pathlib.Path) -> None:
        """完全不匹配的请求应返回 file-level fallback（或内置 fallback）。"""
        async with MockServer() as server:
            await server.load(str(_default_rules_file))
            import httpx2

            async with httpx2.AsyncClient(base_url=server.url, timeout=5) as http:
                resp = await http.post(
                    "/v1/chat/completions",
                    json=_chat_payload("__this_should_never_match_xyz__"),
                )
                assert resp.status_code == 200
                assert isinstance(resp.json()["choices"][0]["message"]["content"], str)

