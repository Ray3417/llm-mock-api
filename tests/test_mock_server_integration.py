"""mock_server.py 的真 HTTP 集成测试。

需要启动真实 uvicorn 服务、用 httpx2 发送 TCP 请求，比单元测试慢。
适合验证：start/stop 生命周期、端口绑定、URL、跨进程响应、context manager。
"""

from __future__ import annotations

import asyncio
import json

import pytest

from llm_mock_api.mock_server import MockServer


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
            await server.start(port=0)
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
        await server.start(port=0)
        assert server._listening is True
        assert server.url.startswith("http://")
        assert server.url.endswith(":0") is False  # 实际端口不是 0
        await server.stop()
        assert server._listening is False

    async def test_stop_before_start_is_safe(self) -> None:
        """重复 stop() 不应抛错。"""
        server = MockServer()
        await server.stop()  # 什么都没做过
        await server.stop()  # 再来一次

    async def test_start_twice_raises(self) -> None:
        """已启动后再次 start 应报 RuntimeError。"""
        server = MockServer()
        await server.start(port=0)
        try:
            with pytest.raises(RuntimeError, match="already running"):
                await server.start(port=0)
        finally:
            await server.stop()

    async def test_real_http_request(self) -> None:
        """完整流程：启动 → 注册规则 → 真 HTTP 请求 → 验证响应 → 关闭。"""
        import httpx2

        server = MockServer()
        server.when("hi").reply("hello from mock server")

        await server.start(port=0)
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
