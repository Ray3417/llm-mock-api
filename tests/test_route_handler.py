"""route_handler 模块单元测试。"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from llm_mock_api.formats.types import SSEChunk
from llm_mock_api.history import RequestHistory
from llm_mock_api.logger import Logger
from llm_mock_api.rule_engine import RuleEngine
from llm_mock_api.route_handler import (
    RouteHandlerDeps,
    create_route_handler,
    normalise_reply,
    resolve_reply,
)
from llm_mock_api.types.reply import ErrorReply, ReplyObject, ReplyOptions
from llm_mock_api.types.request import MockRequest
from llm_mock_api.types.rule import Rule


# ── helpers ────────────────────────────────────────────

def _req(last_message: str = "", model: str = "gpt-4", streaming: bool = False) -> MockRequest:
    return MockRequest(
        format="openai",
        model=model,
        streaming=streaming,
        messages=(),
        last_message=last_message,
        system_message="",
    )


def _fake_format(
    *,
    name: str = "openai",
    route: str = "/v1/chat/completions",
    reply_text: str = "ok",
    is_streaming: bool = False,
) -> "_FakeFormat":
    return _FakeFormat(name=name, route=route, reply_text=reply_text, is_streaming_flag=is_streaming)


@dataclass(slots=True)
class _FakeFormat:
    """满足 Format Protocol 的最简实现，便于测试。"""

    name: str
    route: str
    reply_text: str
    is_streaming_flag: bool

    def parse_request(self, body: Any, meta: Any = None) -> MockRequest:
        msg = body.get("messages", [{}])[-1].get("content", "") if isinstance(body, dict) else ""
        return _req(last_message=msg, model=body.get("model", "gpt-4") if isinstance(body, dict) else "gpt-4",
                    streaming=self.is_streaming_flag)

    def is_streaming(self, body: Any) -> bool:
        return self.is_streaming_flag

    def serialize(self, reply: ReplyObject, model: str, options: ReplyOptions | None = None) -> list[SSEChunk]:
        return [SSEChunk(data=json.dumps({"content": reply.text or ""}))]

    def serialize_complete(self, reply: ReplyObject, model: str) -> dict[str, Any]:
        return {"choices": [{"message": {"content": reply.text or ""}}]}

    def serialize_error(self, error: ErrorReply) -> dict[str, Any]:
        return {"error": {"status": error.status, "message": error.message}}


def _deps(*, engine: RuleEngine | None = None, fallback_text: str = "fallback") -> RouteHandlerDeps:
    return RouteHandlerDeps(
        engine=engine or RuleEngine(),
        history=RequestHistory(),
        logger=Logger(level="none"),
        default_options=ReplyOptions(latency=0, chunk_size=50),
        get_fallback=lambda: fallback_text,
    )


# ── normalise_reply ────────────────────────────────────

class TestNormaliseReply:
    """测试 normalise_reply 的字符串与对象归一化。"""

    def test_string_becomes_reply_object(self) -> None:
        result = normalise_reply("hello")
        assert isinstance(result, ReplyObject)
        assert result.text == "hello"
        assert result.error is None

    def test_reply_object_passthrough(self) -> None:
        obj = ReplyObject(text="directly")
        assert normalise_reply(obj) is obj


# ── _merge_options ─────────────────────────────────────

def _merge_options(default: ReplyOptions, override: ReplyOptions | None) -> ReplyOptions:
    """复用 route_handler 中的同名逻辑（避免直接导入私有函数）。"""
    if override is None:
        return default
    return ReplyOptions(
        latency=override.latency if override.latency is not None else default.latency,
        chunk_size=override.chunk_size if override.chunk_size is not None else default.chunk_size,
    )


class TestMergeOptions:
    """测试默认与规则级 ReplyOptions 的合并。"""

    def test_none_override_returns_default(self) -> None:
        default = ReplyOptions(latency=10, chunk_size=20)
        assert _merge_options(default, None) == default

    def test_override_wins(self) -> None:
        default = ReplyOptions(latency=10, chunk_size=20)
        override = ReplyOptions(latency=100, chunk_size=200)
        result = _merge_options(default, override)
        assert result.latency == 100
        assert result.chunk_size == 200

    def test_partial_override(self) -> None:
        default = ReplyOptions(latency=10, chunk_size=20)
        override = ReplyOptions(latency=50, chunk_size=None)
        result = _merge_options(default, override)
        assert result.latency == 50
        assert result.chunk_size == 20


# ── resolve_reply ──────────────────────────────────────

class TestResolveReply:
    """测试 resolve_reply 的各种分支。"""

    def test_no_match_returns_fallback(self) -> None:
        logger = Logger(level="none")
        reply, rule_desc = asyncio.run(resolve_reply(None, _req(), "fallback", logger))
        assert reply.text == "fallback"
        assert rule_desc is None

    def test_static_string_reply(self) -> None:
        engine = RuleEngine()
        rule = engine.add("hello", "Hi!")
        logger = Logger(level="none")
        reply, rule_desc = asyncio.run(resolve_reply(rule, _req("hello"), "fallback", logger))
        assert reply.text == "Hi!"
        assert rule_desc == rule.description

    def test_static_reply_object(self) -> None:
        engine = RuleEngine()
        obj = ReplyObject(text="structured")
        rule = engine.add("hello", obj)
        reply, _ = asyncio.run(resolve_reply(rule, _req("hello"), "fb", Logger(level="none")))
        assert reply is obj

    def test_callable_resolver(self) -> None:
        engine = RuleEngine()
        rule = engine.add("echo", lambda req: f"You said: {req.last_message}")
        reply, _ = asyncio.run(resolve_reply(rule, _req("echo hi"), "fb", Logger(level="none")))
        assert reply.text == "You said: echo hi"

    def test_async_callable_resolver(self) -> None:
        async def slow(req: MockRequest) -> str:
            return f"async-{req.last_message}"

        engine = RuleEngine()
        rule = engine.add("async", slow)
        reply, _ = asyncio.run(resolve_reply(rule, _req("async ok"), "fb", Logger(level="none")))
        assert reply.text == "async-async ok"

    def test_resolver_exception_uses_fallback(self) -> None:
        engine = RuleEngine()
        def bad(_: MockRequest) -> str:
            raise RuntimeError("boom")
        rule = engine.add("fail", bad)
        reply, desc = asyncio.run(resolve_reply(rule, _req("fail"), "safe", Logger(level="none")))
        # 异常被捕获后走 fallback，但规则描述仍然保留
        assert reply.text == "safe"
        assert desc is not None


# ── create_route_handler ───────────────────────────────

async def _make_request(*, json_body: Any, headers: dict[str, str] | None = None) -> Request:
    """构造一个简易 FastAPI Request 对象。ASGI 要求 header key 为小写 bytes。"""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "query_string": b"",
        "server": ("testserver", 80),
    }
    req = Request(scope)
    req._body = json.dumps(json_body).encode()
    return req


class TestRouteHandler:
    """测试 create_route_handler 返回的 handler 完整流程。"""

    def test_json_response_with_matched_rule(self) -> None:
        engine = RuleEngine()
        engine.add("hello", "Hi!")
        handler = create_route_handler(_fake_format(), _deps(engine=engine))
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "hello"}], "model": "gpt-4"}))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "Hi!"

    def test_fallback_when_no_rule_matches(self) -> None:
        handler = create_route_handler(_fake_format(), _deps(fallback_text="default reply"))
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "nothing"}], "model": "gpt-4"}))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, JSONResponse)
        body = json.loads(resp.body)
        assert body["choices"][0]["message"]["content"] == "default reply"

    def test_streaming_response(self) -> None:
        engine = RuleEngine()
        engine.add("stream", "streaming reply")
        fmt = _fake_format(is_streaming=True)
        handler = create_route_handler(fmt, _deps(engine=engine))
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "stream"}], "model": "gpt-4"}))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, StreamingResponse)

        async def drain() -> str:
            return "".join([chunk async for chunk in resp.body_iterator])

        content = asyncio.run(drain())
        assert "streaming reply" in content

    def test_invalid_request_body_returns_400(self) -> None:
        @dataclass
        class _BadFormat:
            name: str = "openai"
            route: str = "/x"

            def parse_request(self, body: Any, meta: Any = None) -> MockRequest:
                from pydantic import BaseModel, Field
                from pydantic import ValidationError as PydanticValError

                class _ReqModel(BaseModel):
                    required_field: str = Field(...)

                _ReqModel.model_validate(body)
                return _req()

            def is_streaming(self, body: Any) -> bool:
                return False

            def serialize(self, reply: ReplyObject, model: str, options: ReplyOptions | None = None):
                return []

            def serialize_complete(self, reply: ReplyObject, model: str) -> dict[str, Any]:
                return {}

            def serialize_error(self, error: ErrorReply) -> dict[str, Any]:
                return {"error": {"message": error.message, "status": error.status}}

        handler = create_route_handler(_BadFormat(), _deps())
        req = asyncio.run(_make_request(json_body={"x": 1}))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body["error"]["message"] == "Invalid request body"

    def test_error_reply_returns_custom_status(self) -> None:
        engine = RuleEngine()
        engine.add("err", ReplyObject(error=ErrorReply(status=503, message="overloaded")))
        handler = create_route_handler(_fake_format(), _deps(engine=engine))
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "err"}]}))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 503
        body = json.loads(resp.body)
        assert body["error"]["message"] == "overloaded"

    def test_history_is_recorded(self) -> None:
        deps = _deps()
        deps.engine.add("hi", "hey")
        handler = create_route_handler(_fake_format(), deps)
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "hi"}]}))
        asyncio.run(handler(req))
        assert deps.history.count() == 1
        last = deps.history.last()
        assert last is not None
        assert last.request.last_message == "hi"

    def test_headers_are_passed_through(self) -> None:
        """Request 中的 headers 应被正确收集，不会因值是 list 导致错误。"""
        engine = RuleEngine()
        engine.add("test", "ok")
        handler = create_route_handler(_fake_format(), _deps(engine=engine))
        req = asyncio.run(_make_request(
            json_body={"messages": [{"content": "test"}]},
            headers={"Authorization": "Bearer tk", "x-custom": "v"},
        ))
        resp = asyncio.run(handler(req))
        assert isinstance(resp, JSONResponse)
        assert resp.status_code == 200

    def test_callable_resolver_in_full_handler(self) -> None:
        engine = RuleEngine()
        engine.add("echo", lambda r: f"got: {r.last_message}")
        handler = create_route_handler(_fake_format(), _deps(engine=engine))
        req = asyncio.run(_make_request(json_body={"messages": [{"content": "echo hello"}]}))
        resp = asyncio.run(handler(req))
        body = json.loads(resp.body)
        assert "got: echo hello" in body["choices"][0]["message"]["content"]
