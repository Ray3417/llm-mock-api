"""formats 模块单元测试（精简版）。"""

from __future__ import annotations

from llm_mock_api.formats.request_helpers import (
    EMPTY_META,
    RequestMeta,
    build_mock_request,
    is_streaming,
)
from llm_mock_api.formats.serialize_helpers import (
    DEFAULT_USAGE,
    MS_PER_SECOND,
    finish_reason,
    gen_id,
    should_emit_text,
    split_text,
    tool_id,
)
from llm_mock_api.formats.types import SSEChunk
from llm_mock_api.types import Message, ReplyObject
from llm_mock_api.types.reply import ToolCall, Usage


# ── types.py ────────────────────────────────────────────


def test_sse_chunk_basic() -> None:
    """SSEChunk 基本字段与可选 event 字段。"""
    chunk = SSEChunk(data='{"text":"hi"}', event="content_block_delta")
    assert chunk.data == '{"text":"hi"}'
    assert chunk.event == "content_block_delta"

    # event 省略时为 None
    plain = SSEChunk(data="hello")
    assert plain.event is None


def test_request_meta_basic() -> None:
    """RequestMeta 默认值与显式赋值。"""
    meta = RequestMeta(headers={"Authorization": "Bearer xxx"}, path="/v1/chat")
    assert meta.headers["Authorization"] == "Bearer xxx"
    assert meta.path == "/v1/chat"
    assert EMPTY_META.headers == {}
    assert EMPTY_META.path == ""


# ── request_helpers.py ─────────────────────────────────


def test_is_streaming_true_false() -> None:
    """stream=True 返回 True，stream=False 返回 False — 两条核心路径。"""
    assert is_streaming({"stream": True}) is True
    assert is_streaming({"stream": False}) is False
    # 缺失 stream 字段默认流式
    assert is_streaming({"model": "gpt-4"}) is True


def test_build_mock_request_extracts_messages() -> None:
    """build_mock_request 正确提取用户消息和系统提示。"""
    messages = (
        Message(role="system", content="be helpful"),
        Message(role="user", content="hello world"),
    )
    req = build_mock_request(
        format="openai",
        body={"model": "gpt-4"},
        messages=messages,
        tools=None,
        default_model="gpt-3.5",
        raw={"model": "gpt-4"},
    )
    assert req.format == "openai"
    assert req.model == "gpt-4"
    assert req.last_message == "hello world"
    assert req.system_message == "be helpful"
    assert req.tools is None
    assert req.tool_names == ()
    assert req.last_tool_call_id is None


def test_build_mock_request_extracts_tools_and_tool_call() -> None:
    """build_mock_request 正确提取工具名称与 tool_call_id。"""
    messages = (
        Message(role="user", content="list files"),
        Message(role="tool", content="result", tool_call_id="call_42"),
    )
    tools = (
        type("T", (), {"name": "list_files", "description": None, "parameters": None})(),
        type("T", (), {"name": "read_file", "description": None, "parameters": None})(),
    )
    req = build_mock_request(
        format="openai",
        body={"stream": False},
        messages=messages,
        tools=tools,
        default_model="gpt-4",
        raw={},
    )
    assert req.streaming is False
    assert req.tool_names == ("list_files", "read_file")
    assert req.last_tool_call_id == "call_42"
    # body 无 model 时应回退到 default_model
    req2 = build_mock_request(
        format="anthropic",
        body={},
        messages=messages,
        tools=None,
        default_model="claude-sonnet",
        raw={},
    )
    assert req2.model == "claude-sonnet"


# ── serialize_helpers.py ────────────────────────────────


def test_split_text() -> None:
    """正常分段 + chunk_size<=0 边界。"""
    assert split_text("abcdefgh", 3) == ["abc", "def", "gh"]
    assert split_text("hello", 0) == ["hello"]


def test_gen_id_and_tool_id() -> None:
    """gen_id 格式正确且两次调用不同；tool_id 优先使用已有 id。"""
    id1 = gen_id("chatcmpl")
    id2 = gen_id("chatcmpl")
    assert id1.startswith("chatcmpl_")
    assert id1 != id2

    assert tool_id({"id": "existing"}, "call", 0) == "existing"
    generated = tool_id({}, "call", 3)
    assert generated.startswith("call_") and generated.endswith("_3")


def test_should_emit_text_three_branches() -> None:
    """should_emit_text 的三种核心语义分支。"""
    # 有 text → 输出
    assert should_emit_text(ReplyObject(text="hello")) is True
    # 有 tools（即使无 text）→ 不输出纯文本 delta
    tools = (ToolCall(name="func", args={}),)
    assert should_emit_text(ReplyObject(tools=tools)) is False
    # 只有 reasoning → 不输出纯文本 delta
    assert should_emit_text(ReplyObject(reasoning="thinking")) is False
    # 空 reply → 仍需输出（空文本也是一种文本）
    assert should_emit_text(ReplyObject()) is True


def test_finish_reason() -> None:
    """有 tools → 'tool_calls'；无 tools → 'stop'。"""
    tools = (ToolCall(name="func", args={}),)
    assert finish_reason(ReplyObject(tools=tools), "tool_calls", "stop") == "tool_calls"
    assert finish_reason(ReplyObject(text="done"), "tool_calls", "stop") == "stop"


def test_constants() -> None:
    """基本常量值检查。"""
    assert MS_PER_SECOND == 1000
    assert isinstance(DEFAULT_USAGE, Usage)
    assert DEFAULT_USAGE.input == 10
    assert DEFAULT_USAGE.output == 5
