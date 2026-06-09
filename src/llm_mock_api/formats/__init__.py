"""Formats 模块：各 API 线路格式的解析与序列化。

FormatImpl 是通用格式实现容器，字段即函数引用。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..types.reply import ErrorReply, ReplyObject, ReplyOptions
from ..types.request import FormatName, MockRequest
from .request_helpers import RequestMeta
from .types import SSEChunk


@dataclass(frozen=True, slots=True)
class FormatImpl:
    """格式实现容器。字段是函数引用，满足 Format Protocol。

    每个具体格式（OpenAI、Responses 等）只需创建一个 FormatImpl 实例，
    把 parse_request / is_streaming / serialize 等函数作为字段传入。
    """

    name: FormatName
    route: str
    parse_request: Callable[[Any, RequestMeta | None], MockRequest]
    is_streaming: Callable[[Any], bool]
    serialize: Callable[[ReplyObject, str, ReplyOptions | None], Sequence[SSEChunk]]
    serialize_complete: Callable[[ReplyObject, str], dict[str, Any]]
    serialize_error: Callable[[ErrorReply], dict[str, Any]]
