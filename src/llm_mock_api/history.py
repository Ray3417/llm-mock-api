"""请求历史记录模块。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from .types.request import MockRequest


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    """一个已记录的请求，包含匹配的规则和时间戳。"""

    request: MockRequest
    rule: str | None
    timestamp: int


class RequestHistory:
    """记录服务器处理的每个请求。

    可迭代，并提供流畅的查询方法用于测试断言。
    """

    def __init__(self) -> None:
        self._entries: list[RecordedRequest] = []

    def record(self, request: MockRequest, rule: str | None) -> None:
        """记录一个新请求。"""
        self._entries.append(
            RecordedRequest(request=request, rule=rule, timestamp=int(time.time() * 1000))
        )

    def count(self) -> int:
        """已记录请求的数量。"""
        return len(self._entries)

    def first(self) -> RecordedRequest | None:
        """最早记录的请求，空时返回 None。"""
        return self._entries[0] if self._entries else None

    def last(self) -> RecordedRequest | None:
        """最近记录的请求，空时返回 None。"""
        return self._entries[-1] if self._entries else None

    def at(self, index: int) -> RecordedRequest | None:
        """获取指定索引的条目，支持负索引。"""
        try:
            return self._entries[index]
        except IndexError:
            return None

    def where(self, predicate: Callable[[RecordedRequest], bool]) -> list[RecordedRequest]:
        """按谓词过滤条目。"""
        return [entry for entry in self._entries if predicate(entry)]

    @property
    def all(self) -> Sequence[RecordedRequest]:
        """所有条目作为只读序列。"""
        return tuple(self._entries)

    def clear(self) -> None:
        """移除所有已记录的条目。"""
        self._entries.clear()

    def __iter__(self) -> Iterator[RecordedRequest]:
        """支持 for...in 迭代已记录条目。"""
        return iter(self._entries)
