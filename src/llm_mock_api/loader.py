"""工作流示例：
1. JSON5 规则文件 `rules.json5`:
   [
     { when: "hello", reply: "Hi!" },
     { when: "step", replies: [{ reply: "First" }, { reply: "Second", latency: 100 }] },
   ]
2. Python 处理器文件 `handlers/greeting.py`:
   class _Handler:
       def match(self, req): return "echo" in req.last_message
       def respond(self, req): return f"Echo: {req.last_message}"
   default = _Handler()
3. `load_rules_from_path("rules/", ctx)` 读取并注册
4. 后续请求时 `rule_engine.match(req)` 找到匹配规则并返回回复
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import json5

from .rule_engine import RuleEngine, _SequenceStep, create_sequence_resolver
from .types.reply import Reply, ReplyObject, ReplyOptions, ToolCall
from .types.rule import Handler, Match, MatchObject

# ---------------------------------------------------------------------------
# JSON5 原始类型（加载时的中间形态，验证后转换为正式类型）
# ---------------------------------------------------------------------------

type _Json5MatchRaw = str | dict[str, Any]
type _Json5ReplyRaw = str | dict[str, Any]
type _Templates = dict[str, _Json5ReplyRaw] | None
type _FormatName = Any  # 运行时验证后注入 MatchObject.format


# ---------------------------------------------------------------------------
# 加载上下文
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LoadContext:
    """规则加载上下文。"""

    engine: RuleEngine
    set_fallback: Callable[[Reply], None] | None = None


# ---------------------------------------------------------------------------
# 辅助函数：Match 与 Reply 解析
# ---------------------------------------------------------------------------


def _parse_regex_string(s: str) -> re.Pattern[str] | str:
    """把 "/pattern/flags" 编译为正则，普通字符串原样返回。"""
    m = re.fullmatch(r"/(.+)/([dgimsuy]*)", s)
    if m:
        pattern, flags_str = m.group(1), m.group(2)
        flags = 0
        for ch in flags_str:
            flags |= {"i": re.I, "m": re.M, "s": re.S, "u": re.U}.get(ch, 0)
        return re.compile(pattern, flags)
    return s


_KNOWN_FORMATS = frozenset(("openai", "anthropic", "responses"))


def _compile_match(when: _Json5MatchRaw) -> Match:
    """把 JSON5 中的 match 配置转换为运行时 Match。"""
    if isinstance(when, str):
        return _parse_regex_string(when)

    def _regex_or_none(v: object) -> str | re.Pattern[str] | None:
        if isinstance(v, str):
            return _parse_regex_string(v)
        return None

    fmt = when.get("format")
    if fmt is not None and (not isinstance(fmt, str) or fmt not in _KNOWN_FORMATS):
        raise ValueError(f"Invalid 'format' value: {fmt!r}. Expected one of: openai, anthropic, responses")

    return MatchObject(
        message=_regex_or_none(when.get("message")),
        model=_regex_or_none(when.get("model")),
        system=_regex_or_none(when.get("system")),
        format=cast(_FormatName, fmt),
    )


def _parse_reply(raw: _Json5ReplyRaw) -> Reply:
    """把 JSON5 中的 reply 值转换为 Reply（str 或 ReplyObject）。"""
    if isinstance(raw, str):
        return raw
    tools_raw = raw.get("tools")
    tools: list[ToolCall] | None = None
    if tools_raw is not None and isinstance(tools_raw, list):
        tools = []
        for t in tools_raw:
            if isinstance(t, dict) and "name" in t:
                args_val = t.get("args", {})
                tools.append(
                    ToolCall(
                        name=t["name"],
                        args=args_val if isinstance(args_val, dict) else {},
                    )
                )
    return ReplyObject(
        text=raw.get("text") if isinstance(raw.get("text"), str) else None,
        reasoning=raw.get("reasoning") if isinstance(raw.get("reasoning"), str) else None,
        tools=tools,
    )


def _resolve_reply_ref(
    ref: _Json5ReplyRaw,
    templates: _Templates,
    file_path: str,
) -> _Json5ReplyRaw:
    """解析 reply 引用："$name" 从 templates 查，其他原样返回。"""
    if isinstance(ref, str) and ref.startswith("$"):
        name = ref[1:]
        if templates is None or name not in templates:
            raise ValueError(f'Unknown template "{name}" in {file_path}')
        return templates[name]
    return ref


# ---------------------------------------------------------------------------
# 序列规则
# ---------------------------------------------------------------------------


def _add_sequence_rule(
    engine: RuleEngine,
    match: Match,
    entries: list[str | dict[str, Any]],
    templates: _Templates,
    file_path: str,
) -> None:
    """注册一个回复序列规则（每次匹配推进到下一条回复）。"""
    steps: list[_SequenceStep] = []
    for entry in entries:
        if isinstance(entry, str) or not isinstance(entry, dict) or "reply" not in entry:
            resolved = _resolve_reply_ref(entry, templates, file_path)
            steps.append(_SequenceStep(reply=_parse_reply(resolved)))
        else:
            reply_raw = entry["reply"]
            resolved = _resolve_reply_ref(reply_raw, templates, file_path)
            latency = entry.get("latency")
            chunk_size = entry.get("chunkSize")

            # 校验 latency / chunkSize：必须是非负整数
            valid_latency: int | None = None
            valid_chunk_size: int | None = None
            if latency is not None:
                if not isinstance(latency, int) or latency < 0:
                    raise ValueError(
                        f"Rule in {file_path}: 'latency' must be a nonnegative integer"
                    )
                valid_latency = latency
            if chunk_size is not None:
                if not isinstance(chunk_size, int) or chunk_size < 0:
                    raise ValueError(
                        f"Rule in {file_path}: 'chunkSize' must be a nonnegative integer"
                    )
                valid_chunk_size = chunk_size

            opts = None
            if valid_latency is not None or valid_chunk_size is not None:
                opts = ReplyOptions(latency=valid_latency, chunk_size=valid_chunk_size)
            steps.append(_SequenceStep(reply=_parse_reply(resolved), options=opts))

    rule = engine.add(match, "")
    resolver, entry_count = create_sequence_resolver(steps, rule)
    rule.resolve = resolver
    rule.remaining = entry_count


# ---------------------------------------------------------------------------
# JSON5 文件加载
# ---------------------------------------------------------------------------


async def _load_json5_file(file_path: str, ctx: LoadContext) -> None:
    """加载一个 JSON5 规则文件。"""
    content = await asyncio.to_thread(pathlib.Path(file_path).read_text, encoding="utf-8")
    parsed = await asyncio.to_thread(json5.loads, content)

    # 顶层：list 表示规则数组，dict 表示 { templates?, fallback?, rules }
    if isinstance(parsed, list):
        rules = parsed
        templates: _Templates = None
        fallback = None
    elif isinstance(parsed, dict):
        rules_val = parsed.get("rules", [])
        if not isinstance(rules_val, list):
            raise ValueError(f"Invalid 'rules' in {file_path}: expected array")
        rules = rules_val

        templates_val = parsed.get("templates")
        templates = templates_val if isinstance(templates_val, dict) else None

        fallback = parsed.get("fallback") if "fallback" in parsed else None
    else:
        raise ValueError(f"Invalid JSON5 file {file_path}: expected array or object at top level")

    # 设置 fallback
    if fallback is not None and ctx.set_fallback is not None:
        ctx.set_fallback(_parse_reply(fallback))

    # 逐条注册规则
    for r in rules:
        if not isinstance(r, dict):
            raise ValueError(f"Invalid rule in {file_path}: expected object")

        when = r.get("when")
        if when is None:
            raise ValueError(f"Rule in {file_path} missing 'when'")
        if not isinstance(when, (str, dict)):
            raise ValueError(f"Rule in {file_path} has invalid 'when'")

        match = _compile_match(when)

        # replies（序列）优先于 reply（单条）
        if "replies" in r:
            replies_val = r.get("replies")
            if not isinstance(replies_val, list):
                raise ValueError(f"Rule in {file_path}: 'replies' must be an array")
            _add_sequence_rule(ctx.engine, match, replies_val, templates, file_path)
        else:
            reply_val = r.get("reply")
            if reply_val is None:
                raise ValueError(f"Rule in {file_path} missing 'reply'")
            if not isinstance(reply_val, (str, dict)):
                raise ValueError(f"Rule in {file_path} has invalid 'reply'")

            resolved = _resolve_reply_ref(reply_val, templates, file_path)
            reply = _parse_reply(resolved)
            rule = ctx.engine.add(match, reply)
            times_val = r.get("times")
            if times_val is not None:
                if not isinstance(times_val, int) or times_val <= 0:
                    raise ValueError(f"Rule in {file_path}: 'times' must be a positive integer")
                rule.remaining = times_val


# ---------------------------------------------------------------------------
# Python 处理器文件加载
# ---------------------------------------------------------------------------


async def _load_handler_file(file_path: str, ctx: LoadContext) -> None:
    """动态加载一个 Python 处理器文件。

    文件应定义 `default = <Handler 对象或列表>`，可选 `fallback`。
    """

    def _load_module() -> object:
        module_name = f"_llm_mock_api_handler_{pathlib.Path(file_path).stem}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot create module spec for {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    mod = await asyncio.to_thread(_load_module)

    if not hasattr(mod, "default"):
        raise ValueError(
            f"Invalid handler file {file_path}. Expected `default` export with Handler or list[Handler].",
        )

    default_val = getattr(mod, "default")
    handlers: list[Handler]
    if isinstance(default_val, list):
        for h in default_val:
            if not isinstance(h, Handler):
                raise ValueError(
                    f"Invalid handler file {file_path}: `default` list contains non-Handler item"
                )
        handlers = list(default_val)
    elif isinstance(default_val, Handler):
        handlers = [default_val]
    else:
        raise ValueError(
            f"Invalid handler file {file_path}: `default` must be a Handler or list[Handler]",
        )

    # 可选 fallback
    if hasattr(mod, "fallback") and ctx.set_fallback is not None:
        fb = getattr(mod, "fallback")
        if isinstance(fb, (str, dict)):
            ctx.set_fallback(_parse_reply(fb))

    for handler in handlers:
        ctx.engine.add_handler(
            handler.match,
            handler.respond,
            f"(handler: {file_path})",
        )


# ---------------------------------------------------------------------------
# 文件扩展名 → loader 映射
# ---------------------------------------------------------------------------

type _FileLoader = Callable[[str, LoadContext], Awaitable[None]]

_LOADERS: dict[str, _FileLoader] = {
    ".json5": _load_json5_file,
    ".json": _load_json5_file,
    ".py": _load_handler_file,
}


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


async def load_rules_from_path(path_or_dir: str, ctx: LoadContext) -> None:
    """从文件或目录递归加载规则。

    目录内容按名称排序后依次处理。
    """
    path = pathlib.Path(path_or_dir)

    # 是文件？按扩展名派发
    is_file = await asyncio.to_thread(path.is_file)
    if is_file:
        loader = _LOADERS.get(path.suffix)
        if loader is None:
            raise ValueError(f"Unsupported file extension '{path.suffix}' for {path_or_dir}")
        await loader(str(path), ctx)
        return

    # 是目录？递归处理
    is_dir = await asyncio.to_thread(path.is_dir)
    if not is_dir:
        return

    entries = sorted(await asyncio.to_thread(lambda: list(path.iterdir())))
    for entry in entries:
        entry_is_file = await asyncio.to_thread(entry.is_file)
        entry_is_dir = await asyncio.to_thread(entry.is_dir)
        if entry_is_dir:
            await load_rules_from_path(str(entry), ctx)
        elif entry_is_file:
            loader = _LOADERS.get(entry.suffix)
            if loader is not None:
                await loader(str(entry), ctx)
