"""工作流示例：
1. HTTP 请求到达 → create_route_handler 返回的 handler 被调用
2. handler 解析 body + headers → RequestMeta
3. format.parse_request(body, meta) → MockRequest
4. engine.match(mock_req) → Rule | None
5. resolve_reply(rule, req, fallback, logger) → (ReplyObject, rule_desc | None)
6. 有 error → JSONResponse(serialize_error)；无 error → history.record(...)
7. 根据 is_streaming → 非流式 JSONResponse(serialize_complete)
                              或流式 create_sse_response(serialize(...))
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError as PydanticValidationError

from .formats.request_helpers import RequestMeta
from .formats.types import Format
from .history import RequestHistory
from .logger import Logger
from .rule_engine import RuleEngine
from .sse_writer import write_sse
from .types.reply import ErrorReply, Reply, ReplyObject, ReplyOptions
from .types.request import MockRequest
from .types.rule import Rule

HTTP_BAD_REQUEST = 400


def normalise_reply(reply: Reply) -> ReplyObject:
    """将 Reply（字符串或 ReplyObject）规范化为 ReplyObject。

    字符串被包装为 `{ text: reply }`，其他 ReplyObject 原样返回。
    """
    if isinstance(reply, str):
        return ReplyObject(text=reply)
    return reply


async def resolve_reply(
    matched: Rule | None,
    mock_req: MockRequest,
    fallback: Reply,
    logger: Logger,
) -> tuple[ReplyObject, str | None]:
    """解析匹配到的规则返回最终回复。

    - 无匹配：warn 日志 + 返回 fallback
    - resolve 为 callable：调用（支持异步）
    - resolve 为值：直接使用
    - 任何异常：error 日志 + fallback 兜底
    """
    if matched is None:
        logger.warn(f'No matching rule for "{mock_req.last_message}", using fallback')
        return normalise_reply(fallback), None

    try:
        resolve = matched.resolve
        if callable(resolve):
            raw = resolve(mock_req)
            if isinstance(raw, Awaitable):
                raw = await raw
        else:
            raw = resolve
        logger.debug(f"Matched rule {matched.description}")
        return normalise_reply(raw), matched.description
    except Exception as err:
        logger.error(f"Resolver threw for rule {matched.description}", err)
        return normalise_reply(fallback), matched.description


@dataclass(slots=True)
class RouteHandlerDeps:
    """路由处理器的依赖注入容器。"""

    engine: RuleEngine
    history: RequestHistory
    logger: Logger
    default_options: ReplyOptions
    get_fallback: Callable[[], Reply]


def _merge_options(default: ReplyOptions, override: ReplyOptions | None) -> ReplyOptions:
    """对应 TS `{ ...defaultOptions, ...matched?.options }`。"""
    if override is None:
        return default
    return ReplyOptions(
        latency=override.latency if override.latency is not None else default.latency,
        chunk_size=override.chunk_size if override.chunk_size is not None else default.chunk_size,
    )


def create_route_handler(
    fmt: Format,
    deps: RouteHandlerDeps,
) -> Callable[[Request], Awaitable[Response]]:
    """创建一个绑定了 format 和依赖的路由处理器。

    返回的 async 函数可直接作为 FastAPI 路由 handler。
    """
    engine = deps.engine
    history = deps.history
    logger = deps.logger
    default_options = deps.default_options
    get_fallback = deps.get_fallback

    async def handler(request: Request) -> Response:
        body = await request.json()
        # 对应 TS 的 `for (const [key, val] of Object.entries(request.headers))`
        # 值可能是 list（如多个 Set-Cookie）→ join(', ')；其他值直接用
        raw_headers = [(k, v) for k, v in request.headers.items()]
        headers: dict[str, str] = {}
        for key, val in raw_headers:
            if isinstance(val, list):
                headers[key] = ", ".join(str(v) for v in val)
            else:
                headers[key] = str(val) if val is not None else ""
        meta = RequestMeta(headers=headers, path=str(request.url.path))

        try:
            mock_req = fmt.parse_request(body, meta)
        except PydanticValidationError as err:
            # 对应 TS 的 `if (err instanceof ZodError) { ... }`
            # 只捕获已知的请求验证错误，其他异常向上抛出
            logger.warn(f"Invalid {fmt.name} request: {err}")
            return JSONResponse(
                status_code=HTTP_BAD_REQUEST,
                content=fmt.serialize_error(
                    ErrorReply(
                        status=HTTP_BAD_REQUEST,
                        message="Invalid request body",
                        type="invalid_request_error",
                    )
                ),
            )

        start_time = time.time()

        logger.debug(
            f"{fmt.name} request: "
            f"model={mock_req.model} "
            f"streaming={mock_req.streaming} "
            f"messages={len(mock_req.messages)}"
        )

        matched = engine.match(mock_req)
        resolved_reply, rule_desc = await resolve_reply(
            matched,
            mock_req,
            get_fallback(),
            logger,
        )

        if resolved_reply.error is not None:
            error = resolved_reply.error
            logger.info(f"Error reply: {error.status} {error.message}")
            history.record(mock_req, rule_desc)
            return JSONResponse(
                status_code=error.status,
                content=fmt.serialize_error(error),
            )

        history.record(mock_req, rule_desc)

        is_streaming = fmt.is_streaming(body)
        effective_options = _merge_options(
            default_options, matched.options if matched is not None else None
        )
        elapsed = int((time.time() - start_time) * 1000)
        mode = "stream" if is_streaming else "json"

        logger.info(
            f'POST {fmt.route} [{mode}] "{mock_req.last_message}" -> '
            f'{rule_desc if rule_desc is not None else "fallback"} ({elapsed}ms)'
        )
        if resolved_reply.text:
            logger.debug(f'Reply text: "{resolved_reply.text}"')
        if resolved_reply.tools:
            logger.debug(
                f"Reply tool calls: {', '.join(t.name for t in resolved_reply.tools)}"
            )
        if not is_streaming:
            return JSONResponse(content=fmt.serialize_complete(resolved_reply, mock_req.model))

        chunks = fmt.serialize(resolved_reply, mock_req.model, effective_options)
        return write_sse(chunks, effective_options)

    return handler
