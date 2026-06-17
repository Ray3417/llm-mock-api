"""工作流示例：
1. `server = MockServer({logLevel: "info"})` → 创建实例，初始化 RuleEngine + RuleBuilder + RequestHistory
2. `server.when("hello").reply("Hi there!")` → 通过 RuleBuilder 注册规则
3. `await server.start()` → FastAPI 路由注册（chat-completions / responses）并启动 uvicorn
4. HTTP 请求到达 → route_handler 调用 engine.match() → 返回响应
5. `await server.stop()` → 关闭 uvicorn 服务器
6. `async with MockServer() as srv:` → 自动调用 start/stop 清理
"""

from __future__ import annotations

import asyncio
import signal as _signal
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

import uvicorn
from fastapi import FastAPI

from .formats.anthropic import anthropic_format
from .formats.openai.chat_completions import chat_completions_format
from .formats.openai.responses import responses_format
from .formats.types import Format
from .history import RequestHistory
from .loader import LoadContext, load_rules_from_path
from .logger import Logger, LogLevel
from .rule_engine import RuleEngine, RuleSummary
from .rule_builder import PendingRule, RuleBuilder, RuleHandle
from .route_handler import RouteHandlerDeps, create_route_handler
from .types.reply import Reply, ReplyOptions


@runtime_checkable
class RuleAPI(Protocol):
    """MockServer 对外暴露的规则注册 API。

    与 TypeScript 端 `type RuleAPI = Pick<RuleBuilder, "when" | "whenTool" | "whenToolResult" | "nextError">` 对应。
    """

    def when(self, match: object, /) -> PendingRule:
        """注册一条匹配规则。对返回的 PendingRule 调用 `.reply()` 完成注册。"""
        ...

    def when_tool(self, tool_name: str, /) -> PendingRule:
        """`when({ toolName })` 的简写。"""
        ...

    def when_tool_result(self, tool_call_id: str, /) -> PendingRule:
        """`when({ toolCallId })` 的简写。"""
        ...

    def next_error(
        self,
        status: int,
        message: str,
        type_: str | None = None,
        /,
    ) -> RuleHandle:
        """为"下一个请求"排队一个一次性错误。触发后自动移除。"""
        ...


@dataclass(slots=True, kw_only=True)
class MockServerOptions:
    """MockServer 的唯一配置源。

    所有默认值在此集中定义，MockServer.__init__ 只负责读取，不再做二次判定。
    dataclass 自动生成 __init__ / __repr__，避免手写 `__slots__ = ()` 与赋值不匹配的 bug。

    字段分类：

    - 构造参数：`port` / `host` / `log_level` / `default_latency` / `default_chunk_size`
    - 构造后应用：`rules` / `fallback`（`from_json_config()` 负责）
    - 运行时行为：`watch`（`run_until_shutdown()` 负责）
    """

    port: int = 0
    """监听端口。`0` 让 OS 分配随机端口（默认，测试友好）。传具体值如 `8002` 固定端口。"""

    host: str = "127.0.0.1"
    """绑定的主机名。传 `"0.0.0.0"` 以监听所有网卡。默认 `"127.0.0.1"`。"""

    log_level: LogLevel = "none"
    """日志级别。默认 `"none"`（不输出）。"""

    default_latency: int = 0
    """SSE chunk 之间的默认毫秒延迟。各规则可覆盖。默认 `0`（不延迟）。"""

    default_chunk_size: int = 0
    """默认 SSE 文本分块大小（字符数）。各规则可覆盖。默认 `0`（不分块）。"""

    fallback: str | None = None
    """无规则匹配时返回的默认回复。默认 `None`（使用内置 fallback 文案）。"""

    rules: str | None = None
    """json5 规则文件或目录的路径。默认 `None`（不预加载）。"""

    watch: bool = False
    """是否监听 rules 路径的文件改动并自动 reload。默认 `False`。"""

    @staticmethod
    def from_json_file(path: str) -> "MockServerOptions":
        """从 JSON 文件加载配置。

        **相对路径以 config 文件所在目录为基准（而非当前工作目录），
        确保 `llm-mock-api init` 生成的默认配置无论从哪里调用都能正确加载。

        示例（字段全可选；未写的字段走 MockServer 内部默认值）::

            {
                "port": 8002,
                "host": "127.0.0.1,
                "log_level": "info",
                "default_latency": 0,
                "default_chunk_size": 0,
                "fallback": "sorry, I don't know",
                "rules": "./rules.json5",
                "watch": true
            }
        """
        import dataclasses
        import json
        import pathlib

        config_path = pathlib.Path(path).resolve()
        config_dir = config_path.parent

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found: {path}") from None
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in config file {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise RuntimeError(
                f"Config file {path} must be a JSON object at top level, got {type(data).__name__}"
            )

        known = {f.name for f in dataclasses.fields(MockServerOptions)}
        payload: dict[str, Any] = {}
        for k, v in data.items():
            if k not in known:
                continue  # 过滤未知字段
            if k == "rules" and isinstance(v, str):
                # rules 路径以 config 所在目录解析，确保 init 生成的相对路径无论从哪里启动都能正确加载
                rules_path = pathlib.Path(v)
                if not rules_path.is_absolute():
                    v = str(config_dir / rules_path)
            payload[k] = v
        return MockServerOptions(**payload)


# ---------------------------------------------------------------------------
# MockServer：主类
# ---------------------------------------------------------------------------


class MockServer:
    """Mock LLM 服务器。支持 OpenAI Chat Completions 与 Responses 格式。

    通过 `when()` 注册规则，将应用程序的 API endpoint 指向 `url` 即可。

    注意：Python 版本的 `async with` 会**自动调用 start()**，
    这与 TS 的 `await using`（只自动 stop）略有差异——
    但这是 Python 上下文管理器的惯用用法。
    """

    # --- 初始化 ---

    def __init__(self, options: MockServerOptions | None = None) -> None:
        opts = options or MockServerOptions()

        # 所有默认值已在 MockServerOptions 中集中定义，此处直接读取
        self._host: str = opts.host
        self._default_port: int = opts.port
        self._logger: Logger = Logger(opts.log_level)

        # 直接构造 ReplyOptions（默认值已是 0，不需要条件判断）
        self._default_options: ReplyOptions = ReplyOptions(
            latency=opts.default_latency,
            chunk_size=opts.default_chunk_size,
        )

        self._engine: RuleEngine = RuleEngine()
        self._rules: RuleBuilder = RuleBuilder(self._engine)
        self._history: RequestHistory = RequestHistory()
        self._fallback_reply: Reply = "Mock server: no matching rule."
        self._listening: bool = False
        self._watch_path: str | None = None  # from_json_config 记录的 rules 路径

        # Fastify <-> FastAPI 对应：Fastify({ logger: false })
        self._app: FastAPI = FastAPI()

        # 暴露 RuleBuilder 的方法为实例属性（对应 TS: this.when = this.rules_.when.bind(this.rules_)）
        # 在 Python 中，bound method 自动持有 self，因此不需要显式 bind。
        self.when = self._rules.when
        self.when_tool = self._rules.when_tool
        self.when_tool_result = self._rules.when_tool_result
        self.next_error = self._rules.next_error

        self._formats: list[Format] = [
            cast(Format, chat_completions_format),
            cast(Format, responses_format),
            cast(Format, anthropic_format),
        ]

        # 组装路由处理器依赖并注册路由
        deps = RouteHandlerDeps(
            engine=self._engine,
            history=self._history,
            logger=self._logger,
            default_options=self._default_options,
            get_fallback=lambda: self._fallback_reply,
        )
        for fmt in self._formats:
            self._app.post(fmt.route)(create_route_handler(fmt, deps))

    # --- fallback ---

    def fallback(self, reply: Reply) -> None:
        """设置无规则匹配时返回的默认回复。"""
        self._fallback_reply = reply

    # --- 加载规则文件 ---

    async def load(self, path_or_dir: str) -> None:
        """从 `.json5` 文件、`.py` 处理器文件或目录递归加载规则。"""
        before = self._engine.rule_count
        # 将 set_fallback 作为构造参数传入，保持不可变性（对应用户的方案 4）
        ctx = LoadContext(
            engine=self._engine,
            set_fallback=self.fallback,
        )
        await load_rules_from_path(path_or_dir, ctx)
        loaded = self._engine.rule_count - before
        self._logger.info(f'Loaded {loaded} rule{"" if loaded == 1 else "s"} from {path_or_dir}')

    # --- 状态查询 ---

    @property
    def history(self) -> RequestHistory:
        """服务器记录的所有请求。"""
        return self._history

    def is_done(self) -> bool:
        """所有设置了 `.times()` 限制的规则是否都已消耗完毕。"""
        return self._engine.is_done()

    def reset(self) -> None:
        """清空所有规则、请求历史，恢复默认 fallback。"""
        self._engine.clear()
        self._history.clear()
        self._fallback_reply = "Mock server: no matching rule."
        self._logger.info("Server reset: rules and history cleared")

    @property
    def url(self) -> str:
        """服务器监听的 URL，例如 `http://127.0.0.1:12345`。未启动时抛错。

        注意：绑定地址 `0.0.0.0` 不能作为客户端连接目标，自动回退为 `127.0.0.1`。
        """
        if not self._listening:
            raise RuntimeError("Server is not running. Call start() first.")
        assert self._server is not None, "server should be set when listening"
        host = "127.0.0.1" if self._host == "0.0.0.0" else self._host
        # 防御性：确保 servers 与 sockets 非空（理论上 startup() 之后一定有）
        if not self._server.servers or not self._server.servers[0].sockets:
            raise RuntimeError("Server is listening but sockets are not ready")
        port = self._server.servers[0].sockets[0].getsockname()[1]
        return f"http://{host}:{port}"

    @property
    def routes(self) -> list[str]:
        """已注册的 API 路由路径列表。"""
        return [fmt.route for fmt in self._formats]

    @property
    def rule_count(self) -> int:
        """当前注册的规则总数。"""
        return self._engine.rule_count

    @property
    def rules(self) -> list[RuleSummary]:
        """所有已注册规则的人类可读摘要。"""
        return self._engine.describe()

    # --- 生命周期 ---

    async def start(self) -> None:
        """启动 HTTP 服务器。端口由 MockServerOptions.port 决定（默认 0 = 随机分配）。"""
        if self._listening:
            raise RuntimeError("Server is already running.")
        config = uvicorn.Config(
            self._app,
            host=self._host,
            port=self._default_port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)
        # uvicorn.Server.serve() 内部会先做这两步初始化，再进入 startup/main_loop/shutdown。
        # 我们拆开管理以便在 start() 返回时 socket 已就绪。
        if not config.loaded:
            config.load()
        self._server.lifespan = config.lifespan_class(config)
        await self._server.startup()
        # 启动 serve 循环（在后台 task 中运行，stop() 时通过 should_exit 优雅退出）
        self._server_task = asyncio.create_task(self._server.main_loop())
        self._listening = True
        self._logger.info(f"Listening on {self.url}")

    async def stop(self) -> None:
        """停止 HTTP 服务器。可安全多次调用。"""
        if not self._listening:
            return
        # 标记应退出，等待 main_loop 结束并释放资源
        self._server.should_exit = True
        try:
            await self._server_task
            await self._server.shutdown()
        finally:
            self._listening = False
            self._logger.info("Server stopped")

    # --- async context manager（对应 TS: Symbol.asyncDispose）---

    async def __aenter__(self) -> "MockServer":
        """进入上下文时自动调用 start()（默认随机端口）。"""
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.stop()

    # --- 从 JSON 配置启动（对应用户的 json 配置习惯）---

    @staticmethod
    async def from_json_config(path: str) -> "MockServer":
        """从 config.json 构造并初始化 MockServer，但不启动。

        步骤：
          1. 读取 JSON 得到 `MockServerOptions`（包含构造参数 + rules + fallback + watch）
          2. 用 `config` 构造 `MockServer`（构造时只消费构造参数字段，rules/fallback/watch 在此后应用）
          3. 若配置了 fallback，调用 `server.fallback(...)`
          4. 若配置了 rules，调用 `await server.load(rules)`
          5. 若配置了 rules 同时 watch=True，记录 `_watch_path` 供 `run_until_shutdown` 使用

        用法：

            server = await MockServer.from_json_config("config.json")
            await server.start()
            # ...
            await server.stop()

        或者直接用 `run_until_shutdown` 一次性管理启动/监听/关闭。
        """
        config = MockServerOptions.from_json_file(path)
        server = MockServer(config)
        if config.fallback is not None:
            server.fallback(config.fallback)
        if config.rules is not None:
            await server.load(config.rules)
            # 记录路径以便 run_until_shutdown 的 watch 模式使用
            if config.watch:
                server._watch_path = config.rules
        return server

    # --- 运行时主循环（封装 cli 中的 start + watch + signal 监听）---

    async def run_until_shutdown(
        self,
        *,
        watch_path: str | None = None,
    ) -> None:
        """启动服务器并运行直到收到 SIGINT/SIGTERM；可选地 watch 指定路径。

        - watch_path 为空时：若 server 之前通过 `from_json_config` 启动且 JSON 中
          `watch: true` 有规则文件，则自动 watch 规则路径。
        """
        import os

        await self.start()

        # 优先使用显式传入的 watch_path；否则回退到 from_json_config 记录的 _watch_path
        effective_watch = watch_path or self._watch_path

        try:
            watcher_task: asyncio.Task | None = None
            if effective_watch and os.path.exists(effective_watch):
                watcher_task = asyncio.create_task(
                    self._watch_path_for_reload(effective_watch),
                )
                self._logger.info(f"Watching {effective_watch} for changes")

            # 监听信号；失败时忽略（pytest / Windows 子线程场景常见）
            try:
                loop = asyncio.get_event_loop()
                for sig_name in ("SIGINT", "SIGTERM"):
                    sig_value = getattr(_signal, sig_name, None)
                    if sig_value is None:
                        continue
                    try:
                        loop.add_signal_handler(
                            sig_value,
                            lambda s=sig_name: asyncio.create_task(self._on_signal_shutdown(s)),  # type: ignore[misc]
                        )
                    except NotImplementedError:
                        # Windows 某些事件循环不支持 add_signal_handler；降级：仅挂起等待 stop()
                        pass
            except Exception:  # noqa: BLE001 — 信号注册失败不应阻塞主流程
                pass

            # 挂起直到 server 被 stop()（stop() 会把 _listening 设为 False）
            while self._listening:
                await asyncio.sleep(0.2)

            if watcher_task is not None and not watcher_task.done():
                watcher_task.cancel()
                try:
                    await watcher_task
                except asyncio.CancelledError:
                    pass
        finally:
            if self._listening:
                await self.stop()

    async def _on_signal_shutdown(self, signal_name: str) -> None:
        """响应 Ctrl+C / SIGTERM 的回调。"""
        if not self._listening:
            return
        self._logger.info(f"Got {signal_name}, shutting down...")
        await self.stop()
        self._logger.info("Clean shutdown complete")

    async def _watch_path_for_reload(self, path: str, *, debounce_ms: int = 100) -> None:
        """watch 模式：监听指定路径的文件改动，debounce 后 reset + reload + fallback。"""
        snapshot = _fs_snapshot(path)
        while self._listening:
            await asyncio.sleep(debounce_ms / 1000)
            current = _fs_snapshot(path)
            if current != snapshot:
                snapshot = current
                try:
                    fallback_snapshot = self._fallback_reply
                    self.reset()
                    await self.load(path)
                    # 若之前自定义过 fallback，reload 后恢复它
                    if fallback_snapshot != "Mock server: no matching rule.":
                        self.fallback(fallback_snapshot)
                    self._logger.info(f"Reloaded rules from {path}")
                except Exception as err:  # noqa: BLE001
                    self._logger.error(f"Failed to reload rules from {path}: {err}")


def _fs_snapshot(path: str) -> str:
    """返回一个路径的"文件系统快照"字符串，用于比较是否有改动。

    对目录：递归收集所有条目的 (path, mtime, size) 并哈希；对文件：(mtime, size)。
    粒度足够检测到"内容被编辑"，且比逐文件读内容快很多。
    """
    import hashlib
    import os

    if os.path.isfile(path):
        st = os.stat(path)
        return f"file:{st.st_mtime_ns}:{st.st_size}"

    entries: list[str] = []
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            entries.append(f"{fp}|{st.st_mtime_ns}|{st.st_size}")
    entries.sort()
    return "dir:" + hashlib.md5("\n".join(entries).encode("utf-8")).hexdigest()
