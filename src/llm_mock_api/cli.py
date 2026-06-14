"""llm-mock-api 命令行入口。

子命令：
  start     - 启动 Mock 服务器（默认子命令）
  init      - 生成示例配置和规则文件
  validate  - 检查规则文件语法

优先级：CLI 参数 > JSON 配置 > MockServerOptions dataclass 默认值
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .logger import LEVEL_PRIORITY, LogLevel
from .mock_server import MockServer, MockServerOptions

# ---------------------------------------------------------------------------
# 验证器
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS = list(LEVEL_PRIORITY.keys())
_MAX_PORT = 65535


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"Invalid port '{value}'. Must be 1-{_MAX_PORT} or 0 (random).")
    if port < 0 or port > _MAX_PORT:
        raise argparse.ArgumentTypeError(f"Invalid port '{value}'. Must be 0-{_MAX_PORT}.")
    return port


def _parse_log_level(value: str) -> str:
    if value not in _VALID_LOG_LEVELS:
        raise argparse.ArgumentTypeError(
            f"Invalid log level '{value}'. Valid: {', '.join(_VALID_LOG_LEVELS)}"
        )
    return value


def _parse_non_negative_int(value: str) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got '{value}'.")
    if n < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got '{value}'.")
    return n


# ---------------------------------------------------------------------------
# 共享参数
# ---------------------------------------------------------------------------

def _add_start_options(parser: argparse.ArgumentParser) -> None:
    """start 子命令的参数。"""
    parser.add_argument(
        "--config", default="./config.json",
        help="JSON 配置文件路径（默认：./config.json）",
    )
    parser.add_argument("-p", "--port", type=_parse_port, help="监听端口（0 = 随机分配）")
    parser.add_argument("-H", "--host", help="绑定地址，如 127.0.0.1 或 0.0.0.0")
    parser.add_argument("-r", "--rules", help="JSON5 规则文件或目录路径")
    parser.add_argument("-l", "--latency", type=_parse_non_negative_int, help="SSE chunk 间延迟（毫秒）")
    parser.add_argument("-c", "--chunk-size", type=_parse_non_negative_int, dest="chunk_size", help="SSE 分块大小（字符数）")
    parser.add_argument("-f", "--fallback", help="无规则匹配时的默认回复文本")
    parser.add_argument("-w", "--watch", action="store_true", help="监听 rules 变化并热重载")
    parser.add_argument("--log-level", type=_parse_log_level, help=f"日志级别（{'/'.join(_VALID_LOG_LEVELS)}）")


# ---------------------------------------------------------------------------
# 配置合并
# ---------------------------------------------------------------------------

def _load_config_file(path: str) -> dict[str, Any]:
    """尝试从 JSON 文件读取配置。失败时打印提示，返回空 dict。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[warning] Config file {path} is not a JSON object; ignoring.")
            return {}
        return data
    except FileNotFoundError:
        print(f"[info] No config file found at {path}; using default options.")
        return {}
    except json.JSONDecodeError as exc:
        print(f"[warning] Failed to parse {path}: {exc}; using default options.")
        return {}


def _merge_options(args: argparse.Namespace) -> tuple[MockServerOptions, str | None, str | None, bool]:
    """将 CLI 参数与 JSON 配置合并，返回 (options, rules, fallback, watch)。"""
    json_cfg = _load_config_file(args.config)

    def _pick(key: str) -> Any:
        if getattr(args, key, None) is not None:
            return getattr(args, key)
        if key in json_cfg and json_cfg[key] is not None:
            return json_cfg[key]
        return None

    options_kwargs: dict[str, Any] = {}

    for key in ("port", "host", "log_level"):
        val = _pick(key)
        if val is not None:
            options_kwargs[key] = val

    latency = _pick("latency")
    if latency is not None:
        options_kwargs["default_latency"] = latency

    chunk_size = _pick("chunk_size")
    if chunk_size is not None:
        options_kwargs["default_chunk_size"] = chunk_size

    options = MockServerOptions(**options_kwargs)

    fallback = None
    if getattr(args, "fallback", None) is not None:
        fallback = args.fallback
    elif "fallback" in json_cfg and json_cfg["fallback"] is not None:
        fallback = json_cfg["fallback"]

    rules = None
    if getattr(args, "rules", None) is not None:
        rules = args.rules
    elif "rules" in json_cfg and json_cfg["rules"] is not None:
        rules = json_cfg["rules"]

    watch = False
    if getattr(args, "watch", False):
        watch = True
    elif "watch" in json_cfg and json_cfg["watch"]:
        watch = True

    return options, rules, fallback, watch


# ---------------------------------------------------------------------------
# 子命令：start
# ---------------------------------------------------------------------------

async def _cmd_start(args: argparse.Namespace) -> int:
    options, rules, fallback, watch = _merge_options(args)
    server = MockServer(options)
    if fallback is not None:
        server.fallback(fallback)
    if rules is not None:
        await server.load(rules)

    if options.log_level != "none":
        _print_banner(server, options, rules, watch)

    await server.run_until_shutdown(watch_path=rules if watch and rules is not None else None)
    return 0


def _print_banner(server: MockServer, options: MockServerOptions, rules: str | None, watch: bool) -> None:
    print()
    print("  \033[1m\033[36mllm-mock-api\033[0m")
    print()
    print(f"  Port       {options.port}")
    print(f"  Host       {options.host}")
    print(f"  Rules      {server.rule_count} loaded" + (f" ({rules})" if rules else ""))
    if options.default_latency and options.default_latency > 0:
        print(f"  Latency    {options.default_latency}ms per chunk")
    print(f"  Endpoints  {', '.join(server.routes)}")
    if watch:
        print(f"  Watch      enabled")
    print()


# ---------------------------------------------------------------------------
# 子命令：init
# ---------------------------------------------------------------------------

_EXAMPLE_CONFIG = """{
  "port": 8002,
  "host": "127.0.0.1",
  "log_level": "info",
  "default_latency": 50,
  "default_chunk_size": 50,
  "fallback": "Sorry, I don't understand your request. Try saying: hi, weather, joke, step, or echo.",
  "rules": "./rules.json5",
  "watch": true
}
"""

_EXAMPLE_RULES = """// ============================================================================
// llm-mock-api 默认规则文件（JSON5 语法 = JSON + 注释 + 尾逗号 + 裸 key）
// ============================================================================
//
// 顶层字段：
//   rules:      规则数组（按顺序匹配，第一条命中的生效）
//   templates:  （可选）可复用的回复模板，规则中用 "$模板名" 引用
//   fallback:   （可选）无规则匹配时的默认回复，可覆盖 config.json 中的 fallback
//
// 每条规则包含：
//   when:   匹配条件，三种写法：
//             "字符串"           → 对用户消息做大小写不敏感的包含匹配
//             "/正则/标志"       → 对用户消息做正则匹配（标志支持 i / m / s / u）
//             { message, model, system, format } → 多字段条件组合（全满足才命中）
//   reply:  匹配成功时的回复，两种写法：
//             "字符串" → 纯文本回复
//             { text, reasoning, tools } → 结构化回复（可包含工具调用、推理过程）
//   replies: （替代 reply）序列回复，每请求推进到下一条
//             数组元素可以是字符串或 { reply, latency?, chunkSize? }
//   times:  （可选）正整数，表示此规则最多匹配多少次后自动失效
//
// templates 中定义的回复可通过 "$模板名" 在 reply / replies 中引用
// ============================================================================

{
  // -------------------------------------------------------------------------
  // 模板：可复用的回复片段
  // -------------------------------------------------------------------------
  templates: {
    // 天气工具调用：回复中包含一个 tool_call
    weather_tool: {
      text: "Looking up the weather for you...",
      tools: [
        {
          name: "get_weather",
          args: { location: "Beijing", unit: "celsius" },
        },
      ],
    },
    // 标准的 "我无法回答" 模板
    cant_answer: {
      text: "Sorry, I can't help with that right now.",
      reasoning: "User asked something outside my scope.",
    },
  },

  // -------------------------------------------------------------------------
  // 全局 fallback（无规则匹配时返回）
  // -------------------------------------------------------------------------
  fallback: "Sorry, I don't understand. Try: hi, weather, joke, step, or echo.",

  // -------------------------------------------------------------------------
  // 规则列表（按顺序匹配）
  // -------------------------------------------------------------------------
  rules: [
    // [0] 工具结果：当请求携带 tool result（last_tool_call_id 存在）时，返回固定文本
    {
      when: { tool_call_id: "x" },
      reply: "Tool result received and processed.",
    },

    // [1] 基础：字符串匹配 → 文本回复
    {
      when: "hi",
      reply: "Hello! 👋 This is llm-mock-api. How can I help you today?",
    },
    {
      when: "hello",
      reply: "Hello! 👋 This is llm-mock-api. How can I help you today?",
    },
    {
      when: "你好",
      reply: "你好！我是 llm-mock-api，一个用于测试的 Mock LLM 服务器。",
    },

    // [2] 正则匹配：匹配 "echo xxx"，返回 echo 内容
    {
      when: "/^echo (.*)$/i",
      reply: "Echo mode active. Any message starting with 'echo' triggers this.",
    },

    // [3] 对象匹配：按请求中的模型名筛选
    {
      when: {
        message: "model check",
        model: "gpt-4",
      },
      reply: "Detected GPT-4 model request (object match by model field).",
    },

    // [4] 对象匹配：按 API 格式筛选（仅 OpenAI chat completions 路由生效）
    {
      when: {
        message: "openai only",
        format: "openai",
      },
      reply: "This reply only appears on the OpenAI /v1/chat/completions route.",
    },

    // [5] 结构化回复：text + reasoning（思考过程）
    {
      when: "joke",
      reply: {
        reasoning: "User wants a programmer joke. Make it short and relatable.",
        text: "Why did the developer go broke? Because he used up all his cache.",
      },
    },

    // [6] 结构化回复：工具调用（引用上方 templates.weather_tool）
    {
      when: "weather",
      reply: "$weather_tool",
    },

    // [7] 结构化回复：多个 tool calls
    {
      when: "multi tool",
      reply: {
        text: "I need to run several tools to answer this.",
        tools: [
          { name: "search_db", args: { query: "users" } },
          { name: "send_email", args: { to: "admin@example.com" } },
        ],
      },
    },

    // [8] 序列回复：同一关键词每次请求返回下一条
    {
      when: "step",
      replies: [
        "Step 1/3 — Initializing...",
        "Step 2/3 — Processing data...",
        "Step 3/3 — Done!",
      ],
    },

    // [9] 序列回复：带每步自定义 latency / chunkSize（模拟流式节奏）
    {
      when: "slow step",
      replies: [
        { reply: "Fast response (no delay).", latency: 0, chunkSize: 0 },
        { reply: "Medium response (200ms delay, chunked).", latency: 200, chunkSize: 20 },
        { reply: "Slow response (500ms delay, chunked).", latency: 500, chunkSize: 10 },
      ],
    },

    // [10] 次数限制：仅匹配 1 次后自动失效
    {
      when: "once",
      reply: "This rule triggers only ONCE — subsequent requests fall through.",
      times: 1,
    },

    // [11] 次数限制：仅匹配 3 次
    {
      when: "three times",
      reply: "This rule triggers up to 3 times.",
      times: 3,
    },

    // [12] 长文本：测试 SSE 流式输出（分块发送）
    {
      when: "long",
      reply: "This is a longer response to demonstrate SSE streaming behavior. When you request with stream=true, the server splits this text into chunks and sends them one by one with a small delay between each chunk. You can tune default_latency (milliseconds between chunks) and default_chunk_size (characters per chunk) in config.json per server, or override at the individual reply level inside a rule. Try it out: send 'long' with streaming enabled and watch the chunks arrive in real time.",
    },

    // [13] 模板引用：回复复用 templates.cant_answer
    {
      when: "unknown",
      reply: "$cant_answer",
    },
  ],
}
"""


async def _cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.dir or ".")
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / "config.json"
    rules_path = target / "rules.json5"

    if config_path.exists() and not args.force:
        print(f"[skip] {config_path} already exists (use --force to overwrite)")
    else:
        config_path.write_text(_EXAMPLE_CONFIG, encoding="utf-8")
        print(f"[created] {config_path}")

    if rules_path.exists() and not args.force:
        print(f"[skip] {rules_path} already exists (use --force to overwrite)")
    else:
        rules_path.write_text(_EXAMPLE_RULES, encoding="utf-8")
        print(f"[created] {rules_path}")

    print()
    print("Next steps:")
    print(f"  cd {target}")
    print("  llm-mock-api start   # 启动服务器")
    return 0


# ---------------------------------------------------------------------------
# 子命令：validate
# ---------------------------------------------------------------------------

async def _cmd_validate(args: argparse.Namespace) -> int:
    import json5

    path = Path(args.path)
    if not path.exists():
        print(f"[error] {path} does not exist")
        return 1

    # 收集所有 .json5 和 .json 文件
    files: list[Path] = []
    if path.is_file():
        if path.suffix in (".json5", ".json"):
            files.append(path)
    else:
        for p in sorted(path.rglob("*")):
            if p.is_file() and p.suffix in (".json5", ".json"):
                files.append(p)

    if not files:
        print(f"[warning] No rules files found at {path}")
        return 0

    total = 0
    failed = 0

    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json5.load(f)
            if not isinstance(data, dict):
                print(f"[fail] {fpath}: root must be an object")
                failed += 1
                continue
            if "rules" not in data:
                print(f"[fail] {fpath}: missing 'rules' key")
                failed += 1
                continue
            if not isinstance(data["rules"], list):
                print(f"[fail] {fpath}: 'rules' must be an array")
                failed += 1
                continue
            print(f"[ok] {fpath} ({len(data['rules'])} rules)")
            total += len(data["rules"])
        except Exception as exc:
            print(f"[fail] {fpath}: {exc}")
            failed += 1

    print()
    if failed == 0:
        print(f"All good — {len(files)} file(s), {total} rule(s)")
        return 0
    else:
        print(f"Failed: {failed}/{len(files)} file(s)")
        return 1


# ---------------------------------------------------------------------------
# CLI 构建
# ---------------------------------------------------------------------------

def main() -> None:
    # 简化：手动判断用户是否传了子命令
    # 如果 sys.argv[1] 是 "start"/"init"/"validate" → 用对应 parser
    # 否则 → 默认触发 start（模拟 commander.js 的 isDefault: true）
    subcommand = None
    if len(sys.argv) > 1 and sys.argv[1] in ("start", "init", "validate"):
        subcommand = sys.argv[1]
        rest_args = sys.argv[2:]
    else:
        subcommand = "start"  # 默认
        rest_args = sys.argv[1:]

    try:
        import asyncio
        if subcommand == "start":
            start_parser = argparse.ArgumentParser(prog="llm-mock-api start")
            _add_start_options(start_parser)
            args = start_parser.parse_args(rest_args)
            sys.exit(asyncio.run(_cmd_start(args)))
        elif subcommand == "init":
            init_parser = argparse.ArgumentParser(prog="llm-mock-api init")
            init_parser.add_argument("-d", "--dir", help="Output directory (default: current directory)")
            init_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
            args = init_parser.parse_args(rest_args)
            sys.exit(asyncio.run(_cmd_init(args)))
        elif subcommand == "validate":
            validate_parser = argparse.ArgumentParser(prog="llm-mock-api validate")
            validate_parser.add_argument("path", help="Path to rules file or directory")
            args = validate_parser.parse_args(rest_args)
            sys.exit(asyncio.run(_cmd_validate(args)))
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
