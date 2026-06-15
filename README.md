# llm-mock-api

一个用于测试的 Mock LLM 服务器，支持 OpenAI、Anthropic 和 Responses API 格式。

## 快速开始

```bash
uv sync
```

## CLI 命令

### `start`（默认）— 启动服务器

```bash
# 最简方式（自动从 ./config.json 读取配置）
llm-mock-api start

# 或直接（不传子命令也等同于 start）
llm-mock-api

# 指定端口和规则文件
llm-mock-api start -p 8002 -r ./rules.json5

# 开启 watch 模式（规则文件变化时自动重载）
llm-mock-api start -r ./rules.json5 -w

# 设置日志级别
llm-mock-api start --log-level info
```

**所有选项：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--config` | JSON 配置文件路径 | `./config.json` |
| `-p, --port` | 监听端口（0 = 随机分配） | `0` |
| `-H, --host` | 绑定地址 | `127.0.0.1` |
| `-r, --rules` | JSON5 规则文件或目录路径 | 无 |
| `-l, --latency` | SSE chunk 间延迟（毫秒） | `0` |
| `-c, --chunk-size` | SSE 分块大小（字符数） | `0` |
| `-f, --fallback` | 无规则匹配时的默认回复 | 内置默认文案 |
| `-w, --watch` | 监听 rules 文件变化并热重载 | 关闭 |
| `--log-level` | 日志级别（none/error/warning/info/debug/all） | `none` |

> **配置优先级**：CLI 参数 > `config.json` > MockServerOptions dataclass 默认值

### `init` — 生成示例文件

```bash
# 在当前目录生成 config.json 和 rules.json5
llm-mock-api init

# 指定输出目录
llm-mock-api init -d ./my-project

# 覆盖已有文件
llm-mock-api init --force
```

生成的示例：
- `config.json` — 服务器配置（端口 8002、SSE 流式输出、watch 自动重载）
- `rules.json5` — 15 条开箱即用的演示规则（覆盖全部核心功能）

**即开即用：启动服务器后，你可以直接发送以下关键词测试**（详见下方"默认规则速查"章节）：

| 关键词 | 演示功能 | 返回内容 |
|--------|---------|---------|
| `hi` / `hello` / `你好` | 基础字符串匹配 | 英文：`Hello! 👋 This is llm-mock-api. How can I help you today?`；中文：`你好！我是 llm-mock-api，一个用于测试的 Mock LLM 服务器。` |
| `echo test 456` | 正则匹配（`/^echo (.*)$/i`） | `Echo mode active. Any message starting with 'echo' triggers this.`（提示：消息中不能含 hi/hello，否则会被前序字符串规则先匹配） |
| `model check`（模型设为 gpt-4） | 对象匹配（按 model 筛选） | `Detected GPT-4 model request (object match by model field).` |
| `openai only`（请求 `/v1/chat/completions`） | 对象匹配（按 format 筛选） | `This reply only appears on the OpenAI /v1/chat/completions route.`（仅 Chat Completions 路由生效） |
| `joke` | 结构化回复（`text` + `reasoning`，仅 Responses 格式返回 reasoning 字段） | reasoning: `User wants a programmer joke. Make it short and relatable.`；text: `Why did the developer go broke? Because he used up all his cache.`（Chat Completions 仅返回 text 部分） |
| `weather` | 模板引用 + 工具调用（`$weather_tool`） | content: `Looking up the weather for you...`；tool_call: `get_weather(location="Beijing", unit="celsius")` |
| `multi tool` | 多工具调用 | content: `I need to run several tools to answer this.`；tool_calls: `search_db(query="users")` + `send_email(to="admin@example.com")` |
| `step` | 序列回复（replies 数组） | 第1次：`Step 1/3 — Initializing...`；第2次：`Step 2/3 — Processing data...`；第3次：`Step 3/3 — Done!` |
| `slow step` | 序列回复 + 自定义 latency/chunkSize | `Fast response (no delay).`（0ms delay，无分块）→ `Medium response (200ms delay, chunked).`（200ms，20 char/chunk）→ `Slow response (500ms delay, chunked).`（500ms，10 char/chunk） |
| `once` | 次数限制 `times: 1` | 第1次：`This rule triggers only ONCE — subsequent requests fall through.`；第2次起：fallback |
| `three times` | 次数限制 `times: 3` | 前3次均返回：`This rule triggers up to 3 times.`；第4次起：fallback |
| `long` | SSE 流式输出（长文本分块） | `This is a longer response to demonstrate SSE streaming behavior. When you request with stream=true, the server splits this text into chunks and sends them one by one with a small delay between each chunk. You can tune default_latency (milliseconds between chunks) and default_chunk_size (characters per chunk) in config.json per server, or override at the individual reply level inside a rule. Try it out: send 'long' with streaming enabled and watch the chunks arrive in real time.`（分块节奏由 `config.json` 中的 `default_latency` / `default_chunk_size` 控制） |
| `unknown` | 模板引用（`$cant_answer$`，仅 Responses 格式返回 reasoning 字段） | reasoning: `User asked something outside my scope.`；text: `Sorry, I can't help with that right now.`（Chat Completions 仅返回 text 部分） |
| （未匹配） | fallback | `Sorry, I don't understand. Try: hi, weather, joke, step, or echo.` |

### `validate` — 检查规则文件语法

```bash
# 检查单个文件
llm-mock-api validate ./rules.json5

# 检查整个目录（递归查找 .json5/.json 文件）
llm-mock-api validate ./rules/
```

检查内容：
- 文件是否为合法 JSON5
- 根节点是否为对象
- 是否包含 `rules` 数组

## 配置文件 `config.json`

这是 `llm-mock-api init` 生成的默认配置（已调好即开即用）：

```json
{
  "port": 8002,
  "host": "127.0.0.1",
  "log_level": "info",
  "default_latency": 50,
  "default_chunk_size": 50,
  "fallback": "Sorry, I don't understand your request. Try saying: hi, weather, joke, step, or echo.",
  "rules": "./rules.json5",
  "watch": true
}
```

| 字段 | 说明 |
|------|------|
| `port` / `host` | 监听地址，默认 `127.0.0.1:8002` |
| `log_level` | 日志级别（`none` / `error` / `info` / `debug`） |
| `default_latency` | SSE 流式输出的每块延迟（毫秒），默认 50 |
| `default_chunk_size` | SSE 流式输出的每块字符数，默认 50 |
| `fallback` | 无规则匹配时的默认回复 |
| `rules` | 规则文件路径（支持 `.json5` / `.json` / `.py`） |
| `watch` | 监听 `rules` 文件改动自动热重载 |

## Python API

```python
import asyncio
from llm_mock_api import MockServer

# 方式一：从 JSON 配置文件启动
async def main():
    server = await MockServer.from_json_config("config.json")
    await server.run_until_shutdown()

asyncio.run(main())

# 方式二：编程式配置（展示全功能）
from llm_mock_api import MockServer, MockServerOptions
from llm_mock_api.types.rule import MatchObject
from llm_mock_api.types.reply import ReplyObject, ToolCall
import re

async def main():
    # 1. 启动服务器
    opts = MockServerOptions(
        port=8002,
        host="127.0.0.1",
        log_level="info",
        default_latency=50,
        default_chunk_size=50,
    )
    server = MockServer(opts)

    # 2. 设置 fallback（无规则匹配时的默认回复）
    server.fallback("Sorry, no matching rule. Try: hi, weather, joke, step")

    # 3. 基础字符串匹配 + 文本回复
    server.when("hello").reply("world!")

    # 4. 正则匹配（忽略大小写）
    server.when(re.compile(r"^echo", re.IGNORECASE)).reply(
        "Echo mode active — message echoed back"
    )

    # 5. 对象匹配：按 model 字段筛选
    server.when(MatchObject(message="model check", model="gpt-4")).reply(
        "This reply only appears when the request model is gpt-4"
    )

    # 6. 结构化回复：text + reasoning（思考过程）
    server.when("joke").reply(
        ReplyObject(
            text="Why did the developer go broke? Because he used up all his cache.",
            reasoning="User asked for a joke — pick a programmer-related one.",
        )
    )

    # 7. 结构化回复：工具调用
    server.when("weather").reply(
        ReplyObject(
            text="Let me check the weather for you.",
            tools=[ToolCall(name="get_weather", args={"location": "Beijing", "unit": "celsius"})],
        )
    )

    # 8. 序列回复：每次请求推进到下一条，耗尽后重复最后一条
    server.when("step").reply_sequence([
        "Step 1: Initialize",
        "Step 2: Process",
        "Step 3: Done",
    ])

    # 9. 次数限制：仅匹配 1 次后自动失效
    server.when("once").reply("This rule fires only once!").times(1)

    # 10. 运行服务器
    print(f"Server running at {server.url}")
    await server.run_until_shutdown()

asyncio.run(main())
```