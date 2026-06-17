# llm-mock-api

> 本项目是 [theblixguy/llm-mock-server](https://github.com/theblixguy/llm-mock-server) 的 Python 重写版本，主要用于本人学习。  
> 目前功能版本略低于原项目，后续会逐步跟进并对齐上游更新。

一个用于测试的 Mock LLM 服务器，支持 OpenAI（v1/Responses；v1/chat/completions ）与 Anthropic 格式。

## 快速开始

```bash
# 安装依赖
uv pip install llm-mock-api

# 在当前目录生成 config.json 和 rules.json5
llm-mock-api init

# 启动服务
llm-mock-api start
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
- `config.json` — 服务器配置

| 字段 | 说明 |
|------|------|
| `port` / `host` | 监听地址，默认 `127.0.0.1:8002` |
| `log_level` | 日志级别（`none` / `error` / `info` / `debug`） |
| `default_latency` | SSE 流式输出的每块延迟（毫秒），默认 50 |
| `default_chunk_size` | SSE 流式输出的每块字符数，默认 50 |
| `fallback` | 无规则匹配时的默认回复 |
| `rules` | 规则文件路径（支持 `.json5` / `.json` / `.py`） |
| `watch` | 监听 `rules` 文件改动自动热重载 |

- `rules.json5` — 15 条开箱即用的演示规则

**即开即用：启动服务器后，你可以直接发送以下关键词测试**（详见下方"默认规则速查"）：

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
| `long` | SSE 流式输出（长文本分块） | `This is a longer response...`（分块节奏由 `config.json` 中的 `default_latency` / `default_chunk_size` 控制） |
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

## 安全

这是一个测试工具，不是生产服务。它被设计为在本地或 CI 中运行，加载你自己编写的规则文件。以下几点需要注意。

### 规则文件会执行代码

当你在 CLI 中传入 `.py` 规则文件时，它们会通过 Python 的动态导入机制加载，并与你的主进程具有相同的权限。仅加载你信任的文件。

### JSON5 规则文件仅为数据

JSON5 规则文件在加载时会通过 Pydantic 进行数据验证（相比原项目的 Zod，校验策略较为宽松），且永远不会执行代码。规则文件中的正则模式通过 Python 的 `re.compile()` 编译，本身是安全的，但如果你编写类似 `^(a+)+$` 这样的病态模式，可能会导致匹配挂起。请保持模式简洁。

### 请求限制

FastAPI 默认**不启用**请求体大小限制和速率限制。如果你需要将此服务部署到公共服务器上，强烈建议在前方使用反向代理（如 **Nginx / Traefik / Caddy**）来处理请求上限与访问控制。**我们更建议仅在本地使用**。

## 许可证

MIT License

Copyright (c) 2026 Ray3417

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.