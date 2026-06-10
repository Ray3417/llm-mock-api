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
| `hi` / `hello` / `你好` | 基础字符串匹配 | 问候文本回复 |
| `echo hello` | 正则匹配（`/^echo (.*)$/i`） | echo 模式提示 |
| `model check`（模型设为 gpt-4） | 对象匹配（按 model 筛选） | 模型条件命中提示 |
| `openai only`（请求 `/v1/chat/completions`） | 对象匹配（按 format 筛选） | 路由条件命中提示 |
| `joke` | 结构化回复（`text` + `reasoning`） | 带思考过程的笑话 |
| `weather` | 模板引用 + 工具调用 | `$weather_tool` 模板 → get_weather 工具 |
| `multi tool` | 多工具调用 | 两个工具调用并行 |
| `step` | 序列回复（replies 数组） | 依次返回 Step 1/2/3 |
| `slow step` | 序列回复 + 自定义 latency/chunkSize | 带节奏的分步回复 |
| `once` | 次数限制 `times: 1` | 仅第一次匹配 |
| `three times` | 次数限制 `times: 3` | 最多匹配 3 次 |
| `long` | SSE 流式输出（长文本分块） | 长文本分块发送 |
| `unknown` | 模板引用（`$cant_answer`） | 标准拒绝模板 |
| （未匹配） | fallback | 友好的 fallback 提示 |

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

# 方式二：编程式配置
from llm_mock_api import MockServerOptions

opts = MockServerOptions(port=8002, log_level="info")
server = MockServer(opts)
server.when("hello").reply("world")
asyncio.run(server.run_until_shutdown())
```

## 开发命令

| 命令 | 说明 |
|------|------|
| `uv sync` | 安装依赖 |
| `uv run pytest` | 运行测试 |
| `uv run ruff format .` | 格式化代码 |
| `uv run ruff check .` | 检查代码规范 |
| `uv run mypy src/` | 类型检查 |
