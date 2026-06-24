# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式。

## [0.1.1] - 2026-06-24

与上游 [theblixguy/llm-mock-server@v1.1.0](https://github.com/theblixguy/llm-mock-server/tree/v1.1.0) 功能对齐的小版本发布。

### 新增

- **`server_tool` 匹配条件**：在 `MatchObject` / 规则文件中新增 `server_tool` 字段，可按 Anthropic 内置工具（如 `web_search_20250305`）或 OpenAI 工具类型（如 `web_search` / `file_search`）精确匹配请求
- **`when_server_tool()` API**：`server.when_server_tool("web_search_20250305")` 快捷绑定
- **GPT-5 family `custom` tool 识别**：`/v1/chat/completions` 中 `type: "custom"` 的工具会被正常解析并放入 `tool_names`
- **重试检测**：基于请求头 `x-stainless-retry-count` 识别 SDK 自动重试 — 不消耗规则 `remaining`、不推进 `replies` 序列

### 修复

- **`/v1/responses` 请求体中的 `input` 数组因 Pydantic 嵌套模型导致解析为 empty（全部 fallback）** — 现改为保持原始 `dict`，由 parse 阶段的 `FunctionCallOutputSchema` / `FunctionCallInputSchema` / `InputMessageSchema` 安全解析（与 Zod `safeParse` 语义一致）
- **Anthropic tools 同样的嵌套模型解析问题** — 保持为 `list[dict[str, Any]]`，由 `ToolDefinitionSchema` / `ServerToolSchema` 分别 safeParse，正确区分 user tool 与 server tool
- **Windows GBK 控制台打印 emoji 触发 `UnicodeEncodeError`**：示例规则回复、测试期望值中的 👋 已移除为纯文本

## [0.1.0] - 2026-06-17

首个正式发布版本。基于 [theblixguy/llm-mock-server](https://github.com/theblixguy/llm-mock-server) 的 Python 重写。

### 新增

- **CLI 工具**：`llm-mock-api start` / `init` / `validate` 三个子命令
- **MockServer 核心**：异步 FastAPI 服务器，支持 JSON 配置文件启动
- **规则引擎**：按顺序匹配，支持字符串、正则、结构化对象、lambda 谓词
- **三种 API 格式**：
  - OpenAI `/v1/chat/completions`
  - OpenAI `/v1/responses`
  - Anthropic `/v1/messages`
- **规则匹配能力**：
  - 字符串包含匹配（大小写不敏感）
  - 正则表达式匹配
  - 结构化 `MatchObject`（model / format / system / tool_name / tool_call_id / predicate）
  - 模板引用（`$模板名`）
- **回复能力**：
  - 纯文本回复
  - 结构化回复（`text` + `reasoning` + `tools`）
  - 序列回复（`replies` 数组，每请求推进）
  - 次数限制（`times`）
  - SSE 流式输出（可配置延迟和分块大小）
  - Fallback 默认回复
- **规则来源**：JSON5 文件、Python 文件、编程式 API
- **Python API**：`MockServer` / `MockServerOptions` / `MatchObject` / `ReplyObject` / `ToolCall`
- **日志系统**：分级日志（none / error / warning / info / debug / all）
- **请求历史记录**
- **Watch 模式**：规则文件变化时自动重载
- **MIT 许可证**

### 测试

- 214 个 pytest 单元/集成测试
- 32 个端到端测试（OpenAI SDK 直连验证）
