# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 格式。

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
