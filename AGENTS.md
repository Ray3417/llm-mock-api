# llm-mock-api 开发规范

## 编码风格

### 代码重构顺序

重构 TypeScript 时，保持与原文件**完全一致**的顺序。使用 `from __future__ import annotations` + 字符串引用实现前向引用。

### 充分利用 Python 新特性

优先使用 Python 现代特性：`frozen=True` + `slots=True`、`Literal`、`ReadOnly`、`Protocol`、`default_factory`、`match/case`。避免 `Optional[x]`，用 `x | None` 替代。

**类型导入规范**：抽象基类优先从 `collections.abc` 导入（Python 3.9+ 支持泛型），如 `Sequence`、`Awaitable`、`Callable`、`Mapping`、`Iterable`；仅从 `typing` 导入类型系统专用项，如 `Any`、`Literal`、`NoReturn`、`Protocol`、`runtime_checkable`。使用 Python 3.12+ `type` 语句定义类型别名，替代 `TypeAlias`。

**行为契约使用 Protocol**：描述行为接口而非具体数据结构时，使用 `class Protocol(Protocol)` 或 `@runtime_checkable class Protocol(Protocol)`。纯数据结构使用 `@dataclass`。选择标准：有方法定义（`reply()`、`match()`、`respond()` 等）→ `Protocol`；仅描述字段 → `dataclass`。

### 注释语言

所有代码内的注释使用**中文**。保留英文的仅限：类名、变量名、函数名、类型注解中的类型引用。

### 注释翻译原则

翻译技术注释时，需结合对项目架构和上下文的理解，采用**意译**而非**直译**。例如：
- `wire format` → "线路格式"（而非字面翻译"电线格式"）
- `normalised view` → "规范化视图"（而非"归一化视图"）
- 确保翻译后的注释在技术层面准确且易于理解。
