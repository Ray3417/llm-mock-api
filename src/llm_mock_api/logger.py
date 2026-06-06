from __future__ import annotations

import sys
from datetime import datetime, timezone
from typing import Literal, Any

# ANSI 颜色转义码（替代 picocolors）
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _red(text: str) -> str:
    return f"{_RED}{text}{_RESET}"


def _yellow(text: str) -> str:
    return f"{_YELLOW}{text}{_RESET}"


def _cyan(text: str) -> str:
    return f"{_CYAN}{text}{_RESET}"


def _bold(text: str) -> str:
    return f"{_BOLD}{text}{_RESET}"


def _dim(text: str) -> str:
    return f"{_DIM}{text}{_RESET}"


def _iso_now() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串（与 TS 的 toISOString() 一致）。"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


LEVEL_PRIORITY = {
    "none": 0,
    "error": 1,
    "warning": 2,
    "info": 3,
    "debug": 4,
    "all": 5,
}

# 日志级别，从 "none"（静默）到 "all"（全部）
type LogLevel = Literal["none", "error", "warning", "info", "debug", "all"]

_LEVEL_STYLE = {
    "error": {"label": _red(_bold("ERROR")), "symbol": _red("✗")},
    "warn": {"label": _yellow(_bold("WARN")), "symbol": _yellow("!")},
    "info": {"label": _cyan("INFO"), "symbol": _cyan("●")},
    "debug": {"label": _dim("DEBUG"), "symbol": _dim("·")},
}

type _ConsoleMethod = Literal["error", "warn", "log"]

_LEVEL_CONFIG: dict[str, dict[str, Any]] = {
    "error": {"priority": LEVEL_PRIORITY["error"], "method": "error"},
    "warn": {"priority": LEVEL_PRIORITY["warning"], "method": "warn"},
    "info": {"priority": LEVEL_PRIORITY["info"], "method": "log"},
    "debug": {"priority": LEVEL_PRIORITY["debug"], "method": "log", "dim": True},
}


class Logger:
    def __init__(self, level: LogLevel = "info") -> None:
        self.level: LogLevel = level
        self._threshold: int = LEVEL_PRIORITY[level]

    def _log(self, key: str, msg: str, args: tuple[Any, ...]) -> None:
        config = _LEVEL_CONFIG[key]
        if self._threshold < config["priority"]:
            return
        style = _LEVEL_STYLE[key]
        text = _dim(msg) if config.get("dim") else msg
        prefix = f"{_dim(_iso_now())} {style['symbol']} {style['label']} {text}"
        stream = sys.stderr if config["method"] in ("error", "warn") else sys.stdout
        print(prefix, *args, file=stream, sep=" ")

    def error(self, msg: str, *args: Any) -> None:
        self._log("error", msg, args)

    def warn(self, msg: str, *args: Any) -> None:
        self._log("warn", msg, args)

    def info(self, msg: str, *args: Any) -> None:
        self._log("info", msg, args)

    def debug(self, msg: str, *args: Any) -> None:
        self._log("debug", msg, args)
