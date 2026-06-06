"""logger 模块单元测试。"""

from __future__ import annotations

import pytest

from llm_mock_api.logger import Logger


class TestLevelFilter:
    """测试日志级别过滤。"""

    def test_info_blocks_debug(self, capsys: pytest.CaptureFixture[str]) -> None:
        """info 级别应屏蔽 debug。"""
        logger = Logger("info")
        logger.debug("hidden")
        logger.info("visible")
        captured = capsys.readouterr()
        assert "hidden" not in captured.out
        assert "visible" in captured.out

    def test_none_blocks_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        """none 级别应屏蔽所有日志。"""
        logger = Logger("none")
        logger.error("e")
        logger.warn("w")
        logger.info("i")
        logger.debug("d")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_all_allows_all(self, capsys: pytest.CaptureFixture[str]) -> None:
        """all 级别应允许所有日志。"""
        logger = Logger("all")
        logger.error("error-msg")
        logger.warn("warn-msg")
        logger.info("info-msg")
        logger.debug("debug-msg")
        captured = capsys.readouterr()
        assert "info-msg" in captured.out
        assert "debug-msg" in captured.out
        assert "error-msg" in captured.err
        assert "warn-msg" in captured.err


class TestOutputStream:
    """测试输出流选择。"""

    def test_streams(self, capsys: pytest.CaptureFixture[str]) -> None:
        """error/warn 去 stderr，info/debug 去 stdout。"""
        logger = Logger("debug")
        logger.error("err")
        logger.warn("wrn")
        logger.info("inf")
        logger.debug("dbg")
        captured = capsys.readouterr()
        assert "err" in captured.err and "err" not in captured.out
        assert "wrn" in captured.err and "wrn" not in captured.out
        assert "inf" in captured.out and "inf" not in captured.err
        assert "dbg" in captured.out and "dbg" not in captured.err


class TestOutputContent:
    """测试输出内容格式。"""

    def test_contains_message_and_label(self, capsys: pytest.CaptureFixture[str]) -> None:
        """输出应包含消息文本和级别标签。"""
        logger = Logger("info")
        logger.info("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out
        assert "INFO" in captured.out

    def test_contains_timestamp(self, capsys: pytest.CaptureFixture[str]) -> None:
        """输出应包含 ISO 格式时间戳。"""
        logger = Logger("info")
        logger.info("ts")
        captured = capsys.readouterr()
        assert "T" in captured.out
        assert "Z" in captured.out


class TestMultiArgs:
    """测试多参数支持。"""

    def test_extra_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        """额外参数应被拼接到输出中。"""
        logger = Logger("info")
        logger.info("user", "alice", {"host": "localhost", "port": 8080})
        captured = capsys.readouterr()
        assert "user" in captured.out
        assert "alice" in captured.out
        assert "localhost" in captured.out
        assert "8080" in captured.out


class TestMultiInstance:
    """测试多实例隔离。"""

    def test_independent_levels(self, capsys: pytest.CaptureFixture[str]) -> None:
        """不同 logger 实例应有独立的级别设置。"""
        quiet = Logger("error")
        verbose = Logger("debug")
        quiet.info("q")
        verbose.info("v")
        captured = capsys.readouterr()
        assert "q" not in captured.out
        assert "v" in captured.out
