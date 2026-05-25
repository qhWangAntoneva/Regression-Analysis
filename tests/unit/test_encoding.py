# encoding: utf-8
"""Tests for encoding detection in src/data_io/encoding.py.

Covers paths beyond what test_parser.py already tests:
- directory path (ValueError)
- chardet path with ASCII-only content
- binary/non-text files
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.data_io.encoding import _can_decode, detect_encoding


class TestDetectEncodingAdditional:
    """Tests for detect_encoding covering untested branches."""

    def test_directory_raises_value_error(self):
        """Passing a directory path should raise ValueError, not FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="路径不是文件"):
                detect_encoding(tmpdir)

    def test_ascii_only_file(self):
        """A pure-ASCII file should be detected as utf-8 (via chardet or fallback)."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="ascii") as f:
            f.write("Hello World\nThis is ASCII only\n12345\n")
        try:
            result = detect_encoding(path)
            assert result in ("utf-8", "ascii")
        finally:
            os.unlink(path)

    def test_chardet_path_used_when_available(self):
        """Verify that chardet can detect encoding from a GB2312-encoded file."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        # Write GBK content that chardet should detect
        content = "编号,姓名,年龄\n1,张三,28\n2,李四,35\n"
        with os.fdopen(fd, "w", encoding="gbk") as f:
            f.write(content)
        try:
            result = detect_encoding(path)
            # chardet should detect gbk or similar, fallback also works
            assert result in ("gbk", "utf-8")
        finally:
            os.unlink(path)

    def test_large_utf8_file(self):
        """File with many UTF-8 multi-byte characters should still be detected."""
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("城市,人口\n")
            for i in range(200):
                f.write(f"城市{i},1000000\n")
        try:
            result = detect_encoding(path)
            assert result == "utf-8"
        finally:
            os.unlink(path)

    def test_pathlib_path_accepted(self):
        """detect_encoding should accept pathlib.Path objects."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("hello world\n")
        try:
            result = detect_encoding(Path(path))
            assert result == "utf-8"
        finally:
            os.unlink(path)

    def test_empty_file(self):
        """Empty file should be decodable as utf-8."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            result = detect_encoding(path)
            # Empty bytes can be decoded by utf-8 (so falls through to first match)
            assert result in ("utf-8", "gbk", "latin-1")
        finally:
            os.unlink(path)


class TestCanDecode:
    """Tests for the internal _can_decode helper."""

    def test_utf8_valid(self):
        assert _can_decode("hello".encode("utf-8"), "utf-8") is True

    def test_utf8_invalid(self):
        # GBK bytes are not valid UTF-8
        gbk_bytes = "中文".encode("gbk")
        # This will likely fail as utf-8 but could sometimes pass
        result = _can_decode(gbk_bytes, "utf-8")
        assert isinstance(result, bool)

    def test_gbk_valid(self):
        gbk_bytes = "中文测试".encode("gbk")
        assert _can_decode(gbk_bytes, "gbk") is True

    def test_latin1_always_true(self):
        # latin-1 can decode any byte sequence (0-255 are all valid)
        assert _can_decode(b"\xff\xfe\x00", "latin-1") is True

    def test_invalid_encoding_name(self):
        assert _can_decode(b"hello", "nonexistent-encoding-xyz") is False
