"""
编码检测模块

自动检测文本文件的字符编码。尝试顺序: UTF-8 -> GBK -> latin-1。
每个编码尝试读取文件前 10000 字节验证可行性，返回最可能的编码名。
"""

from __future__ import annotations

from pathlib import Path

try:
    import chardet
except ImportError:
    chardet = None  # type: ignore[assignment]


def detect_encoding(filepath: str | Path) -> str:
    """检测文件编码。

    使用 chardet 库（如果可用）进行智能检测，否则回退到尝试常见编码。
    尝试顺序: utf-8 -> gbk -> latin-1。

    Args:
        filepath: 文件路径。

    Returns:
        最可能的编码名称（小写字符串）。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 无法确定编码。
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    if not path.is_file():
        raise ValueError(f"路径不是文件: {filepath}")

    # 先尝试 chardet 智能检测
    if chardet is not None:
        try:
            raw_data = path.read_bytes()[:10000]
            result = chardet.detect(raw_data)
            detected = result.get("encoding", "")
            if detected:
                # normalize encoding names
                detected = detected.lower().replace("-", "")
                if detected in ("utf8", "utf-8", "ascii"):
                    if _can_decode(raw_data, "utf-8"):
                        return "utf-8"
                elif detected in ("gbk", "gb2312", "gb18030", "cp936"):
                    if _can_decode(raw_data, "gbk"):
                        return "gbk"
                elif _can_decode(raw_data, detected):
                    return detected
        except Exception:
            pass

    # 回退: 手动尝试常见编码
    raw_data = path.read_bytes()[:10000]

    for encoding in ("utf-8", "gbk", "latin-1"):
        if _can_decode(raw_data, encoding):
            return encoding

    raise ValueError(f"无法确定文件的编码: {filepath}")


def _can_decode(data: bytes, encoding: str) -> bool:
    """测试字节数据能否用指定编码解码。"""
    try:
        data.decode(encoding)
        return True
    except (UnicodeDecodeError, LookupError):
        return False
