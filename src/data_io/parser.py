"""
文件解析模块

提供统一的文件解析入口，支持 CSV 和 Excel 格式。
自动检测编码（UTF-8/GBK），支持预览模式（nrows 限制）。
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_io.encoding import detect_encoding


class FileParser:
    """文件解析器。

    根据文件扩展名自动选择解析逻辑。

    Usage:
        parser = FileParser()
        df = parser.parse("data.csv", nrows=100)
    """

    def parse(
        self,
        filepath: str | Path,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """主解析入口。

        根据文件扩展名自动选择解析器。

        Args:
            filepath: 文件路径。
            nrows: 限制读取行数（预览模式），None 表示读取全部。

        Returns:
            解析后的 DataFrame。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 不支持的文件格式或解析失败。
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        suffix = path.suffix.lower()

        if suffix in (".csv", ".tsv", ".txt"):
            return self.parse_csv(filepath, nrows=nrows)[0]
        elif suffix in (".xls", ".xlsx"):
            return self.parse_excel(filepath, nrows=nrows)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}（支持: .csv, .tsv, .txt, .xls, .xlsx）")

    def parse_csv(
        self,
        filepath: str | Path,
        nrows: int | None = None,
        encoding: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """解析 CSV 文件。

        自动检测编码（UTF-8/GBK），支持 nrows 限制预览行数。

        Args:
            filepath: 文件路径。
            nrows: 限制读取行数。
            encoding: 指定编码。None 表示自动检测。

        Returns:
            (DataFrame, encoding) 元组。
        """
        path = Path(filepath)

        # 确定编码
        if encoding is None:
            encoding = detect_encoding(filepath)

        # 确定分隔符
        suffix = path.suffix.lower()
        if suffix == ".tsv":
            sep = "\t"
        elif suffix == ".txt":
            sep = self._detect_separator(filepath, encoding)
        else:
            sep = ","

        # 解析 CSV
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=encoding,
                nrows=nrows,
                dtype_backend="numpy_nullable",
                engine="python" if encoding == "gbk" else "c",
                on_bad_lines="warn",
            )
        except UnicodeDecodeError:
            # 如果指定编码失败，回退到另一种常见编码
            fallback = "gbk" if encoding == "utf-8" else "utf-8"
            warnings.warn(
                f"编码 {encoding} 解码失败，回退到 {fallback}",
                UserWarning,
                stacklevel=2,
            )
            df = pd.read_csv(
                path,
                sep=sep,
                encoding=fallback,
                nrows=nrows,
                dtype_backend="numpy_nullable",
                engine="python" if encoding == "gbk" else "c",
                on_bad_lines="warn",
            )
            encoding = fallback

        return df, encoding

    def parse_excel(
        self,
        filepath: str | Path,
        nrows: int | None = None,
    ) -> pd.DataFrame:
        """解析 Excel 文件。

        读取第一个 sheet。

        Args:
            filepath: 文件路径。
            nrows: 限制读取行数。

        Returns:
            解析后的 DataFrame。
        """
        path = Path(filepath)
        df = pd.read_excel(path, sheet_name=0, nrows=nrows, dtype_backend="numpy_nullable")
        return df

    @staticmethod
    def _detect_separator(filepath: str | Path, encoding: str) -> str:
        """检测文本文件的分隔符（逗号、制表符、分号）。"""
        path = Path(filepath)
        try:
            with path.open("r", encoding=encoding) as f:
                first_line = f.readline()
        except UnicodeDecodeError:
            # fallback for encoding issues
            raw = path.read_bytes()[:4096]
            try:
                first_line = raw.decode(encoding, errors="replace")
            except Exception:
                first_line = raw.decode("utf-8", errors="replace")

        for sep, name in [("\t", "tab"), (";", "semicolon"), (",", "comma")]:
            if sep in first_line:
                return sep
        return ","  # default


def preview_dataframe(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """返回前 n 行预览。"""
    return df.head(n)


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """简单列类型推断。

    返回 dict: col_name -> 'numeric' / 'categorical' / 'text' / 'id'

    Args:
        df: 输入 DataFrame。

    Returns:
        列名到推断类型的映射。
    """
    types: dict[str, str] = {}
    nrows = len(df)

    for col in df.columns:
        series = df[col]

        # 检查是否是 pandas nullable numeric types
        if pd.api.types.is_numeric_dtype(series):
            # 检查是否是 id 列（列名模式 + 唯一值 == 总行数）
            col_lower = str(col).lower().strip()
            id_patterns = ("id", "code", "num", "no.", "number", "序号", "编号", "代码")
            is_id_name = any(col_lower.startswith(p) or col_lower.endswith(p) for p in id_patterns)

            if is_id_name and series.nunique() == nrows:
                types[str(col)] = "id"
            else:
                types[str(col)] = "numeric"
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # 尝试数值转换
            try:
                converted = pd.to_numeric(series.dropna(), errors="raise")
                # 如果能转换，且不是 id 列
                col_lower = str(col).lower().strip()
                id_patterns = ("id", "code", "num", "no.", "number", "序号", "编号", "代码")
                is_id_name = any(col_lower.startswith(p) or col_lower.endswith(p) for p in id_patterns)  # noqa: E501
                if is_id_name and len(converted) == nrows:
                    types[str(col)] = "id"
                else:
                    types[str(col)] = "numeric"
            except (ValueError, TypeError):
                # 检查是否可能是 id 列
                unique_ratio = series.nunique() / max(nrows, 1)
                if unique_ratio > 0.9:
                    types[str(col)] = "id"
                else:
                    types[str(col)] = "categorical"
        elif pd.api.types.is_bool_dtype(series):
            types[str(col)] = "categorical"
        else:
            # 分类、时间等
            types[str(col)] = "categorical"

    return types


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """数据摘要。

    返回包含行数、列数、各列缺失率、各列类型、内存占用等信息的字典。

    Args:
        df: 输入 DataFrame。

    Returns:
        数据摘要字典。
    """
    n_rows, n_cols = df.shape
    memory_bytes = df.memory_usage(deep=True).sum()

    missing_info: dict[str, float] = {}
    col_types = infer_column_types(df)

    for col in df.columns:
        missing_rate = float(df[col].isna().sum() / max(n_rows, 1))
        missing_info[str(col)] = round(missing_rate, 4)

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_bytes": int(memory_bytes),
        "memory_formatted": _format_bytes(memory_bytes),
        "missing_rates": missing_info,
        "column_types": col_types,
    }


def _format_bytes(size: int) -> str:
    """将字节数格式化为人类可读的字符串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"
