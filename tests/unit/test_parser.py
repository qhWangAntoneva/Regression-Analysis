# encoding: utf-8
"""解析器单元测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data_io.encoding import detect_encoding
from src.data_io.parser import (
    FileParser,
    get_data_summary,
    infer_column_types,
    preview_dataframe,
)


class TestParseCSVBasic:
    """基本 CSV 解析测试。"""

    def test_parse_csv_basic(self, sample_csv_utf8: str):
        """测试基本 CSV 解析。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 5)
        assert list(df.columns) == ["id", "name", "age", "income", "city"]
        assert df["id"].tolist() == [1, 2, 3, 4, 5]

    def test_parse_csv_with_nrows(self, sample_csv_utf8: str):
        """测试 nrows 参数限制预览行数。"""
        parser = FileParser()
        df, encoding = parser.parse_csv(sample_csv_utf8, nrows=3)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 5)
        assert df["id"].tolist() == [1, 2, 3]
        assert encoding == "utf-8"


class TestParseCSVEncoding:
    """编码检测测试。"""

    def test_parse_csv_encoding_utf8(self, sample_csv_utf8: str):
        """测试 UTF-8 编码文件解析。"""
        parser = FileParser()
        df, encoding = parser.parse_csv(sample_csv_utf8)

        assert encoding == "utf-8"
        assert df["name"].tolist() == ["张三", "李四", "王五", "赵六", "陈七"]

    def test_parse_csv_encoding_gbk(self, sample_csv_gbk: str):
        """测试 GBK 编码文件解析。"""
        parser = FileParser()
        df, encoding = parser.parse_csv(sample_csv_gbk)

        assert encoding == "gbk"
        assert df["name"].tolist() == ["张三", "李四", "王五", "赵六", "陈七"]
        assert "city" in df.columns

    def test_encoding_detection_utf8(self, sample_csv_utf8: str):
        """测试 detect_encoding 函数识别 UTF-8。"""
        detected = detect_encoding(sample_csv_utf8)
        assert detected == "utf-8"

    def test_encoding_detection_gbk(self, sample_csv_gbk: str):
        """测试 detect_encoding 函数识别 GBK。"""
        detected = detect_encoding(sample_csv_gbk)
        assert detected == "gbk"


class TestParseCSVPreview:
    """预览模式测试。"""

    def test_preview_dataframe(self, sample_csv_utf8: str):
        """测试 preview_dataframe 函数。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        preview = preview_dataframe(df, n=3)
        assert isinstance(preview, pd.DataFrame)
        assert len(preview) == 3

    def test_preview_all_rows(self, sample_csv_utf8: str):
        """测试 preview_dataframe 返回全部数据。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        preview = preview_dataframe(df, n=100)
        assert len(preview) == 5

    def test_nrows_zero_returns_all(self, sample_csv_utf8: str):
        """测试 nrows=None 返回全部数据。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8, nrows=None)
        assert len(df) == 5


class TestInferColumnTypes:
    """类型推断测试。"""

    def test_infer_column_types_numeric(self, sample_csv_utf8: str):
        """测试数值列推断。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        types = infer_column_types(df)

        assert types["id"] in ("numeric", "id")  # id 可能是 numeric
        assert types["age"] == "numeric"
        assert types["income"] == "numeric"

    def test_infer_column_types_categorical(self, sample_csv_utf8: str):
        """测试分类列推断。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        types = infer_column_types(df)
        assert types["name"] in ("categorical", "id")  # name 可能是 id

    def test_infer_column_types_id_column(self):
        """测试 ID 列识别。"""
        import numpy as np

        df = pd.DataFrame(
            {
                "user_id": range(1, 101),
                "value": np.random.randn(100),
                "category": ["A", "B"] * 50,
            }
        )
        types = infer_column_types(df)
        assert types["user_id"] == "id"


class TestGetDataSummary:
    """数据摘要测试。"""

    def test_get_data_summary_shape(self, sample_csv_utf8: str):
        """测试数据摘要的行列信息。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        summary = get_data_summary(df)

        assert summary["n_rows"] == 5
        assert summary["n_cols"] == 5
        assert summary["memory_bytes"] > 0
        assert "memory_formatted" in summary

    def test_get_data_summary_missing_values(self, sample_csv_missing: str):
        """测试缺失值统计。"""
        parser = FileParser()
        df = parser.parse(sample_csv_missing)

        summary = get_data_summary(df)

        assert "missing_rates" in summary
        missing = summary["missing_rates"]
        # value 列有 2/5 缺失
        assert missing["value"] > 0
        # category 列有 1/5 缺失
        assert missing["category"] > 0
        # id 列无缺失
        assert missing["id"] == 0.0

    def test_get_data_summary_column_types(self, sample_csv_utf8: str):
        """测试数据摘要中的列类型信息。"""
        parser = FileParser()
        df = parser.parse(sample_csv_utf8)

        summary = get_data_summary(df)

        assert "column_types" in summary
        assert len(summary["column_types"]) == 5


class TestDetectEncoding:
    """编码检测函数测试。"""

    def test_detect_encoding_utf8(self, sample_csv_utf8: str):
        """测试 detect_encoding 检测 UTF-8。"""
        detected = detect_encoding(sample_csv_utf8)
        assert detected == "utf-8"

    def test_detect_encoding_gbk(self, sample_csv_gbk: str):
        """测试 detect_encoding 检测 GBK。"""
        detected = detect_encoding(sample_csv_gbk)
        assert detected == "gbk"

    def test_detect_encoding_file_not_found(self):
        """测试文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            detect_encoding("/nonexistent/path/file.csv")
