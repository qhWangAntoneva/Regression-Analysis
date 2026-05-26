"""pytest 配置和 fixtures。"""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Return a 100-row, 8-column synthetic DataFrame."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "y": 2 + 0.5 * rng.normal(0, 1, 100) + rng.normal(0, 0.5, 100),
        "x1": rng.normal(0, 1, 100),
        "x2": rng.uniform(0, 1, 100),
        "x3": rng.choice(["A", "B", "C"], 100),
        "x4": rng.normal(10, 2, 100),
        "id": range(100),
        "cat1": rng.integers(0, 2, 100),
    })
    df.loc[0:4, "x4"] = np.nan
    return df


@pytest.fixture
def sample_csv_path(sample_df):
    """Save sample_df to a temporary CSV and return the path."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".csv")
    sample_df.to_csv(path, index=False)
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_csv_utf8() -> str:
    """创建一个 UTF-8 编码的 CSV 临时文件。"""
    content = [
        ["id", "name", "age", "income", "city"],
        ["1", "张三", "28", "15000", "北京"],
        ["2", "李四", "35", "22000", "上海"],
        ["3", "王五", "42", "18000", "广州"],
        ["4", "赵六", "31", "25000", "深圳"],
        ["5", "陈七", "29", "12000", "杭州"],
    ]
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(content)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_csv_gbk() -> str:
    """创建一个 GBK 编码的 CSV 临时文件。"""
    content = [
        ["id", "name", "age", "income", "city"],
        ["1", "张三", "28", "15000", "北京"],
        ["2", "李四", "35", "22000", "上海"],
        ["3", "王五", "42", "18000", "广州"],
        ["4", "赵六", "31", "25000", "深圳"],
        ["5", "陈七", "29", "12000", "杭州"],
    ]
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="gbk", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(content)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_csv_numeric() -> str:
    """创建一个纯数值 CSV 临时文件（用于回归测试）。"""
    content = [
        ["x1", "x2", "x3", "y"],
        ["1.0", "5.0", "10.0", "3.5"],
        ["2.0", "4.0", "20.0", "5.2"],
        ["3.0", "3.0", "30.0", "7.1"],
        ["4.0", "2.0", "40.0", "8.8"],
        ["5.0", "1.0", "50.0", "10.5"],
        ["6.0", "0.0", "60.0", "12.2"],
    ]
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(content)
    yield path
    os.unlink(path)


@pytest.fixture
def sample_csv_missing() -> str:
    """创建一个含缺失值的 CSV 临时文件。"""
    content = [
        ["id", "value", "category"],
        ["1", "10.0", "A"],
        ["2", "", "B"],
        ["3", "30.0", ""],
        ["4", "40.0", "A"],
        ["5", "", "C"],
    ]
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(content)
    yield path
    os.unlink(path)
