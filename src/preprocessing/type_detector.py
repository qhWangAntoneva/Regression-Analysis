# encoding: utf-8
"""
变量类型自动检测模块

独立于 modeling 模块，只处理数据视图的变量类型推断。
提供 VariableInfo dataclass 和 VariableTypeDetector 类。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class VariableInfo:
    """单个变量的信息。"""

    name: str
    dtype: str  # pandas dtype string
    inferred_type: str  # 'continuous' / 'categorical' / 'binary' / 'ordinal' / 'id' / 'text'
    n_unique: int
    n_missing: int
    missing_rate: float
    mean: float | None = None
    std: float | None = None
    min_val: float | None = None
    max_val: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type,
            "n_unique": self.n_unique,
            "n_missing": self.n_missing,
            "missing_rate": self.missing_rate,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
        }


class VariableTypeDetector:
    """变量类型检测器。

    对所有列运行类型检测，返回 VariableInfo 列表。
    """

    def detect(self, df: pd.DataFrame) -> list[VariableInfo]:
        """检测 DataFrame 中所有列的类型。"""
        variables: list[VariableInfo] = []
        nrows = len(df)

        for col in df.columns:
            series = df[col]
            col_name = str(col)

            inferred_type = self._detect_column(series, col_name)
            n_unique = int(series.nunique())
            n_missing = int(series.isna().sum())
            missing_rate = round(n_missing / max(nrows, 1), 4)

            # 计算统计量（仅对数值类型）
            mean_val: float | None = None
            std_val: float | None = None
            min_val: float | None = None
            max_val: float | None = None

            numeric_series = self._try_to_numeric(series)
            if numeric_series is not None and len(numeric_series) > 0:
                mean_val = float(numeric_series.mean()) if not pd.isna(numeric_series.mean()) else None
                std_val = float(numeric_series.std()) if not pd.isna(numeric_series.std()) else None
                min_val = float(numeric_series.min()) if not pd.isna(numeric_series.min()) else None
                max_val = float(numeric_series.max()) if not pd.isna(numeric_series.max()) else None

            info = VariableInfo(
                name=col_name,
                dtype=str(series.dtype),
                inferred_type=inferred_type,
                n_unique=n_unique,
                n_missing=n_missing,
                missing_rate=missing_rate,
                mean=mean_val,
                std=std_val,
                min_val=min_val,
                max_val=max_val,
            )
            variables.append(info)

        return variables

    def _detect_column(self, col: pd.Series, col_name: str) -> str:
        """单列检测逻辑。

        Args:
            col: 列数据。
            col_name: 列名。

        Returns:
            推断类型字符串。
        """
        nrows = len(col)
        n_unique = col.nunique()

        # 步骤 1: 全 NaN 列
        if col.isna().all():
            return "categorical"

        # 步骤 2: ID 列（列名匹配 id/code/num 模式 + 唯一值 == 总行数）
        col_lower = col_name.lower().strip()
        id_patterns = ("id", "code", "num", "no.", "number", "序号", "编号", "代码")
        is_id_name = any(col_lower.startswith(p) or col_lower.endswith(p) for p in id_patterns)
        if is_id_name and n_unique == nrows:
            return "id"

        # 步骤 3: object 类型尝试数值转换
        if pd.api.types.is_object_dtype(col) or pd.api.types.is_string_dtype(col):
            numeric_col = self._try_to_numeric(col)
            if numeric_col is not None:
                # 如果数值转换成功，用转换后的数据继续判断
                return self._classify_numeric_column(numeric_col, col_name, nrows)
            else:
                # 纯文本列
                if n_unique == nrows or n_unique / max(nrows, 1) > 0.9:
                    return "id"
                return "text"

        # 步骤 4: 布尔 → binary
        if pd.api.types.is_bool_dtype(col):
            return "binary"

        # 步骤 5: 数值列分类
        return self._classify_numeric_column(col, col_name, nrows)

    def _classify_numeric_column(
        self, col: pd.Series, col_name: str, nrows: int
    ) -> str:
        """对数值列进行细分类。"""
        n_unique = col.nunique()

        # 唯一值 ≤ 2 → binary
        if n_unique <= 2:
            return "binary"

        # 唯一值 ≤ 行数 5% 且 ≤ 20 → categorical
        if n_unique <= max(nrows * 0.05, 1) and n_unique <= 20:
            return "categorical"

        # 其余 → continuous
        return "continuous"

    @staticmethod
    def _try_to_numeric(series: pd.Series) -> pd.Series | None:
        """尝试将列转换为数值类型。

        Returns:
            转换后的数值 Series，或 None（无法转换）。
        """
        # 先去除缺失值再尝试转换
        non_null = series.dropna()
        if len(non_null) == 0:
            return None

        try:
            converted = pd.to_numeric(non_null, errors="raise")
            return converted
        except (ValueError, TypeError):
            return None
