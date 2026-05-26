"""
缺失值处理模块

提供 MissingValueHandler 类，用于分析 DataFram 中的缺失值
并应用不同的填充/删除策略。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class MissingValueHandler:
    """缺失值处理器。

    用法:
        handler = MissingValueHandler()
        stats = handler.analyze(df)
        cleaned = handler.handle(df, strategy='mean')
    """

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """分析 DataFrame 的缺失值情况。

        Args:
            data: 输入 DataFrame。

        Returns:
            包含缺失值统计信息的字典:
            {
                'total_rows': int,
                'total_columns': int,
                'total_missing': int,
                'columns': {
                    col_name: {
                        'count': int,
                        'percentage': float,
                        'dtype': str,
                        'warn': bool,   # >5%
                        'critical': bool,  # >20%
                    }
                }
            }
        """
        nrows = len(data)
        ncols = len(data.columns)
        total_missing = int(data.isna().sum().sum())

        columns: dict[str, dict[str, Any]] = {}
        for col in data.columns:
            col_str = str(col)
            missing_count = int(data[col].isna().sum())
            pct = missing_count / max(nrows, 1) * 100

            columns[col_str] = {
                "count": missing_count,
                "percentage": round(pct, 2),
                "dtype": str(data[col].dtype),
                "warn": pct > 5.0,
                "critical": pct > 20.0,
            }

        return {
            "total_rows": nrows,
            "total_columns": ncols,
            "total_missing": total_missing,
            "columns": columns,
        }

    def handle(
        self,
        data: pd.DataFrame,
        strategy: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """应用缺失值处理策略。

        Args:
            data: 输入 DataFrame。
            strategy: 处理策略，可选 'drop'、'mean'、'median'。
            columns: 要处理的列名列表。为 None 时处理所有列。

        Returns:
            处理后的 DataFrame。
        """
        if strategy not in ("drop", "mean", "median"):
            raise ValueError(f"不支持的缺失值处理策略: {strategy}。可选: 'drop', 'mean', 'median'")

        df = data.copy()

        if strategy == "drop":
            target_cols = columns if columns else df.columns.tolist()
            df = df.dropna(subset=target_cols)
            return df

        target_cols = columns if columns else df.columns.tolist()

        for col in target_cols:
            if col not in df.columns:
                continue
            if df[col].isna().sum() == 0:
                continue

            if strategy in ("mean", "median"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    if strategy == "mean":
                        fill_val = df[col].mean()
                    else:
                        fill_val = df[col].median()
                else:
                    # 非数值列用众数（mode）
                    mode_vals = df[col].mode()
                    fill_val = mode_vals.iloc[0] if len(mode_vals) > 0 else None

                if fill_val is not None and not (isinstance(fill_val, float) and np.isnan(fill_val)):
                    df[col] = df[col].fillna(fill_val)

        return df
