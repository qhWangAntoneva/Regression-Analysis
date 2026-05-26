"""
异常值检测模块

提供 OutlierDetector 类，支持 IQR 和 Z-Score 两种异常值检测方法。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class OutlierDetector:
    """异常值检测器。

    用法:
        detector = OutlierDetector()
        mask = detector.detect_iqr(df, 'column_name')
        df_with_flags = detector.flag_outliers(df, ['col1', 'col2'], method='iqr')
    """

    def detect_iqr(
        self,
        data: pd.DataFrame,
        column: str,
        multiplier: float = 1.5,
    ) -> pd.Series:
        """使用 IQR 方法检测异常值。

        Args:
            data: 输入 DataFrame。
            column: 要检测的列名。
            multiplier: IQR 倍数，默认 1.5（常规），3.0（极端）。

        Returns:
            布尔 Series，True 表示异常值。
        """
        if column not in data.columns:
            raise ValueError(f"列 '{column}' 不存在于 DataFrame 中。")

        series = data[column]
        if not pd.api.types.is_numeric_dtype(series):
            raise ValueError(f"列 '{column}' 不是数值类型（{series.dtype}），无法计算 IQR。")

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        return (series < lower_bound) | (series > upper_bound)

    def detect_zscore(
        self,
        data: pd.DataFrame,
        column: str,
        threshold: float = 3.0,
    ) -> pd.Series:
        """使用 Z-Score 方法检测异常值。

        Args:
            data: 输入 DataFrame。
            column: 要检测的列名。
            threshold: Z-Score 阈值，默认 3.0。

        Returns:
            布尔 Series，True 表示异常值。
        """
        if column not in data.columns:
            raise ValueError(f"列 '{column}' 不存在于 DataFrame 中。")

        series = data[column].dropna()
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"列 '{column}' 不是数值类型（{data[column].dtype}），无法计算 Z-Score。")

        mean = series.mean()
        std = series.std()

        if std == 0 or pd.isna(std):
            # 标准差为0，所有值相同，没有异常值
            return pd.Series(False, index=data.index)

        z_scores = (data[column] - mean) / std
        return z_scores.abs() > threshold

    def flag_outliers(
        self,
        data: pd.DataFrame,
        columns: list[str],
        method: str = "iqr",
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """为指定列添加异常值标记列。

        Args:
            data: 输入 DataFrame。
            columns: 要检测的列名列表。
            method: 检测方法，'iqr' 或 'zscore'，默认 'iqr'。
            **kwargs: 传递给 detect_iqr/detect_zscore 的额外参数。

        Returns:
            (带标记列的 DataFrame, 摘要字典) 的元组。
            摘要格式: {column: {'n_outliers': int, 'percentage': float}}
        """
        if method not in ("iqr", "zscore"):
            raise ValueError(
                f"不支持的检测方法: {method}。可选: 'iqr', 'zscore'"
            )

        df = data.copy()
        summary: dict[str, dict[str, Any]] = {}

        for col in columns:
            if col not in df.columns:
                summary[col] = {"n_outliers": 0, "percentage": 0.0, "error": "列不存在"}
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                summary[col] = {"n_outliers": 0, "percentage": 0.0, "error": "非数值列"}
                continue

            try:
                if method == "iqr":
                    outlier_mask = self.detect_iqr(df, col, **kwargs)
                else:
                    outlier_mask = self.detect_zscore(df, col, **kwargs)
            except Exception:
                summary[col] = {"n_outliers": 0, "percentage": 0.0, "error": "检测失败"}
                continue

            n_outliers = int(outlier_mask.sum())
            total_valid = int(len(df))
            pct = round(n_outliers / max(total_valid, 1) * 100, 2)

            df[f"{col}_outlier"] = outlier_mask
            summary[col] = {"n_outliers": n_outliers, "percentage": pct}

        return df, summary
