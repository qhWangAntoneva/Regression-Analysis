# encoding: utf-8
"""
残差图模块

提供残差诊断图：残差 vs 拟合值散点图、正态 Q-Q 图。
使用 plotly 实现交互式可视化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure
    from plotly.subplots import make_subplots
except ImportError:
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
except ImportError:
    lowess = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Type stub for ModelResult — avoid importing modeling module at top level
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.results.table import ModelResult


def residual_vs_fitted_plot(
    result: Any,
    data: pd.DataFrame,
) -> "Figure":
    """残差 vs 拟合值散点图。

    Args:
        result: ModelResult 对象（或具有 .fitted_values, .residuals 属性的对象）。
        data: 原始数据（未使用，但保持接口一致性）。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 结果数据不足。
    """
    if px is None or go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    # 从 result 对象提取残差和拟合值
    residuals = _get_residuals(result)
    fitted_values = _get_fitted_values(result)

    if residuals is None or fitted_values is None:
        raise ValueError("ModelResult 对象缺少残差或拟合值")

    residuals = np.asarray(residuals).flatten()
    fitted_values = np.asarray(fitted_values).flatten()

    if len(residuals) < 2:
        raise ValueError(f"数据点不足（至少需要 2 个，当前 {len(residuals)} 个）")

    # 创建图表
    fig = go.Figure()

    # 添加散点
    fig.add_trace(
        go.Scatter(
            x=fitted_values,
            y=residuals,
            mode="markers",
            marker=dict(color="steelblue", size=6, opacity=0.6),
            name="残差",
            hovertemplate="拟合值: %{x:.4f}<br>残差: %{y:.4f}<extra></extra>",
        )
    )

    # 添加 y=0 水平参考线
    fig.add_hline(
        y=0,
        line=dict(color="red", width=1.5, dash="dash"),
        name="y=0",
    )

    # 添加 loess 平滑曲线
    if lowess is not None and len(residuals) >= 5:
        try:
            # lowess 返回排序后的结果
            sorted_idx = np.argsort(fitted_values)
            xs_sorted = fitted_values[sorted_idx]
            ys_sorted = residuals[sorted_idx]
            loess_result = lowess(ys_sorted, xs_sorted, frac=0.5, it=3)
            fig.add_trace(
                go.Scatter(
                    x=loess_result[:, 0],
                    y=loess_result[:, 1],
                    mode="lines",
                    line=dict(color="darkorange", width=2),
                    name="LOESS 平滑",
                )
            )
        except Exception:
            pass  # loess 失败时不显示平滑曲线

    # 布局
    fig.update_layout(
        title={"text": "残差 vs 拟合值", "x": 0.5, "xanchor": "center"},
        xaxis_title="拟合值 (Fitted Values)",
        yaxis_title="残差 (Residuals)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def qq_plot(residuals: np.ndarray) -> "Figure":
    """正态 Q-Q 图。

    Args:
        residuals: 残差数组。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 数据不足。
    """
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    residuals = np.asarray(residuals).flatten()
    if len(residuals) < 4:
        raise ValueError(f"数据点不足（至少需要 4 个，当前 {len(residuals)} 个）")

    # 计算理论分位数
    n = len(residuals)
    theoretical_quantiles = np.sort(_norm_ppf(np.arange(1, n + 1) / (n + 1)))
    sample_quantiles = np.sort(residuals)

    # 创建图表
    fig = go.Figure()

    # 散点
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode="markers",
            marker=dict(color="steelblue", size=6, opacity=0.6),
            name="样本分位数",
            hovertemplate="理论分位数: %{x:.4f}<br>样本分位数: %{y:.4f}<extra></extra>",
        )
    )

    # 对角线参考线
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", width=1.5, dash="dash"),
            name="理论正态线",
        )
    )

    # 布局
    fig.update_layout(
        title={"text": "正态 Q-Q 图", "x": 0.5, "xanchor": "center"},
        xaxis_title="理论分位数 (Theoretical Quantiles)",
        yaxis_title="样本分位数 (Sample Quantiles)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_residuals(result: Any) -> np.ndarray | None:
    """从 ModelResult 或类似对象提取残差。"""
    if hasattr(result, "residuals"):
        return np.asarray(result.residuals)
    if isinstance(result, dict):
        return result.get("residuals")
    return None


def _get_fitted_values(result: Any) -> np.ndarray | None:
    """从 ModelResult 或类似对象提取拟合值。"""
    if hasattr(result, "fitted_values"):
        return np.asarray(result.fitted_values)
    if isinstance(result, dict):
        return result.get("fitted_values")
    return None


def _norm_ppf(q: np.ndarray) -> np.ndarray:
    """正态分布的分位函数（百分位函数）。

    使用 scipy（如果可用）或近似公式。

    Args:
        q: 概率值数组 (0 < q < 1)。

    Returns:
        对应的分位数。
    """
    try:
        from scipy.stats import norm

        return norm.ppf(q)
    except ImportError:
        pass

    # 无 scipy 时的近似（Abramowitz and Stegun 近似）
    return _approx_norm_ppf(q)


def _approx_norm_ppf(q: np.ndarray) -> np.ndarray:
    """正态分布分位函数的近似计算。

    使用 Abramowitz and Stegun 公式 26.2.23 的近似。

    Args:
        q: 概率值数组。

    Returns:
        近似分位数。
    """
    p = np.asarray(q, dtype=float)
    # 限制范围避免极端值
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # Abramowitz and Stegun 近似
    t = np.sqrt(-2 * np.log(p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    result = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)

    return result
