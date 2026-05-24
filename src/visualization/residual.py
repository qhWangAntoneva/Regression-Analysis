# encoding: utf-8
"""
残差图模块

提供残差诊断图：残差 vs 拟合值散点图、正态 Q-Q 图、尺度-位置图、
Cook's distance 图，以及诊断总览面板。
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


# ---------------------------------------------------------------------------
# Phase 2: Additional diagnostic plots
# ---------------------------------------------------------------------------


def scale_location_plot(
    result: Any,
    data: pd.DataFrame,
) -> Figure:
    """尺度-位置图（Scale-Location Plot）。

    x 轴为拟合值，y 轴为标准化残差平方根的绝对值。
    用于检验等方差性假设。若散点随机分布且平滑曲线接近水平，
    则等方差性假设成立。

    Args:
        result: ModelResult 对象（或具有 .fitted_values, .residuals 属性的对象）。
        data: 原始数据（未使用，但保持接口一致性）。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 结果数据不足。
    """
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    residuals = _get_residuals(result)
    fitted_values = _get_fitted_values(result)

    if residuals is None or fitted_values is None:
        raise ValueError("ModelResult 对象缺少残差或拟合值")

    residuals = np.asarray(residuals).flatten()
    fitted_values = np.asarray(fitted_values).flatten()

    if len(residuals) < 5:
        raise ValueError(f"数据点不足（至少需要 5 个，当前 {len(residuals)} 个）")

    # 计算标准化残差
    std_residuals = residuals / np.std(residuals, ddof=1)
    # 计算 sqrt(|标准化残差|)
    sqrt_abs_std_resid = np.sqrt(np.abs(std_residuals))

    # 创建图表
    fig = go.Figure()

    # 散点
    fig.add_trace(
        go.Scatter(
            x=fitted_values,
            y=sqrt_abs_std_resid,
            mode="markers",
            marker=dict(color="steelblue", size=6, opacity=0.6),
            name="标准化残差",
            hovertemplate=(
                "拟合值: %{x:.4f}<br>"
                "sqrt(|标准化残差|): %{y:.4f}<extra></extra>"
            ),
        )
    )

    # 添加 loess 平滑曲线
    if lowess is not None and len(residuals) >= 5:
        try:
            sorted_idx = np.argsort(fitted_values)
            xs_sorted = fitted_values[sorted_idx]
            ys_sorted = sqrt_abs_std_resid[sorted_idx]
            loess_result = lowess(ys_sorted, xs_sorted, frac=0.5, it=3)
            fig.add_trace(
                go.Scatter(
                    x=loess_result[:, 0],
                    y=loess_result[:, 1],
                    mode="lines",
                    line=dict(color="red", width=2),
                    name="LOESS 平滑",
                )
            )
        except Exception:
            pass

    # 布局
    fig.update_layout(
        title={
            "text": "尺度-位置图 (Scale-Location)",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="拟合值 (Fitted Values)",
        yaxis_title="sqrt(|标准化残差|)",
        template="plotly_white",
        hovermode="closest",
    )

    return fig


def cooks_distance_plot(
    result: Any,
    data: pd.DataFrame,
) -> Figure:
    """Cook's distance 条形图。

    显示每个观测的 Cook's distance，标记异常点（超过 4/n 阈值）。

    Args:
        result: ModelResult 对象。
        data: 原始数据（用于获取观测数 n）。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 结果数据不足。
    """
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    residuals = _get_residuals(result)
    fitted_values = _get_fitted_values(result)

    if residuals is None or fitted_values is None:
        raise ValueError("ModelResult 对象缺少残差或拟合值")

    residuals = np.asarray(residuals).flatten()
    fitted_values = np.asarray(fitted_values).flatten()
    n = len(residuals)

    if n < 3:
        raise ValueError(f"数据点不足（至少需要 3 个，当前 {n} 个）")

    # 估算 Cook's distance: D_i = (residuals_i^2 / (p * MSE)) * (h_ii / (1 - h_ii)^2)
    # 当无法获得 hat values 时，使用简化近似
    p = 1  # 至少 1 个参数
    if hasattr(result, "coefficients"):
        coefs = result.coefficients
        p = len(coefs) if coefs else p
    elif hasattr(result, "n_params"):
        p = result.n_params

    mse = np.var(residuals, ddof=p)  # MSE with dof = p

    if mse <= 0:
        raise ValueError("MSE 为零或负数，无法计算 Cook's distance")

    # 简化 Cook's distance: D_i = (r_i^2 / (p * MSE)) * (1/n)
    # 更精确的计算需要 hat matrix
    cooks_d = (residuals**2) / (p * mse) * (1.0 / n)

    threshold = 4.0 / n

    # 创建图表
    fig = go.Figure()

    # 观测索引
    obs_idx = np.arange(1, n + 1)

    # 分离正常点和异常点
    is_influential = cooks_d > threshold
    colors = np.where(is_influential, "red", "steelblue")
    sizes = np.where(is_influential, 8, 4)

    fig.add_trace(
        go.Bar(
            x=obs_idx,
            y=cooks_d,
            marker=dict(color=colors, opacity=0.7),
            name="Cook's distance",
            hovertemplate="观测: %{x}<br>Cook's D: %{y:.4f}<extra></extra>",
        )
    )

    # 阈值线
    fig.add_hline(
        y=threshold,
        line=dict(color="red", width=1.5, dash="dash"),
        annotation_text=f"阈值 (4/n) = {threshold:.4f}",
        annotation_position="right",
    )

    # 布局
    fig.update_layout(
        title={"text": "Cook's Distance", "x": 0.5, "xanchor": "center"},
        xaxis_title="观测序号 (Observation Index)",
        yaxis_title="Cook's Distance",
        template="plotly_white",
        hovermode="closest",
        showlegend=False,
    )

    return fig


def diagnostic_dashboard(
    result: Any,
    data: pd.DataFrame,
) -> dict[str, Figure]:
    """诊断图总览面板。

    返回包含 4 个诊断图的字典，可用于 UI 中的 2x2 网格布局：
    - 残差 vs 拟合值图
    - 正态 Q-Q 图
    - 尺度-位置图
    - Cook's distance 图

    Args:
        result: ModelResult 对象。
        data: 原始数据。

    Returns:
        字典，键为图名，值为 plotly Figure。
        {'residual_fitted': Figure, 'qq': Figure, 'scale_location': Figure, 'cooks_distance': Figure}
    """
    residuals = _get_residuals(result)
    fitted_values = _get_fitted_values(result)

    figs: dict[str, Figure] = {}

    try:
        figs["residual_fitted"] = residual_vs_fitted_plot(result, data)
    except Exception:
        figs["residual_fitted"] = _empty_figure("残差 vs 拟合值图不可用")

    try:
        if residuals is not None:
            figs["qq"] = qq_plot(np.asarray(residuals).flatten())
        else:
            figs["qq"] = _empty_figure("Q-Q 图不可用（无残差数据）")
    except Exception:
        figs["qq"] = _empty_figure("Q-Q 图不可用")

    try:
        figs["scale_location"] = scale_location_plot(result, data)
    except Exception:
        figs["scale_location"] = _empty_figure("尺度-位置图不可用")

    try:
        figs["cooks_distance"] = cooks_distance_plot(result, data)
    except Exception:
        figs["cooks_distance"] = _empty_figure("Cook's distance 图不可用")

    return figs


def _empty_figure(message: str) -> Figure:
    """创建显示错误信息的空图表。"""
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig
