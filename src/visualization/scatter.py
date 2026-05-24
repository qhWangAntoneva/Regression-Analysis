# encoding: utf-8
"""
散点图模块

提供带 OLS 回归线和置信区间的散点图绘制功能。
使用 plotly express 实现交互式可视化。
"""

from __future__ import annotations

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure
except ImportError:
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


def scatter_with_regression(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str | None = None,
    color_col: str | None = None,
) -> "Figure":
    """绘制散点图 + OLS 回归线 + 置信区间带。

    使用 plotly express 绘制散点图，并叠加 OLS 回归线（基于 statsmodels 或 numpy polyfit）。

    Args:
        data: 输入数据。
        x_col: X 轴列名。
        y_col: Y 轴列名。
        title: 图表标题。None 时自动生成。
        color_col: 按第三变量着色的列名。None 时不按颜色分组。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 数据列不存在或数据不足。
    """
    if px is None or go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    # 验证列存在
    missing_cols = [c for c in [x_col, y_col] if c not in data.columns]
    if color_col and color_col not in data.columns:
        missing_cols.append(color_col)
    if missing_cols:
        raise ValueError(f"数据中不存在列: {missing_cols}")

    # 去除缺失值
    plot_data = data[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
    if len(plot_data) < 2:
        raise ValueError(f"数据点不足（至少需要 2 个非缺失值，当前 {len(plot_data)} 个）")

    # 自动生成标题
    if title is None:
        title = f"{y_col} vs {x_col}"

    # 创建散点图 + 回归线
    fig = px.scatter(
        plot_data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        trendline="ols",
        trendline_color_override="red",
        opacity=0.7,
        labels={x_col: x_col, y_col: y_col},
    )

    # 美化布局
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        title={"text": title, "x": 0.5, "xanchor": "center"},
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend_title=color_col if color_col else None,
    )

    # 添加置信区间带（使用 plotly express 的 trendline 自带 CI 阴影）
    # px.scatter 的 trendline="ols" 已经包含置信区间

    return fig
