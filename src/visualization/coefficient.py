"""
系数图模块

提供 dot-whisker 系数图：多模型系数对比图、单模型系数图。
使用 plotly 实现交互式可视化。
"""

from __future__ import annotations

from typing import Any

try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure
except ImportError:
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Type stubs
# ---------------------------------------------------------------------------
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def coefficient_plot(results: list[Any], model_labels: list[str] | None = None) -> Figure:
    """多模型 dot-whisker 系数对比图。

    每个模型的系数以点估计 + 95% 置信区间线段展示。
    不同模型以不同颜色区分。

    Args:
        results: ModelResult 对象列表。
        model_labels: 模型标签列表。为 None 时自动生成 "模型 1", "模型 2", ...。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 输入为空或缺少系数数据。
    """
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    if not results:
        raise ValueError("至少需要一个模型结果")

    n_models = len(results)

    if model_labels is None:
        model_labels = [f"模型 {i + 1}" for i in range(n_models)]

    if len(model_labels) != n_models:
        raise ValueError("模型标签数量必须与模型结果数量一致")

    # 收集所有变量名
    all_vars: list[str] = []
    for result in results:
        coefs = _get_coefficients(result)
        if not coefs:
            continue
        for c in coefs:
            name = _get_coef_name(c)
            if name and name != "Intercept" and name not in all_vars:
                all_vars.append(name)

    if not all_vars:
        raise ValueError("所有模型均无可用系数数据")

    # 颜色调色板
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # 创建图表
    fig = go.Figure()

    # 为每个模型添加系数轨迹
    for i, (result, label) in enumerate(zip(results, model_labels)):
        coefs = _get_coefficients(result)
        if not coefs:
            continue

        color = colors[i % len(colors)]

        # 提取截距项
        intercept_est = None
        intercept_ci_low = None
        intercept_ci_high = None

        # 构建变量名到系数的映射
        coef_map: dict[str, dict[str, Any]] = {}
        for c in coefs:
            name = _get_coef_name(c)
            if name == "Intercept":
                intercept_est = _get_coef_value(c)  # noqa: F841
                intercept_ci_low = _get_ci_low(c)  # noqa: F841
                intercept_ci_high = _get_ci_high(c)  # noqa: F841
                continue
            if name:
                coef_map[name] = {
                    "est": _get_coef_value(c),
                    "ci_low": _get_ci_low(c),
                    "ci_high": _get_ci_high(c),
                }

        # 为当前模型的所有变量准备数据
        var_ests: list[float] = []
        var_ci_low: list[float] = []
        var_ci_high: list[float] = []

        for var in all_vars:
            if var in coef_map:
                var_ests.append(coef_map[var]["est"])
                var_ci_low.append(coef_map[var]["ci_low"])
                var_ci_high.append(coef_map[var]["ci_high"])
            else:
                # 模型中不存在的变量用 0
                var_ests.append(0.0)
                var_ci_low.append(0.0)
                var_ci_high.append(0.0)

        # y 轴位置：为每个模型分配偏移
        y_positions = [all_vars.index(var) * (n_models + 1) + i for var in all_vars]

        # 点估计
        fig.add_trace(
            go.Scatter(
                x=var_ests,
                y=y_positions,
                mode="markers",
                marker=dict(color=color, size=8, symbol="circle"),
                name=label,
                legendgroup=label,
                hovertemplate=(
                    f"模型: {label}<br>"
                    "变量: %{customdata}<br>"
                    "系数: %{x:.4f}<extra></extra>"
                ),
                customdata=all_vars,
            )
        )

        # 置信区间线段
        for j, var in enumerate(all_vars):
            if var in coef_map:
                fig.add_trace(
                    go.Scatter(
                        x=[var_ci_low[j], var_ci_high[j]],
                        y=[y_positions[j], y_positions[j]],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        legendgroup=label,
                        hoverinfo="skip",
                    )
                )

    # 零点垂直线
    fig.add_vline(
        x=0,
        line=dict(color="gray", width=1, dash="dash"),
    )

    # y 轴刻度标签
    y_tick_pos = [
        (len(all_vars) - 1 - i) * (n_models + 1) + (n_models - 1) / 2
        for i in range(len(all_vars))
    ]
    y_tick_labels = list(reversed(all_vars))

    # 布局
    fig.update_layout(
        title={"text": "系数对比图 (Dot-Whisker)", "x": 0.5, "xanchor": "center"},
        xaxis_title="系数估计值 (Coefficient Estimate)",
        yaxis=dict(
            tickvals=y_tick_pos,
            ticktext=y_tick_labels,
            title="",
        ),
        template="plotly_white",
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        height=max(400, len(all_vars) * (n_models + 1) * 30),
    )

    return fig


def coefficient_plot_single(result: Any) -> Figure:
    """单模型系数图。

    系数按绝对值从大到小排序（截距项置于顶部）。
    标注显著性星标在点旁。

    Args:
        result: ModelResult 对象。

    Returns:
        plotly Figure 对象。

    Raises:
        ImportError: plotly 未安装。
        ValueError: 缺少系数数据。
    """
    if go is None:
        msg = "plotly 未安装。请运行: pip install plotly"
        raise ImportError(msg)

    coefs = _get_coefficients(result)
    if not coefs:
        raise ValueError("模型结果中无可用系数数据")

    # 分离截距项和非截距项
    intercept: dict[str, Any] | None = None
    non_intercept: list[dict[str, Any]] = []

    for c in coefs:
        name = _get_coef_name(c)
        entry = {
            "name": name,
            "est": _get_coef_value(c),
            "ci_low": _get_ci_low(c),
            "ci_high": _get_ci_high(c),
            "pvalue": _get_pvalue(c) if _get_pvalue(c) is not None else 1.0,
            "stars": _get_stars(c) or "",
        }
        if name == "Intercept":
            intercept = entry
        else:
            non_intercept.append(entry)

    # 按绝对值从大到小排序
    non_intercept.sort(key=lambda x: abs(x["est"]), reverse=True)

    # 合并：截距在前，其他按绝对值排序
    sorted_coefs: list[dict[str, Any]] = []
    if intercept:
        sorted_coefs.append(intercept)
    sorted_coefs.extend(non_intercept)

    n_coefs = len(sorted_coefs)
    var_names = [c["name"] for c in sorted_coefs]
    estimates = [c["est"] for c in sorted_coefs]
    ci_lows = [c["ci_low"] for c in sorted_coefs]
    ci_highs = [c["ci_high"] for c in sorted_coefs]
    star_labels = [c["stars"] for c in sorted_coefs]

    # y 轴位置（从上到下）
    y_positions = list(range(n_coefs - 1, -1, -1))

    # 创建图表
    fig = go.Figure()

    # 置信区间线段
    fig.add_trace(
        go.Scatter(
            x=ci_lows + ci_highs,
            y=list(y_positions) + list(y_positions),
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 实际数据点需要逐对线段绘制
    for j in range(n_coefs):
        fig.add_trace(
            go.Scatter(
                x=[ci_lows[j], ci_highs[j]],
                y=[y_positions[j], y_positions[j]],
                mode="lines",
                line=dict(color="#1f77b4", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # 点估计（带显著性星标注释）
    fig.add_trace(
        go.Scatter(
            x=estimates,
            y=y_positions,
            mode="markers+text",
            marker=dict(color="#1f77b4", size=10, symbol="diamond"),
            text=star_labels,
            textposition="middle right",
            textfont=dict(size=12, color="red"),
            name="系数估计",
            hovertemplate=(
                "变量: %{customdata}<br>"
                "系数: %{x:.4f}<br>"
                "95%% CI: [%{customdata2:.4f}, %{customdata3:.4f}]<extra></extra>"
            ),
            customdata=list(
                zip(
                    var_names,
                    ci_lows,
                    ci_highs,
                )
            ),
        )
    )

    # 零点垂直线
    fig.add_vline(
        x=0,
        line=dict(color="gray", width=1, dash="dash"),
    )

    # 布局
    fig.update_layout(
        title={"text": "系数估计图", "x": 0.5, "xanchor": "center"},
        xaxis_title="系数估计值 (Coefficient Estimate)",
        yaxis=dict(
            tickvals=y_positions,
            ticktext=var_names,
            title="",
        ),
        template="plotly_white",
        hovermode="closest",
        height=max(300, n_coefs * 40),
    )

    # 添加星标图例
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1,
        y=-0.05,
        text="*** p<0.01, ** p<0.05, * p<0.1",
        showarrow=False,
        font=dict(size=10, color="gray"),
        xanchor="right",
    )

    return fig


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _get_coefficients(result: Any) -> list[Any]:
    """从 ModelResult 提取系数列表。"""
    if hasattr(result, "coefficients"):
        return list(result.coefficients)
    return []


def _get_coef_name(c: Any) -> str:
    """获取系数名称。"""
    if hasattr(c, "name"):
        return str(c.name)
    return ""


def _get_coef_value(c: Any) -> float:
    """获取系数估计值。"""
    if hasattr(c, "coef"):
        return float(c.coef)
    if hasattr(c, "coefficient"):
        return float(c.coefficient)
    if hasattr(c, "estimate"):
        return float(c.estimate)
    return 0.0


def _get_ci_low(c: Any) -> float:
    """获取置信区间下限。"""
    if hasattr(c, "ci_lower"):
        return float(c.ci_lower)
    return 0.0


def _get_ci_high(c: Any) -> float:
    """获取置信区间上限。"""
    if hasattr(c, "ci_upper"):
        return float(c.ci_upper)
    return 0.0


def _get_pvalue(c: Any) -> float | None:
    """获取 p 值。"""
    if hasattr(c, "pvalue"):
        return float(c.pvalue)
    if hasattr(c, "p_value"):
        return float(c.p_value)
    return None


def _get_stars(c: Any) -> str:
    """获取显著性星标。"""
    if hasattr(c, "significance"):
        return str(c.significance) if c.significance else ""
    pval = _get_pvalue(c)
    if pval is not None:
        if pval < 0.01:
            return "***"
        if pval < 0.05:
            return "**"
        if pval < 0.1:
            return "*"
    return ""
