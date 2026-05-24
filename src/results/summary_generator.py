# encoding: utf-8
"""Automatic generation of human-readable result summary text.

Produces Chinese-language model interpretation text for regression results,
coefficient interpretations, and assumption check summaries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.results.table import CoefficientRow, ModelResult


def _sig_label(pvalue: float) -> str:
    """Return a Chinese significance label for a p-value."""
    if pvalue < 0.001:
        return "<0.001"
    if pvalue < 0.01:
        return f"={pvalue:.3f}"
    if pvalue < 0.05:
        return f"={pvalue:.3f}"
    if pvalue < 0.1:
        return f"={pvalue:.3f}"
    return f"={pvalue:.3f}"


def _sig_star_text(pvalue: float) -> str:
    """Return significance verdict in Chinese."""
    if pvalue < 0.001:
        return "p<0.001"
    if pvalue < 0.01:
        return "p<0.01"
    if pvalue < 0.05:
        return "p<0.05"
    if pvalue < 0.1:
        return "p<0.1"
    return "p>=0.1（不显著）"


def generate_summary_text(result: ModelResult) -> str:
    """Generate a Chinese-language model summary text.

    Produces a paragraph describing overall model fit, significance,
    and key statistics. Suitable for inclusion in reports or automated
    result annotation.

    Args:
        result: A ModelResult object.

    Returns:
        A human-readable string summarizing the model.
    """
    dep_var = result.dep_var or "因变量"
    specification = result.specification or "未指定"

    # Model significance
    f_text = ""
    if result.f_statistic is not None:
        f_val, f_p = result.f_statistic
        if f_p < 0.001:
            f_sign_text = "<0.001"
        elif f_p < 0.05:
            f_sign_text = f"={f_p:.4f}"
        elif f_p < 0.1:
            f_sign_text = f"={f_p:.4f}（边缘显著）"
        else:
            f_sign_text = f"={f_p:.4f}（不显著）"

        df1 = result.n_params - 1
        df2 = result.df_resid
        f_text = (
            f"模型整体显著(F({df1},{df2})={f_val:.4f}, p{f_sign_text})。"
        )

    # R-squared
    r2_text = ""
    if result.r_squared is not None:
        r2_text = f"R²={result.r_squared:.4f}"
        if result.adj_r_squared is not None:
            r2_text += f"，调整R²={result.adj_r_squared:.4f}"

    # AIC / BIC
    ic_text = f"AIC={result.aic:.2f}，BIC={result.bic:.2f}"

    # Construct the summary paragraph
    parts = [
        f"本研究报告了{result.method}回归结果，因变量为「{dep_var}」。",
        f"模型规格: {specification}。",
        f_text,
        f"{r2_text}，RMSE={result.rmse:.4f}。",
        ic_text,
    ]
    if result.n_obs:
        parts.append(f"有效样本量N={result.n_obs}。")

    return " ".join(parts)


def generate_coefficient_interpretation(
    row: CoefficientRow,
    dep_var: str,
) -> str:
    """Generate a Chinese-language interpretation for a single coefficient.

    Args:
        row: A CoefficientRow object.
        dep_var: Name of the dependent variable.

    Returns:
        A human-readable string interpreting the coefficient.
    """
    var_name = row.name
    direction = "增加" if row.coef >= 0 else "减少"
    coef_abs = abs(row.coef)
    sig_text = _sig_star_text(row.pvalue)

    text = (
        f"在其他变量保持不变的情况下，{var_name}每增加一个单位，"
        f"{dep_var}平均{direction}{coef_abs:.4f}个单位"
        f"({sig_text})。"
    )
    return text


def generate_assumption_check_text(
    result: ModelResult,
    vif_df: Any = None,
    residual_tests: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate a Chinese-language assumption check summary.

    Covers multicollinearity (VIF), normality of residuals (Shapiro-Wilk),
    and autocorrelation (Durbin-Watson).

    Args:
        result: A ModelResult object.
        vif_df: A DataFrame produced by ``diagnostics.vif()``, or None.
        residual_tests: A dict from ``diagnostics.residual_tests()``, or None.

    Returns:
        A human-readable string describing the assumption check results.
    """
    lines: list[str] = ["=== 假设检验检查 ==="]

    # --- Multicollinearity ---
    if vif_df is not None and not vif_df.empty:
        high_vif: list[str] = []
        moderate_vif: list[str] = []
        for _, row in vif_df.iterrows():
            var_name = str(row.get("variable", ""))
            if var_name.lower() in ("const", "intercept"):
                continue
            vif_val = row.get("vif", 0)
            diagnosis = str(row.get("diagnosis", ""))
            if diagnosis == "High":
                high_vif.append(f"{var_name}(VIF={vif_val:.2f})")
            elif diagnosis == "Moderate":
                moderate_vif.append(f"{var_name}(VIF={vif_val:.2f})")

        if high_vif:
            lines.append(
                f"多重共线性检查: 发现严重多重共线性变量: {', '.join(high_vif)}。"
                f"建议考虑剔除或合并相关变量。"
            )
        elif moderate_vif:
            lines.append(
                f"多重共线性检查: 发现中等程度多重共线性: {', '.join(moderate_vif)}。"
                f"可考虑进一步诊断。"
            )
        else:
            lines.append(
                "多重共线性检查: 未发现严重的多重共线性(VIF均<5)。"
            )
    else:
        lines.append("多重共线性检查: 未提供VIF数据。")

    # --- Residual diagnostics ---
    if residual_tests is not None:
        # Normality
        shapiro_normal = residual_tests.get("shapiro_normal", "")
        if shapiro_normal == "Yes":
            lines.append(
                "残差正态性检查: Shapiro-Wilk检验表明残差近似正态分布(p>0.05)。"
            )
        elif shapiro_normal == "No":
            shapiro_p = residual_tests.get("shapiro_pvalue", None)
            p_str = f"p={shapiro_p:.4f}" if shapiro_p is not None else ""
            lines.append(
                f"残差正态性检查: Shapiro-Wilk检验表明残差不服从正态分布({p_str})。"
                f"模型推断在中小样本下可能不稳健。"
            )
        else:
            lines.append(f"残差正态性检查: {shapiro_normal}。")

        # Autocorrelation
        dw_auto = residual_tests.get("dw_autocorrelation", "")
        dw_stat = residual_tests.get("dw_stat", float("nan"))
        if dw_auto == "None":
            lines.append(
                f"残差自相关检查: Durbin-Watson统计量={dw_stat:.4f}，"
                f"无明显自相关。"
            )
        elif dw_auto and dw_auto != "Insufficient data":
            lines.append(
                f"残差自相关检查: Durbin-Watson统计量={dw_stat:.4f}，"
                f"存在{dw_auto}自相关。模型标准误可能被低估。"
            )
        else:
            lines.append("残差自相关检查: 数据不足以判断。")
    else:
        lines.append("残差诊断: 未提供残差检验数据。")

    return "\n".join(lines)
