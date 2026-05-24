# encoding: utf-8
"""Unified result data structures for regression model output.

Provides dataclasses for storing coefficient-level and model-level results,
along with utilities for formatting and converting to pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _significance_stars(pvalue: float) -> str:
    """Return significance stars based on p-value thresholds.

    Args:
        pvalue: The p-value.

    Returns:
        ``'***'`` if p < 0.01, ``'**'`` if p < 0.05, ``'*'`` if p < 0.1,
        otherwise ``''``.
    """
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.1:
        return "*"
    return ""


def _pvalue_label(pvalue: float) -> str:
    """Return a human-readable p-value significance label.

    Args:
        pvalue: The p-value.

    Returns:
        ``'p<0.01'``, ``'p<0.05'``, ``'p<0.1'``, or ``'p>=0.1'``.
    """
    if pvalue < 0.01:
        return "p<0.01"
    if pvalue < 0.05:
        return "p<0.05"
    if pvalue < 0.1:
        return "p<0.1"
    return "p>=0.1"


@dataclass
class CoefficientRow:
    """A single coefficient row in a regression results table.

    Attributes:
        name: Variable name (e.g., ``'x1'``, ``'C(cat)[T.B]'``, ``'Intercept'``).
        coef: Estimated coefficient value.
        se: Standard error of the coefficient.
        t_stat: t-statistic for the coefficient.
        pvalue: p-value for the t-test.
        ci_lower: Lower bound of the confidence interval (default 95%).
        ci_upper: Upper bound of the confidence interval (default 95%).
        significance: Significance stars string (``'***'``, ``'**'``, ``'*'``, or ``''``).
    """

    name: str
    coef: float
    se: float
    t_stat: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significance: str = ""

    def __post_init__(self) -> None:
        """Auto-compute significance stars if not provided."""
        if not self.significance:
            self.significance = _significance_stars(self.pvalue)


@dataclass
class ModelResult:
    """Complete regression model results.

    Attributes:
        model_type: A label for the model type (e.g., ``'OLS'``).
        coefficients: List of CoefficientRow objects.
        n_obs: Number of observations used in the model.
        n_params: Number of parameters estimated (including intercept).
        df_resid: Residual degrees of freedom.
        r_squared: R-squared (coefficient of determination), or None if unavailable.
        adj_r_squared: Adjusted R-squared, or None if unavailable.
        f_statistic: Tuple of ``(F-statistic, p-value)`` or None.
        log_likelihood: Log-likelihood of the fitted model, or None.
        aic: Akaike Information Criterion.
        bic: Bayesian Information Criterion.
        rmse: Root mean squared error of the model.
        dep_var: Name of the dependent variable.
        specification: String representation of the model formula/spec.
        method: Estimation method (default ``'OLS'``).
    """

    model_type: str
    coefficients: List[CoefficientRow]
    n_obs: int
    n_params: int
    df_resid: int
    r_squared: Optional[float] = None
    adj_r_squared: Optional[float] = None
    f_statistic: Optional[Tuple[float, float]] = None
    log_likelihood: Optional[float] = None
    aic: float = 0.0
    bic: float = 0.0
    rmse: float = 0.0
    dep_var: str = ""
    specification: str = ""
    method: str = "OLS"

    # ------------------------------------------------------------------
    # Existing methods
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the coefficient table to a formatted DataFrame.

        Returns:
            A DataFrame with columns:
            [变量, 系数, 标准误, t值, p值, 95%CI低, 95%CI高, 显著性]
        """
        rows: List[Dict[str, object]] = []
        for coef_row in self.coefficients:
            rows.append(
                {
                    "变量": coef_row.name,
                    "系数": round(coef_row.coef, 6),
                    "标准误": round(coef_row.se, 6),
                    "t值": round(coef_row.t_stat, 4),
                    "p值": coef_row.pvalue,
                    "95%CI低": round(coef_row.ci_lower, 6),
                    "95%CI高": round(coef_row.ci_upper, 6),
                    "显著性": coef_row.significance,
                }
            )

        df = pd.DataFrame(rows)
        df = df.set_index("变量")
        return df

    def summary(self) -> str:
        """Return a human-readable model summary string (similar to R's summary.lm).

        Returns:
            A formatted string summarizing the model fit.
        """
        lines: List[str] = []
        lines.append(f"{'=' * 60}")
        lines.append(f"  {self.method} Regression Results")
        lines.append(f"{'=' * 60}")
        lines.append(f"  Dependent Variable:    {self.dep_var}")
        lines.append(f"  Specification:         {self.specification}")
        lines.append(f"  Method:                {self.method}")
        lines.append(f"  No. Observations:      {self.n_obs}")
        lines.append(f"  No. Parameters:        {self.n_params}")
        lines.append(f"  Residual DF:           {self.df_resid}")
        lines.append("")
        lines.append(f"  R-squared:             {self.r_squared:.6f}" if self.r_squared is not None else "  R-squared:             N/A")
        lines.append(f"  Adj. R-squared:        {self.adj_r_squared:.6f}" if self.adj_r_squared is not None else "  Adj. R-squared:        N/A")
        lines.append(f"  RMSE:                  {self.rmse:.6f}")
        if self.f_statistic is not None:
            lines.append(f"  F-statistic:           {self.f_statistic[0]:.4f}")
            lines.append(f"  Prob (F-statistic):    {self.f_statistic[1]:.6e}")
        lines.append(f"  Log-Likelihood:        {self.log_likelihood:.4f}" if self.log_likelihood is not None else "  Log-Likelihood:        N/A")
        lines.append(f"  AIC:                   {self.aic:.4f}")
        lines.append(f"  BIC:                   {self.bic:.4f}")
        lines.append("")
        lines.append(f"{'-' * 60}")
        lines.append(f"  {'Variable':<20} {'Coefficient':>12} {'Std.Err.':>10} {'t':>8} {'p>|t|':>8}")
        lines.append(f"{'-' * 60}")
        for c in self.coefficients:
            lines.append(
                f"  {c.name:<20} {c.coef:>12.6f} {c.se:>10.6f} {c.t_stat:>8.4f} {c.pvalue:>8.4f} {c.significance}"
            )
        lines.append(f"{'=' * 60}")
        lines.append(f"  Significance: *** p<0.01, ** p<0.05, * p<0.1")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Phase 2 enhancements
    # ------------------------------------------------------------------

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary of all model statistics for UI display.

        Returns:
            A flat dictionary containing all model-level statistics.
        """
        d: Dict[str, Any] = {
            "dep_var": self.dep_var,
            "n_obs": self.n_obs,
            "n_params": self.n_params,
            "df_resid": self.df_resid,
            "r_squared": self.r_squared,
            "adj_r_squared": self.adj_r_squared,
            "rmse": self.rmse,
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "method": self.method,
            "specification": self.specification,
        }
        if self.f_statistic is not None:
            d["f_statistic"] = round(self.f_statistic[0], 6)
            d["f_pvalue"] = self.f_statistic[1]
        else:
            d["f_statistic"] = None
            d["f_pvalue"] = None
        return d

    def anova_table(self) -> pd.DataFrame:
        """Build an ANOVA (analysis of variance) table.

        Decomposes the total sum of squares into regression (explained) and
        residual components.

        Returns:
            A DataFrame with columns:
            [来源, SS, df, MS, F, p-value]
            and rows: 回归(Explained), 残差(Residual), 总计(Total).
        """
        # Recover sums of squares from available statistics
        # RMSE = sqrt(SS_resid / df_resid)  =>  SS_resid = RMSE^2 * df_resid
        ss_resid = self.rmse ** 2 * self.df_resid

        # Total SS is not stored directly; recover from R² = 1 - SS_resid / SS_total
        if self.r_squared is not None and self.r_squared < 0:
            self.r_squared = abs(self.r_squared) if self.r_squared is not None else 0.0
        ss_total = float("nan")
        ms_explained = float("nan")
        ms_resid = float("nan")
        f_val = float("nan")
        f_p = float("nan")

        if self.r_squared is not None and self.r_squared != 1.0:
            ss_total = ss_resid / (1.0 - self.r_squared)
        elif self.r_squared == 1.0:
            ss_total = ss_resid  # degenerate, set equal
            ss_resid = 0.0

        ss_explained = ss_total - ss_resid if not np.isnan(ss_total) else float("nan")

        df_explained = self.n_params - 1 if self.n_params > 1 else 0
        df_total = self.n_obs - 1

        if df_explained > 0:
            ms_explained = ss_explained / df_explained
        if self.df_resid > 0:
            ms_resid = ss_resid / self.df_resid

        if ms_resid > 0 and ms_explained > 0:
            f_val = ms_explained / ms_resid

        if self.f_statistic is not None:
            f_val = self.f_statistic[0]
            f_p = self.f_statistic[1]

        rows: List[Dict[str, object]] = [
            {
                "来源": "回归(Explained)",
                "SS": round(ss_explained, 6) if not np.isnan(ss_explained) else float("nan"),
                "df": df_explained,
                "MS": round(ms_explained, 6) if not np.isnan(ms_explained) else float("nan"),
                "F": round(f_val, 6) if not np.isnan(f_val) else float("nan"),
                "p-value": round(f_p, 6) if not np.isnan(f_p) else float("nan"),
            },
            {
                "来源": "残差(Residual)",
                "SS": round(ss_resid, 6),
                "df": self.df_resid,
                "MS": round(ms_resid, 6) if not np.isnan(ms_resid) else float("nan"),
                "F": float("nan"),
                "p-value": float("nan"),
            },
            {
                "来源": "总计(Total)",
                "SS": round(ss_total, 6) if not np.isnan(ss_total) else float("nan"),
                "df": df_total,
                "MS": float("nan"),
                "F": float("nan"),
                "p-value": float("nan"),
            },
        ]

        return pd.DataFrame(rows)

    def to_latex_row(self) -> str:
        """Generate a single LaTeX table row for the model summary.

        Useful for exporting model results into academic papers.

        Returns:
            A LaTeX string with model statistics separated by ``&``,
            terminated by ``\\\\``.
        """
        r2_str = f"{self.r_squared:.4f}" if self.r_squared is not None else "N/A"
        adj_r2_str = f"{self.adj_r_squared:.4f}" if self.adj_r_squared is not None else "N/A"
        aic_str = f"{self.aic:.2f}"
        bic_str = f"{self.bic:.2f}"
        n_str = str(self.n_obs)

        if self.f_statistic is not None:
            f_str = f"{self.f_statistic[0]:.4f}"
            fp_str = f"{self.f_statistic[1]:.4f}"
        else:
            f_str = "N/A"
            fp_str = "N/A"

        return (
            f"{self.dep_var} & {n_str} & {r2_str} & {adj_r2_str} & "
            f"{f_str} & {fp_str} & {aic_str} & {bic_str} \\\\"
        )


# ------------------------------------------------------------------
# Module-level helper: compare multiple models
# ------------------------------------------------------------------

def compare_models(results: Sequence[ModelResult]) -> pd.DataFrame:
    """Horizontally compare multiple regression models.

    Produces a coefficient comparison table where each column is a model.
    The bottom rows include model-level statistics (N, R², Adj-R², AIC, BIC).

    Each coefficient cell shows ``coef (se)`` followed by significance stars.

    Args:
        results: A list of ModelResult objects.

    Returns:
        A DataFrame with models as columns, variable names as index rows,
        and summary statistics appended at the bottom.
    """
    if not results:
        return pd.DataFrame()

    # Collect all variable names across all models
    all_vars: List[str] = []
    for res in results:
        for c in res.coefficients:
            if c.name not in all_vars:
                all_vars.append(c.name)

    # Build the comparison table
    model_names: List[str] = []
    model_data: Dict[str, List[Optional[str]]] = {}
    for var_name in all_vars:
        model_data[var_name] = []

    for i, res in enumerate(results):
        coef_map = {c.name: c for c in res.coefficients}
        label = f"Model {i+1}"
        if res.specification:
            label += f"\n({res.specification})"
        model_names.append(label)

        for var_name in all_vars:
            if var_name in coef_map:
                c = coef_map[var_name]
                cell = f"{c.coef:.4f} ({c.se:.4f}){c.significance}"
            else:
                cell = ""
            model_data[var_name].append(cell)

    # Create coefficient rows
    rows: List[Dict[str, object]] = []
    for var_name in all_vars:
        row: Dict[str, object] = {"变量": var_name}
        for i, label in enumerate(model_names):
            row[label] = model_data[var_name][i]
        rows.append(row)

    # Add summary statistics rows at the bottom
    stat_labels = ["N", "R²", "Adj-R²", "AIC", "BIC"]
    stat_getters = [
        lambda r: str(r.n_obs),
        lambda r: f"{r.r_squared:.4f}" if r.r_squared is not None else "N/A",
        lambda r: f"{r.adj_r_squared:.4f}" if r.adj_r_squared is not None else "N/A",
        lambda r: f"{r.aic:.2f}",
        lambda r: f"{r.bic:.2f}",
    ]

    for stat_label, getter in zip(stat_labels, stat_getters):
        row: Dict[str, object] = {"变量": stat_label}
        for i, label in enumerate(model_names):
            row[label] = getter(results[i])
        rows.append(row)

    return pd.DataFrame(rows)
