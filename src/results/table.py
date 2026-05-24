# encoding: utf-8
"""Unified result data structures for regression model output.

Provides dataclasses for storing coefficient-level and model-level results,
along with utilities for formatting and converting to pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
