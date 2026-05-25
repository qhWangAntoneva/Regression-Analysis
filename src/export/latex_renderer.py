# encoding: utf-8
"""
Publication-quality LaTeX table renderer for regression results.

Generates booktabs-style LaTeX tables suitable for academic papers
(APA7 format by default).  Uses Python string formatting instead of
Jinja2 to avoid conflicts with LaTeX backslash sequences.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from src.results.table import ModelResult


class LatexRenderer:
    """Render regression results as publication-quality LaTeX tables.

    Two primary methods:
        - ``render_single`` — one model as a coefficient table.
        - ``render_comparison`` — multiple models side-by-side.
    """

    # Shared significance footnote
    _SIG_FOOTNOTE = (
        "\\multicolumn{6}{l}{\\textit{Notes:} "
        "*** $p<0.01$, ** $p<0.05$, * $p<0.1$}"
    )

    # ==================================================================
    # Public API
    # ==================================================================

    @staticmethod
    def render_single(
        model_result: ModelResult,
        title: str = "",
        caption: str = "",
        label: str = "model",
    ) -> str:
        """Render a single-model LaTeX coefficient table.

        For OLS models the table shows coefficients with t-statistics and
        R-squared goodness-of-fit.  For logit models it shows odds ratios
        (OR = exp(B)) with z-statistics and pseudo R-squared.

        Args:
            model_result: A ``ModelResult`` instance.
            title: Optional table title (wrapped in ``table`` environment).
            caption: Optional table caption (requires ``title`` to be set).
            label: LaTeX ``\\label`` suffix (default ``'model'``).

        Returns:
            LaTeX source code string.
        """
        is_logit = model_result.model_type == "logit"
        lines: List[str] = []

        has_title = bool(title)

        if not caption:
            caption = (
                "Logistic Regression Results"
                if is_logit
                else "OLS Regression Results"
            )

        if has_title:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append(f"\\caption{{ {caption} }}")
            lines.append(f"\\label{{tab:{label}}}")

        if is_logit:
            # Logit table: OR column, z-statistic, 7 columns
            stat_letter = "z"
            col_header = "OR (exp(B))"
            lines.append("\\begin{tabular}{lrrrrrr}")
            lines.append("\\toprule")
            lines.append(
                "Variable & OR (exp($B$)) & SE & $z$ & $p$ & "
                "\\multicolumn{1}{c}{95\\% CI} \\\\"
            )
        else:
            # OLS table: Coefficient, t-statistic, 6 columns
            stat_letter = "t"
            col_header = "Coefficient"
            lines.append("\\begin{tabular}{lrrrrr}")
            lines.append("\\toprule")
            lines.append(
                "Variable & Coefficient & SE & $t$ & $p$ & "
                "\\multicolumn{1}{c}{95\\% CI} \\\\"
            )
        lines.append("\\midrule")

        for coef in model_result.coefficients:
            if is_logit:
                # Show odds ratios instead of raw coefficients
                or_val = math.exp(coef.coef)
                ci_lo = math.exp(coef.ci_lower)
                ci_hi = math.exp(coef.ci_upper)
                row = (
                    f"{coef.name} & "
                    f"{or_val:.4f} & "
                    f"{coef.se:.4f} & "
                    f"{coef.t_stat:.4f} & "
                    f"{coef.pvalue:.4f} & "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]{coef.significance} \\\\"
                )
            else:
                row = (
                    f"{coef.name} & "
                    f"{coef.coef:.4f} & "
                    f"{coef.se:.4f} & "
                    f"{coef.t_stat:.4f} & "
                    f"{coef.pvalue:.4f} & "
                    f"[{coef.ci_lower:.4f}, {coef.ci_upper:.4f}]{coef.significance} \\\\"
                )
            lines.append(row)

        lines.append("\\midrule")

        if is_logit:
            # Logit footer: pseudo-R² and LR χ²
            n_cols = 7
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Fit statistics}}}} \\\\"
            )

            pseudo_str = (
                f"{model_result.pseudo_r_squared:.4f}"
                if model_result.pseudo_r_squared is not None
                else "N/A"
            )
            llr_str = (
                f"{model_result.llr:.4f}"
                if model_result.llr is not None
                else "N/A"
            )
            llr_p_str = (
                f"{model_result.llr_pvalue:.4f}"
                if model_result.llr_pvalue is not None
                else "N/A"
            )

            lines.append(
                f"N & \\multicolumn{{{n_cols - 1}}}{{l}}{{{model_result.n_obs}}} \\\\"
            )
            lines.append(
                "McFadden's pseudo-$R^2$ & "
                f"\\multicolumn{{{n_cols - 1}}}{{l}}{{{pseudo_str}}} \\\\"
            )
            lines.append(
                f"LR $\\chi^2$ & \\multicolumn{{{n_cols - 1}}}{{l}}"
                f"{{{llr_str} ($p$ = {llr_p_str})}} \\\\"
            )
            lines.append(
                "AIC & "
                f"\\multicolumn{{{n_cols - 1}}}{{l}}{{{model_result.aic:.2f}}} \\\\"
            )
            lines.append(
                "BIC & "
                f"\\multicolumn{{{n_cols - 1}}}{{l}}{{{model_result.bic:.2f}}} \\\\"
            )

            # Significance footnote
            lines.append("\\bottomrule")
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Notes:}} "
                "*** $p<0.01$, ** $p<0.05$, * $p<0.1$}"
            )
        else:
            # OLS footer: R² and F-test
            n_cols = 6
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Fit statistics}}}} \\\\"
            )

            r2_str = (
                f"{model_result.r_squared:.4f}"
                if model_result.r_squared is not None
                else "N/A"
            )
            adj_r2_str = (
                f"{model_result.adj_r_squared:.4f}"
                if model_result.adj_r_squared is not None
                else "N/A"
            )

            lines.append(
                f"N & \\multicolumn{{{n_cols - 1}}}{{l}}{{{model_result.n_obs}}} \\\\"
            )
            lines.append(
                f"R$^2$ & \\multicolumn{{{n_cols - 1}}}{{l}}{{{r2_str}}} \\\\"
            )
            lines.append(
                f"Adj. R$^2$ & \\multicolumn{{{n_cols - 1}}}{{l}}{{{adj_r2_str}}} \\\\"
            )

            if model_result.f_statistic is not None:
                f_val, f_p = model_result.f_statistic
                lines.append(
                    f"F-statistic & \\multicolumn{{{n_cols - 1}}}{{l}}"
                    f"{{{f_val:.4f} ($p$ = {f_p:.4f})}} \\\\"
                )
            else:
                lines.append(
                    f"F-statistic & \\multicolumn{{{n_cols - 1}}}{{l}}{{N/A}} \\\\"
                )

            lines.append("\\bottomrule")
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{Notes:}} "
                "*** $p<0.01$, ** $p<0.05$, * $p<0.1$}"
            )

        lines.append("\\end{tabular}")

        if has_title:
            lines.append("\\end{table}")

        return "\n".join(lines) + "\n"

    @staticmethod
    def render_comparison(
        model_results: Sequence[ModelResult],
        captions: Optional[List[str]] = None,
        model_labels: Optional[List[str]] = None,
    ) -> str:
        """Render a multi-model side-by-side LaTeX comparison table.

        Each coefficient cell shows ``coef (se)`` followed by significance
        stars.  A footer section reports fit statistics per model.  For mixed
        OLS/logit models a "Model Type" row is added and appropriate
        diagnostics are shown per model.

        Args:
            model_results: List of ``ModelResult`` instances.
            captions: Optional list of captions (only first is used).
            model_labels: Optional custom column headers.  Defaults to
                ``'Model 1'``, ``'Model 2'``, ...

        Returns:
            LaTeX source code string.
        """
        if not model_results:
            return ""

        model_count = len(model_results)

        if model_labels is None:
            model_labels = [f"Model {i + 1}" for i in range(model_count)]

        # Detect mixed model types
        model_types = [r.model_type for r in model_results]
        has_logit = any(t == "logit" for t in model_types)
        has_ols = any(t != "logit" for t in model_types)
        mixed = has_logit and has_ols

        lines: List[str] = []

        has_caption = captions and len(captions) > 0 and bool(captions[0])

        if has_caption:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append(f"\\caption{{ {captions[0]} }}")
            lines.append("\\label{tab:comparison}")

        # Column spec: left column + one center column per model
        col_spec = "l " + " ".join(["c"] * model_count)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header row
        header_parts = ["Variable"] + [
            f"\\multicolumn{{1}}{{c}}{{{l}}}" for l in model_labels
        ]
        lines.append(" & ".join(header_parts) + " \\\\")
        lines.append("\\midrule")

        # Model type row (shown when mixed or logit models present)
        if mixed or has_logit:
            type_parts = ["\\textit{Model type}"] + [
                "Logit" if r.model_type == "logit" else "OLS"
                for r in model_results
            ]
            lines.append(" & ".join(type_parts) + " \\\\")
            lines.append("\\midrule")

        # Collect all variable names across all models
        all_vars: List[str] = []
        for res in model_results:
            for coef in res.coefficients:
                if coef.name not in all_vars:
                    all_vars.append(coef.name)

        # Coefficient rows
        for var_name in all_vars:
            cells: List[str] = []
            for i, res in enumerate(model_results):
                match = [c for c in res.coefficients if c.name == var_name]
                if match:
                    c = match[0]
                    # Show raw coefficient + SE for all models
                    cells.append(
                        f"$ {c.coef:.4f} $ \\newline ({c.se:.4f}){c.significance}"
                    )
                else:
                    cells.append("")
            lines.append(" & ".join([var_name] + cells) + " \\\\")

        # Fit statistics
        lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{model_count + 1}}}{{l}}{{\\textit{{Fit statistics}}}} \\\\"
        )

        # N (always)
        lines.append(
            "N & " + " & ".join(str(r.n_obs) for r in model_results) + " \\\\"
        )

        # R² or pseudo R² (conditional on mix)
        has_any_r2 = any(r.r_squared is not None for r in model_results)
        has_any_pseudo = any(r.pseudo_r_squared is not None for r in model_results)

        if has_any_r2 and has_any_pseudo and mixed:
            # Show both R² and pseudo R² rows when mixed
            r2_vals = []
            pseudo_vals = []
            for r in model_results:
                if r.pseudo_r_squared is not None:
                    r2_vals.append("--")
                    pseudo_vals.append(f"{r.pseudo_r_squared:.4f}")
                elif r.r_squared is not None:
                    r2_vals.append(f"{r.r_squared:.4f}")
                    pseudo_vals.append("--")
                else:
                    r2_vals.append("N/A")
                    pseudo_vals.append("N/A")
            lines.append("R$^2$ & " + " & ".join(r2_vals) + " \\\\")
            lines.append(
                "Pseudo-$R^2$ & " + " & ".join(pseudo_vals) + " \\\\"
            )
        elif has_any_r2:
            lines.append(
                "R$^2$ & "
                + " & ".join(
                    f"{r.r_squared:.4f}" if r.r_squared is not None else "N/A"
                    for r in model_results
                )
                + " \\\\"
            )
            # Adj R² for OLS-only
            lines.append(
                "Adj. R$^2$ & "
                + " & ".join(
                    f"{r.adj_r_squared:.4f}"
                    if r.adj_r_squared is not None
                    else "N/A"
                    for r in model_results
                )
                + " \\\\"
            )
        elif has_any_pseudo:
            lines.append(
                "Pseudo-$R^2$ & "
                + " & ".join(
                    f"{r.pseudo_r_squared:.4f}"
                    if r.pseudo_r_squared is not None
                    else "N/A"
                    for r in model_results
                )
                + " \\\\"
            )

        # F or LR test (conditional)
        if mixed:
            # Show both F and LR rows
            f_vals = []
            lr_vals = []
            for r in model_results:
                if r.f_statistic is not None:
                    f_vals.append(f"{r.f_statistic[0]:.4f}")
                else:
                    f_vals.append("--")
                if r.llr is not None:
                    lr_vals.append(f"{r.llr:.4f}")
                else:
                    lr_vals.append("--")
            lines.append("F-statistic & " + " & ".join(f_vals) + " \\\\")
            lines.append("LR $\\chi^2$ & " + " & ".join(lr_vals) + " \\\\")
        elif has_ols:
            lines.append(
                "F-statistic & "
                + " & ".join(
                    f"{r.f_statistic[0]:.4f}"
                    if r.f_statistic is not None
                    else "N/A"
                    for r in model_results
                )
                + " \\\\"
            )
        else:
            # All logit — show LR chi2
            lines.append(
                "LR $\\chi^2$ & "
                + " & ".join(
                    f"{r.llr:.4f}" if r.llr is not None else "N/A"
                    for r in model_results
                )
                + " \\\\"
            )

        # AIC and BIC (always)
        lines.append(
            "AIC & "
            + " & ".join(f"{r.aic:.2f}" for r in model_results)
            + " \\\\"
        )
        lines.append(
            "BIC & "
            + " & ".join(f"{r.bic:.2f}" for r in model_results)
            + " \\\\"
        )

        lines.append("\\bottomrule")
        lines.append(
            f"\\multicolumn{{{model_count + 1}}}{{l}}{{\\textit{{Notes:}} "
            "*** $p<0.01$, ** $p<0.05$, * $p<0.1$}"
        )
        lines.append("\\end{tabular}")

        if has_caption:
            lines.append("\\end{table}")

        return "\n".join(lines) + "\n"
