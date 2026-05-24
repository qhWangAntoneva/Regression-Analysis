# encoding: utf-8
"""
Publication-quality LaTeX table renderer for regression results.

Generates booktabs-style LaTeX tables suitable for academic papers
(APA7 format by default).  Uses Python string formatting instead of
Jinja2 to avoid conflicts with LaTeX backslash sequences.
"""

from __future__ import annotations

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

        Args:
            model_result: A ``ModelResult`` instance.
            title: Optional table title (wrapped in ``table`` environment).
            caption: Optional table caption (requires ``title`` to be set).
            label: LaTeX ``\\label`` suffix (default ``'model'``).

        Returns:
            LaTeX source code string.
        """
        lines: List[str] = []

        has_title = bool(title)

        if has_title:
            lines.append("\\begin{table}[htbp]")
            lines.append("\\centering")
            lines.append(f"\\caption{{ {caption} }}")
            lines.append(f"\\label{{tab:{label}}}")

        lines.append("\\begin{tabular}{lrrrrr}")
        lines.append("\\toprule")
        lines.append(
            "Variable & Coefficient & SE & $t$ & $p$ & \\multicolumn{1}{c}{95\\% CI} \\\\"
        )
        lines.append("\\midrule")

        for coef in model_result.coefficients:
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
        lines.append("\\multicolumn{6}{l}{\\textit{Fit statistics}} \\\\")

        r2_str = f"{model_result.r_squared:.4f}" if model_result.r_squared is not None else "N/A"
        adj_r2_str = f"{model_result.adj_r_squared:.4f}" if model_result.adj_r_squared is not None else "N/A"

        lines.append(f"N & \\multicolumn{{5}}{{l}}{{{model_result.n_obs}}} \\\\")
        lines.append(f"R$^2$ & \\multicolumn{{5}}{{l}}{{{r2_str}}} \\\\")
        lines.append(f"Adj. R$^2$ & \\multicolumn{{5}}{{l}}{{{adj_r2_str}}} \\\\")

        if model_result.f_statistic is not None:
            f_val, f_p = model_result.f_statistic
            lines.append(
                f"F-statistic & \\multicolumn{{5}}{{l}}{{{f_val:.4f} "
                f"($p$ = {f_p:.4f})}} \\\\"
            )
        else:
            lines.append("F-statistic & \\multicolumn{5}{l}{N/A} \\\\")

        lines.append("\\bottomrule")
        lines.append(LatexRenderer._SIG_FOOTNOTE)
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
        stars.  A footer section reports fit statistics per model.

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
        header_parts = ["Variable"] + [f"\\multicolumn{{1}}{{c}}{{{l}}}" for l in model_labels]
        lines.append(" & ".join(header_parts) + " \\\\")
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
            for res in model_results:
                match = [c for c in res.coefficients if c.name == var_name]
                if match:
                    c = match[0]
                    cells.append(f"$ {c.coef:.4f} $ \\newline ({c.se:.4f}){c.significance}")
                else:
                    cells.append("")
            lines.append(" & ".join([var_name] + cells) + " \\\\")

        # Fit statistics
        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{{model_count + 1}}}{{l}}{{\\textit{{Fit statistics}}}} \\\\")

        stat_defs: List[tuple] = [
            ("N", lambda r: str(r.n_obs)),
            ("R$^2$", lambda r: f"{r.r_squared:.4f}" if r.r_squared is not None else "N/A"),
            ("Adj. R$^2$", lambda r: f"{r.adj_r_squared:.4f}" if r.adj_r_squared is not None else "N/A"),
            ("F-statistic", lambda r: f"{r.f_statistic[0]:.4f}" if r.f_statistic is not None else "N/A"),
            ("AIC", lambda r: f"{r.aic:.2f}"),
            ("BIC", lambda r: f"{r.bic:.2f}"),
        ]

        for stat_label, getter in stat_defs:
            values = [str(getter(res)) for res in model_results]
            lines.append(" & ".join([stat_label] + values) + " \\\\")

        lines.append("\\bottomrule")
        lines.append(
            f"\\multicolumn{{{model_count + 1}}}{{l}}{{\\textit{{Notes:}} "
            "*** $p<0.01$, ** $p<0.05$, * $p<0.1$}"
        )
        lines.append("\\end{tabular}")

        if has_caption:
            lines.append("\\end{table}")

        return "\n".join(lines) + "\n"
