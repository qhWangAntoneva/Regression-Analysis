# encoding: utf-8
"""
Logit visualization module.

Provides ROC curve and Odds Ratio forest plot for binary logistic
regression results.  Both plots use Plotly for interactive rendering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from src.results.table import ModelResult

# ---------------------------------------------------------------------------
# Try importing sklearn, fall back to manual computation
# ---------------------------------------------------------------------------

try:
    from sklearn.metrics import auc as sklearn_auc
    from sklearn.metrics import roc_curve as sklearn_roc_curve

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Manual ROC computation (used when sklearn is not available)
# ---------------------------------------------------------------------------


def _manual_roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute ROC curve manually.

    Sorts observations by descending predicted probability and scans
    threshold-by-threshold to accumulate TPR and FPR.

    Args:
        y_true: 1-D array of binary ground-truth labels (0 or 1).
        y_score: 1-D array of predicted probabilities.

    Returns:
        A tuple ``(fpr, tpr, thresholds)``, each a 1-D float64 array.
    """
    # Sort by descending score
    desc_inds = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[desc_inds]
    y_score_sorted = y_score[desc_inds]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    # Degenerate cases
    if n_pos == 0 or n_neg == 0:
        # Perfect separation: trivial ROC
        return (
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0], dtype=np.float64),
        )

    tpr_list: list[float] = []
    fpr_list: list[float] = []
    thresholds: list[float] = []

    tp = 0
    fp = 0
    prev_score: float | None = None

    for i in range(len(y_true_sorted)):
        score = float(y_score_sorted[i])
        if prev_score is not None and score != prev_score:
            tpr_list.append(tp / n_pos)
            fpr_list.append(fp / n_neg)
            thresholds.append(prev_score)
        prev_score = score

        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

    # Final point (all classified as positive)
    tpr_list.append(tp / n_pos)
    fpr_list.append(fp / n_neg)
    thresholds.append(float(y_score_sorted[-1]))

    # Add the (0,0) start point
    fpr = np.array([0.0] + fpr_list, dtype=np.float64)
    tpr = np.array([0.0] + tpr_list, dtype=np.float64)
    thr = np.array([1.0] + thresholds, dtype=np.float64)

    return fpr, tpr, thr


def _manual_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Compute AUC via the trapezoidal rule (identical to sklearn's auc)."""
    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def roc_curve_plot(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> Figure:
    """Plot an ROC curve with AUC annotation.

    Uses ``sklearn.metrics.roc_curve`` and ``sklearn.metrics.auc`` when
    scikit-learn is available; otherwise falls back to a manual computation
    that produces identical results.

    Args:
        y_true: 1-D array of binary ground-truth labels (0 or 1).
        y_pred_prob: 1-D array of predicted probabilities in [0, 1].

    Returns:
        A plotly ``Figure`` containing the ROC curve, a diagonal
        reference line (random classifier), and an AUC annotation.

    Raises:
        ImportError: Plotly is not installed.
        ValueError: Input arrays are empty or have mismatched lengths.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is not installed. Run: pip install plotly")

    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred_prob = np.asarray(y_pred_prob, dtype=np.float64).ravel()

    if y_true.size == 0:
        raise ValueError("y_true is empty")
    if y_pred_prob.size == 0:
        raise ValueError("y_pred_prob is empty")
    if y_true.shape != y_pred_prob.shape:
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred_prob={len(y_pred_prob)}"
        )

    # Compute ROC curve
    if _HAS_SKLEARN:
        fpr, tpr, _ = sklearn_roc_curve(y_true, y_pred_prob)
        auc_val = sklearn_auc(fpr, tpr)
    else:
        fpr, tpr, _ = _manual_roc_curve(y_true, y_pred_prob)
        auc_val = _manual_auc(fpr, tpr)

    # Build figure
    fig = go.Figure()

    # Diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="gray", width=1, dash="dash"),
            name="Random classifier (AUC=0.5)",
            hoverinfo="skip",
        )
    )

    # ROC curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.1)",
            name=f"ROC curve (AUC={auc_val:.4f})",
            hovertemplate="FPR: %{x:.4f}<br>TPR: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": f"ROC Curve (AUC = {auc_val:.4f})",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity)",
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(range=[-0.02, 1.02], constrain="domain"),
        yaxis=dict(range=[-0.02, 1.02], constrain="domain"),
        width=600,
        height=500,
        showlegend=True,
        legend=dict(
            x=0.6,
            y=0.1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        ),
    )

    return fig


def odds_ratio_plot(result: ModelResult) -> Figure:
    """Forest plot of odds ratios with 95% confidence intervals on a log scale.

    Computes OR = exp(coef) and OR CI = exp(conf_int) for each coefficient
    (excluding the intercept).  Coefficients with p < 0.05 are coloured blue;
    non-significant ones are grey.

    Args:
        result: A ``ModelResult`` whose ``model_type`` attribute equals ``'logit'``.

    Returns:
        A plotly ``Figure`` with a dot-whisker forest plot.  The x-axis is
        on a log scale and a vertical reference line is drawn at OR = 1.

    Raises:
        ImportError: Plotly is not installed.
        ValueError: ``result.model_type`` is not ``'logit'``, or no non-intercept
            coefficients are available.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("plotly is not installed. Run: pip install plotly")

    if result.model_type != "logit":
        raise ValueError(
            f"odds_ratio_plot requires model_type='logit', got '{result.model_type}'"
        )

    # Filter out Intercept and build OR entries
    entries: list[dict] = []
    for c in result.coefficients:
        if c.name.lower() in ("intercept", "const"):
            continue
        or_val = float(np.exp(c.coef))
        or_low = float(np.exp(c.ci_lower))
        or_high = float(np.exp(c.ci_upper))
        entries.append(
            {
                "name": c.name,
                "or": or_val,
                "or_low": or_low,
                "or_high": or_high,
                "pvalue": c.pvalue,
                "significant": c.pvalue < 0.05,
            }
        )

    if not entries:
        raise ValueError("No non-intercept coefficients available for odds ratio plot")

    # Sort by OR magnitude descending
    entries.sort(key=lambda e: abs(np.log(e["or"])), reverse=True)

    n = len(entries)
    y_positions = list(range(n - 1, -1, -1))
    names = [e["name"] for e in entries]
    or_vals = [e["or"] for e in entries]
    or_lows = [e["or_low"] for e in entries]
    or_highs = [e["or_high"] for e in entries]
    sig_flags = [e["significant"] for e in entries]
    pvals = [e["pvalue"] for e in entries]

    blue = "#1f77b4"
    grey = "#7f7f7f"

    fig = go.Figure()

    # Draw whiskers (CI lines) for each coefficient
    for i in range(n):
        color = blue if sig_flags[i] else grey
        fig.add_trace(
            go.Scatter(
                x=[or_lows[i], or_highs[i]],
                y=[y_positions[i], y_positions[i]],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Draw dot markers
    for i in range(n):
        color = blue if sig_flags[i] else grey
        fig.add_trace(
            go.Scatter(
                x=[or_vals[i]],
                y=[y_positions[i]],
                mode="markers",
                marker=dict(color=color, size=10, symbol="circle"),
                name="Significant (p<0.05)" if sig_flags[i] and i == next(
                    j for j, s in enumerate(sig_flags) if s
                ) else (
                    "Non-significant (p>=0.05)" if not sig_flags[i] and i == next(
                        j for j, s in enumerate(sig_flags) if not s
                    ) else None
                ),
                showlegend=False,  # We add a manual legend below
                hovertemplate=(
                    f"<b>{names[i]}</b><br>"
                    f"OR: {or_vals[i]:.4f}<br>"
                    f"95% CI: [{or_lows[i]:.4f}, {or_highs[i]:.4f}]<br>"
                    f"p-value: {pvals[i]:.4f}"
                    "<extra></extra>"
                ),
            )
        )

    # Add dummy traces for legend
    has_sig = any(sig_flags)
    has_nonsig = any(not s for s in sig_flags)
    if has_sig:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=blue, size=10, symbol="circle"),
                name="Significant (p<0.05)",
                showlegend=True,
            )
        )
    if has_nonsig:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(color=grey, size=10, symbol="circle"),
                name="Non-significant (p>=0.05)",
                showlegend=True,
            )
        )

    # Reference line at OR=1
    fig.add_vline(
        x=1,
        line=dict(color="gray", width=1, dash="dash"),
        annotation_text="OR = 1 (no effect)",
        annotation_position="top",
        annotation_font=dict(size=10, color="gray"),
    )

    fig.update_layout(
        title={
            "text": "Odds Ratio Forest Plot (Logit)",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Odds Ratio (log scale)",
        xaxis_type="log",
        yaxis=dict(
            tickvals=y_positions,
            ticktext=names,
            title="",
        ),
        template="plotly_white",
        hovermode="closest",
        height=max(300, n * 50),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig
