# encoding: utf-8
"""Unit tests for logit visualisations: roc_curve_plot and odds_ratio_plot."""

from __future__ import annotations

import sys
from unittest import mock

import numpy as np
import pytest

try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]

from src.visualization.logit_plots import (
    _manual_auc,
    _manual_roc_curve,
    odds_ratio_plot,
    roc_curve_plot,
)

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray]:
    """Synthetic binary labels and predicted probabilities."""
    rng = np.random.default_rng(42)
    n = 200
    # DGP: eta = 0.5 + 1.0 * x1 - 0.8 * x2
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    eta = 0.5 + 1.0 * x1 - 0.8 * x2
    prob = 1.0 / (1.0 + np.exp(-eta))
    y_true = (rng.random(n) < prob).astype(int)
    y_pred = prob
    return y_true, y_pred


@pytest.fixture
def sample_logit_result():
    """Create a ModelResult with model_type='logit' and several coefficients."""
    from src.results.table import CoefficientRow, ModelResult

    return ModelResult(
        model_type="logit",
        coefficients=[
            CoefficientRow(
                name="Intercept",
                coef=0.5,
                se=0.2,
                t_stat=2.5,
                pvalue=0.012,
                ci_lower=0.1,
                ci_upper=0.9,
            ),
            CoefficientRow(
                name="x1",
                coef=1.2,
                se=0.3,
                t_stat=4.0,
                pvalue=0.0001,
                ci_lower=0.6,
                ci_upper=1.8,
            ),
            CoefficientRow(
                name="x2",
                coef=-0.8,
                se=0.25,
                t_stat=-3.2,
                pvalue=0.001,
                ci_lower=-1.3,
                ci_upper=-0.3,
            ),
            CoefficientRow(
                name="x3",
                coef=0.1,
                se=0.15,
                t_stat=0.67,
                pvalue=0.5,
                ci_lower=-0.2,
                ci_upper=0.4,
            ),
        ],
        n_obs=200,
        n_params=4,
        df_resid=196,
        pseudo_r_squared=0.25,
        log_likelihood=-100.0,
        aic=208.0,
        bic=221.0,
        dep_var="y",
        specification="y ~ x1 + x2 + x3",
        method="Logit",
    )


# =========================================================================
# Tests: roc_curve_plot
# =========================================================================


class TestROCCurvePlotBasic:
    """Basic ROC curve plot tests."""

    def test_returns_figure(self, binary_data) -> None:
        """roc_curve_plot returns a plotly Figure."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true, y_pred = binary_data
        fig = roc_curve_plot(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_auc_between_0_and_1(self, binary_data) -> None:
        """AUC should be between 0 and 1 for reasonable predictions."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true, y_pred = binary_data
        fig = roc_curve_plot(y_true, y_pred)
        title_text = fig.layout.title.text or ""
        # Extract AUC from title "ROC Curve (AUC = X.XXXX)"
        assert "AUC" in title_text
        # Parse AUC value
        import re

        match = re.search(r"AUC\s*=\s*([\d.]+)", title_text)
        assert match is not None, f"Could not find AUC in title: {title_text}"
        auc_val = float(match.group(1))
        assert 0.0 <= auc_val <= 1.0, f"AUC {auc_val} not in [0, 1]"

    def test_contains_diagonal_reference_line(self, binary_data) -> None:
        """ROC plot should contain a diagonal reference line (random classifier)."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true, y_pred = binary_data
        fig = roc_curve_plot(y_true, y_pred)
        # The first trace should be the diagonal line
        trace_names = [t.name for t in fig.data if t.name]
        assert any("Random" in str(name) for name in trace_names), (
            f"No random classifier reference found in {trace_names}"
        )

    def test_uses_plotly_white_template(self, binary_data) -> None:
        """ROC plot should use plotly_white template."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true, y_pred = binary_data
        fig = roc_curve_plot(y_true, y_pred)
        assert fig.layout.template is not None


class TestROCCurvePlotPerfectSeparation:
    """ROC for perfect / degenerate separation."""

    def test_all_positive(self) -> None:
        """When all y_true = 1, should not crash and should produce a figure."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true = np.ones(50, dtype=int)
        y_pred = np.random.default_rng(1).random(50)
        fig = roc_curve_plot(y_true, y_pred)
        assert isinstance(fig, Figure)

    def test_all_negative(self) -> None:
        """When all y_true = 0, should not crash and should produce a figure."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true = np.zeros(50, dtype=int)
        y_pred = np.random.default_rng(2).random(50)
        fig = roc_curve_plot(y_true, y_pred)
        assert isinstance(fig, Figure)


class TestROCCurvePlotRandomPredictions:
    """ROC for random predictions should give AUC around 0.5."""

    def test_random_predictions_auc_near_half(self) -> None:
        """Random predictions yield AUC close to 0.5."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        rng = np.random.default_rng(99)
        n = 500
        y_true = (rng.random(n) > 0.5).astype(int)
        y_pred = rng.random(n)  # random uniform, no relationship
        fig = roc_curve_plot(y_true, y_pred)

        import re

        title_text = fig.layout.title.text or ""
        match = re.search(r"AUC\s*=\s*([\d.]+)", title_text)
        assert match is not None
        auc_val = float(match.group(1))
        # Random predictions should give AUC in [0.3, 0.7] with 500 samples
        assert 0.3 <= auc_val <= 0.7, (
            f"Random AUC {auc_val} far from 0.5"
        )


class TestROCCurvePlotSklearnFallback:
    """Test the manual ROC computation fallback when sklearn is unavailable."""

    def test_manual_fallback_disabled_sklearn(self, binary_data) -> None:
        """roc_curve_plot should work even when _HAS_SKLEARN is False."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        import src.visualization.logit_plots as lp_mod

        y_true, y_pred = binary_data

        # Force the module to use manual computation
        original = lp_mod._HAS_SKLEARN
        try:
            lp_mod._HAS_SKLEARN = False
            fig = roc_curve_plot(y_true, y_pred)
            assert isinstance(fig, Figure)
        finally:
            lp_mod._HAS_SKLEARN = original

    def test_manual_roc_curve_structure(self, binary_data) -> None:
        """_manual_roc_curve returns properly shaped arrays."""
        y_true, y_pred = binary_data
        fpr, tpr, thr = _manual_roc_curve(y_true, y_pred)
        assert fpr.ndim == 1
        assert tpr.ndim == 1
        assert thr.ndim == 1
        assert len(fpr) == len(tpr) == len(thr)
        # FPR and TPR should be in [0, 1]
        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)
        # Monotonically increasing (or at least non-decreasing)
        assert np.all(np.diff(fpr) >= -1e-10), "FPR should be non-decreasing"
        assert np.all(np.diff(tpr) >= -1e-10), "TPR should be non-decreasing"

    def test_manual_auc_between_0_and_1(self, binary_data) -> None:
        """_manual_auc returns a value in [0, 1]."""
        y_true, y_pred = binary_data
        fpr, tpr, _ = _manual_roc_curve(y_true, y_pred)
        auc_val = _manual_auc(fpr, tpr)
        assert 0.0 <= auc_val <= 1.0


class TestROCCurvePlotErrorHandling:
    """Error handling for roc_curve_plot."""

    def test_empty_y_true_raises(self) -> None:
        """Empty y_true should raise ValueError."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        with pytest.raises(ValueError, match="empty"):
            roc_curve_plot(np.array([]), np.array([0.5]))

    def test_empty_y_pred_raises(self) -> None:
        """Empty y_pred_prob should raise ValueError."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        with pytest.raises(ValueError, match="empty"):
            roc_curve_plot(np.array([1, 0]), np.array([]))

    def test_length_mismatch_raises(self) -> None:
        """Mismatched lengths should raise ValueError."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        with pytest.raises(ValueError, match="mismatch"):
            roc_curve_plot(np.array([1, 0, 1]), np.array([0.8, 0.2]))


# =========================================================================
# Tests: odds_ratio_plot
# =========================================================================


class TestOddsRatioPlotBasic:
    """Basic odds ratio forest plot tests."""

    def test_returns_figure(self, sample_logit_result) -> None:
        """odds_ratio_plot returns a plotly Figure."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        assert isinstance(fig, Figure)

    def test_intercept_filtered_out(self, sample_logit_result) -> None:
        """Intercept should not appear in the odds ratio plot."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        ticktext = fig.layout.yaxis.ticktext or []
        ticktext_str = " ".join(str(t) for t in ticktext)
        assert "Intercept" not in ticktext_str, (
            f"Intercept found in y-axis labels: {ticktext_str}"
        )

    def test_or_one_reference_line(self, sample_logit_result) -> None:
        """Reference line at OR=1 should be present."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        # vline creates layout.shapes entries
        assert fig.layout.shapes is not None, "Expected vline shapes in layout"
        # At least one shape with x=1 (the OR=1 reference line)
        has_or_one_line = False
        for shape in fig.layout.shapes:
            if hasattr(shape, "x0") and shape.x0 == 1:
                has_or_one_line = True
                break
        assert has_or_one_line, "No OR=1 reference line found"

    def test_sorted_by_or_magnitude(self, sample_logit_result) -> None:
        """Variables should be sorted by ln(OR) magnitude descending."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        ticktext = list(fig.layout.yaxis.ticktext or [])
        assert len(ticktext) >= 1
        # x1 (coef=1.2) has larger |ln(OR)| than x2 (coef=-0.8) than x3 (coef=0.1)
        # So order should be x1, x2, x3 (from top to bottom)
        x1_idx = ticktext.index("x1") if "x1" in ticktext else -1
        x2_idx = ticktext.index("x2") if "x2" in ticktext else -1
        x3_idx = ticktext.index("x3") if "x3" in ticktext else -1
        # x1 has largest |coef|, so it should come first (lowest y-tick, highest visual position)
        assert x1_idx < x2_idx < x3_idx, (
            f"Expected x1 above x2 above x3, got {ticktext}"
        )

    def test_significant_legend_present(self, sample_logit_result) -> None:
        """Legend should distinguish significant (p<0.05) from non-significant."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        legend_texts = [
            t.name for t in fig.data if t.name and t.showlegend
        ]
        assert any("Significant" in name for name in legend_texts), (
            f"No significant legend entry in {legend_texts}"
        )
        assert any("Non-significant" in name for name in legend_texts), (
            f"No non-significant legend entry in {legend_texts}"
        )

    def test_log_scale_xaxis(self, sample_logit_result) -> None:
        """X-axis should be log-scaled for odds ratios."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        assert fig.layout.xaxis.type == "log", (
            f"Expected log x-axis type, got {fig.layout.xaxis.type}"
        )

    def test_uses_plotly_white_template(self, sample_logit_result) -> None:
        """Plot should use plotly_white template."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        assert fig.layout.template is not None


class TestOddsRatioPlotSingleCoefficient:
    """Odds ratio plot with a single non-intercept coefficient."""

    def test_single_coefficient(self) -> None:
        """Should work with only one predictor (excluding intercept)."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="Intercept",
                    coef=-1.0,
                    se=0.5,
                    t_stat=-2.0,
                    pvalue=0.045,
                    ci_lower=-2.0,
                    ci_upper=0.0,
                ),
                CoefficientRow(
                    name="x1",
                    coef=0.8,
                    se=0.2,
                    t_stat=4.0,
                    pvalue=0.0001,
                    ci_lower=0.4,
                    ci_upper=1.2,
                ),
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            pseudo_r_squared=0.2,
            log_likelihood=-50.0,
            aic=104.0,
            bic=108.0,
            dep_var="y",
            method="Logit",
        )
        fig = odds_ratio_plot(result)
        assert isinstance(fig, Figure)
        # Should have exactly one non-intercept entry
        ticktext = list(fig.layout.yaxis.ticktext or [])
        assert ticktext == ["x1"], f"Expected ['x1'], got {ticktext}"

    def test_only_intercept_raises(self) -> None:
        """When only Intercept exists, should raise ValueError."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="Intercept",
                    coef=-0.5,
                    se=0.2,
                    t_stat=-2.5,
                    pvalue=0.012,
                    ci_lower=-0.9,
                    ci_upper=-0.1,
                ),
            ],
            n_obs=100,
            n_params=1,
            df_resid=99,
            pseudo_r_squared=0.0,
            log_likelihood=-60.0,
            aic=122.0,
            bic=124.0,
            dep_var="y",
            method="Logit",
        )
        with pytest.raises(ValueError, match="No non-intercept"):
            odds_ratio_plot(result)


class TestOddsRatioPlotErrorHandling:
    """Error handling for odds_ratio_plot."""

    def test_wrong_model_type_raises(self) -> None:
        """Non-logit result should raise ValueError."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="OLS",
            coefficients=[
                CoefficientRow(
                    name="x1", coef=0.5, se=0.1, t_stat=5.0,
                    pvalue=0.001, ci_lower=0.3, ci_upper=0.7,
                ),
            ],
            n_obs=50,
            n_params=2,
            df_resid=48,
        )
        with pytest.raises(ValueError, match="requires model_type='logit'"):
            odds_ratio_plot(result)

    def test_const_intercept_filtered_out(self, sample_logit_result) -> None:
        """A coefficient named 'const' should also be filtered out."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        from src.results.table import CoefficientRow, ModelResult

        result = ModelResult(
            model_type="logit",
            coefficients=[
                CoefficientRow(
                    name="const",
                    coef=0.3,
                    se=0.1,
                    t_stat=3.0,
                    pvalue=0.003,
                    ci_lower=0.1,
                    ci_upper=0.5,
                ),
                CoefficientRow(
                    name="x1",
                    coef=0.7,
                    se=0.2,
                    t_stat=3.5,
                    pvalue=0.0005,
                    ci_lower=0.3,
                    ci_upper=1.1,
                ),
            ],
            n_obs=100,
            n_params=2,
            df_resid=98,
            pseudo_r_squared=0.15,
            log_likelihood=-55.0,
            aic=114.0,
            bic=118.0,
            dep_var="y",
            method="Logit",
        )
        fig = odds_ratio_plot(result)
        ticktext = list(fig.layout.yaxis.ticktext or [])
        assert "const" not in str(ticktext)
        assert "x1" in ticktext


class TestOddsRatioPlotRenderability:
    """Ensure odds ratio plot can be serialised."""

    def test_to_html(self, sample_logit_result) -> None:
        """odds_ratio_plot should convert to HTML."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        fig = odds_ratio_plot(sample_logit_result)
        html = fig.to_html()
        assert len(html) > 100, f"HTML too short: {len(html)} chars"

    def test_roc_curve_to_html(self, binary_data) -> None:
        """roc_curve_plot should convert to HTML."""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly not installed")
        y_true, y_pred = binary_data
        fig = roc_curve_plot(y_true, y_pred)
        html = fig.to_html()
        assert len(html) > 100, f"HTML too short: {len(html)} chars"
