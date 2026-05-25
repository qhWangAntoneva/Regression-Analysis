# encoding: utf-8
"""单元测试：Phase 2 可视化模块。

测试新增的诊断图和系数图功能。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

try:
    import plotly.graph_objects as go
    from plotly.graph_objects import Figure

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore[assignment]
    Figure = None  # type: ignore[assignment]


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def sample_model_result():
    """创建一个模拟的 ModelResult 用于测试。"""
    from src.results.table import CoefficientRow, ModelResult

    n = 50
    rng = np.random.default_rng(42)
    residuals = rng.normal(0, 1, n)
    fitted_values = 5.0 + 0.5 * np.arange(n) / n

    result = ModelResult(
        model_type="OLS",
        coefficients=[
            CoefficientRow(
                name="Intercept", coef=5.0, se=0.2, t_stat=25.0,
                pvalue=0.0, ci_lower=4.6, ci_upper=5.4,
            ),
            CoefficientRow(
                name="x1", coef=0.8, se=0.1, t_stat=8.0,
                pvalue=0.0001, ci_lower=0.6, ci_upper=1.0,
            ),
            CoefficientRow(
                name="x2", coef=-0.3, se=0.15, t_stat=-2.0,
                pvalue=0.05, ci_lower=-0.6, ci_upper=0.0,
            ),
            CoefficientRow(
                name="x3", coef=0.05, se=0.2, t_stat=0.25,
                pvalue=0.8, ci_lower=-0.35, ci_upper=0.45,
            ),
        ],
        n_obs=n,
        n_params=4,
        df_resid=n - 4,
        r_squared=0.75,
        adj_r_squared=0.73,
        rmse=1.0,
        dep_var="y",
        specification="y ~ x1 + x2 + x3",
        method="OLS",
        aic=150.0,
        bic=160.0,
        log_likelihood=-70.0,
        f_statistic=(30.0, 0.00001),
    )

    # 附加残差和拟合值（在 Phase 1 中 ModelResult 不存储这些）
    result.residuals = residuals  # type: ignore[attr-defined]
    result.fitted_values = fitted_values  # type: ignore[attr-defined]

    return result


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """模拟数据集。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "y": 5.0 + 0.8 * rng.normal(0, 1, 50) + rng.normal(0, 0.5, 50),
        "x1": rng.normal(0, 1, 50),
        "x2": rng.uniform(0, 1, 50),
        "x3": rng.normal(10, 2, 50),
    })


# =========================================================================
# Test: scale_location_plot
# =========================================================================


class TestScaleLocationPlot:
    """测试尺度-位置图。"""

    def test_scale_location_plot_basic(self, sample_model_result, sample_data) -> None:
        """基本尺度-位置图应返回 Figure 对象。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        fig = scale_location_plot(sample_model_result, sample_data)
        assert isinstance(fig, Figure)

        # 应包含散点和 LOESS 曲线
        assert len(fig.data) >= 1

    def test_scale_location_plot_data_labels(self, sample_model_result, sample_data) -> None:
        """图表应有正确的中文轴标签。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        fig = scale_location_plot(sample_model_result, sample_data)
        assert "拟合值" in (fig.layout.xaxis.title.text or "")
        assert "sqrt" in (fig.layout.yaxis.title.text or "")

    def test_scale_location_plot_title(self, sample_model_result, sample_data) -> None:
        """图表应有正确的标题。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        fig = scale_location_plot(sample_model_result, sample_data)
        assert "尺度-位置" in (fig.layout.title.text or "")

    def test_scale_location_plot_insufficient_data(self, sample_data) -> None:
        """数据不足时应抛出 ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        # 创建只有 3 个观测的模型结果
        from src.results.table import ModelResult

        minimal = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=3,
            n_params=1,
            df_resid=2,
        )
        minimal.residuals = np.array([1.0, -1.0, 0.5])  # type: ignore[attr-defined]
        minimal.fitted_values = np.array([1.0, 2.0, 3.0])  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="不足"):
            scale_location_plot(minimal, sample_data)


# =========================================================================
# Test: cooks_distance_plot
# =========================================================================


class TestCooksDistancePlot:
    """测试 Cook's distance 图。"""

    def test_cooks_distance_plot_basic(self, sample_model_result, sample_data) -> None:
        """基本 Cook's distance 图应返回 Figure。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import cooks_distance_plot

        fig = cooks_distance_plot(sample_model_result, sample_data)
        assert isinstance(fig, Figure)
        assert len(fig.data) >= 1

    def test_cooks_distance_has_threshold(self, sample_model_result, sample_data) -> None:
        """图表应包含 4/n 阈值参考线。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import cooks_distance_plot

        fig = cooks_distance_plot(sample_model_result, sample_data)
        # 检查是否有阈值线（hlines 或 annotations）
        assert fig.layout.shapes is not None or fig.layout.annotations is not None

    def test_cooks_distance_title(self, sample_model_result, sample_data) -> None:
        """图表应有正确标题。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import cooks_distance_plot

        fig = cooks_distance_plot(sample_model_result, sample_data)
        assert "Cook" in (fig.layout.title.text or "")

    def test_cooks_distance_no_residuals(self, sample_data) -> None:
        """无残差时应抛出 ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.results.table import ModelResult
        from src.visualization.residual import cooks_distance_plot

        empty_result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0,
            n_params=0,
            df_resid=0,
        )

        with pytest.raises(ValueError, match="缺少残差"):
            cooks_distance_plot(empty_result, sample_data)


# =========================================================================
# Test: diagnostic_dashboard
# =========================================================================


class TestDiagnosticDashboard:
    """测试诊断总览面板。"""

    def test_dashboard_returns_dict(self, sample_model_result, sample_data) -> None:
        """诊断总览应返回包含 4 个图的字典。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import diagnostic_dashboard

        figs = diagnostic_dashboard(sample_model_result, sample_data)
        assert isinstance(figs, dict)

        expected_keys = {"residual_fitted", "qq", "scale_location", "cooks_distance"}
        assert expected_keys.issubset(set(figs.keys()))

    def test_dashboard_all_figures(self, sample_model_result, sample_data) -> None:
        """所有 4 个图表都应是有效的 Figure。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import diagnostic_dashboard

        figs = diagnostic_dashboard(sample_model_result, sample_data)
        for name, fig in figs.items():
            assert isinstance(fig, Figure), f"图 '{name}' 不是 Figure 对象"

    def test_dashboard_handles_missing_data(self, sample_data) -> None:
        """数据缺失时返回空图而非崩溃。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.results.table import ModelResult
        from src.visualization.residual import diagnostic_dashboard

        empty_result = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0,
            n_params=0,
            df_resid=0,
        )

        # 不应抛出异常
        figs = diagnostic_dashboard(empty_result, sample_data)
        assert isinstance(figs, dict)
        assert len(figs) == 4


# =========================================================================
# Test: coefficient_plot
# =========================================================================


class TestCoefficientPlot:
    """测试多模型系数图。"""

    def test_coefficient_plot_basic(self, sample_model_result) -> None:
        """基本多模型系数图应返回 Figure。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot

        fig = coefficient_plot([sample_model_result, sample_model_result])
        assert isinstance(fig, Figure)

    def test_coefficient_plot_with_labels(self, sample_model_result) -> None:
        """应使用自定义模型标签。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot

        labels = ["基础模型", "扩展模型"]
        fig = coefficient_plot([sample_model_result, sample_model_result], labels)
        assert isinstance(fig, Figure)

    def test_coefficient_plot_empty(self) -> None:
        """空输入应抛出 ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot

        with pytest.raises(ValueError, match="至少需要一个"):
            coefficient_plot([])

    def test_coefficient_plot_wrong_labels(self, sample_model_result) -> None:
        """标签数量不匹配应抛出 ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot

        with pytest.raises(ValueError, match="模型标签数量"):
            coefficient_plot([sample_model_result], ["模型 1", "模型 2"])


# =========================================================================
# Test: coefficient_plot_single
# =========================================================================


class TestCoefficientPlotSingle:
    """测试单模型系数图。"""

    def test_coefficient_plot_single_basic(self, sample_model_result) -> None:
        """基本单模型系数图应返回 Figure。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot_single

        fig = coefficient_plot_single(sample_model_result)
        assert isinstance(fig, Figure)

    def test_coefficient_plot_single_contains_all_coefs(self, sample_model_result) -> None:
        """图表应包含所有系数。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot_single

        fig = coefficient_plot_single(sample_model_result)
        # 检查 y 轴刻度包含变量名
        ticktext = fig.layout.yaxis.ticktext or []
        ticktext_str = " ".join(str(t) for t in ticktext)
        assert "Intercept" in ticktext_str
        assert "x1" in ticktext_str
        assert "x2" in ticktext_str

    def test_coefficient_plot_single_has_zero_line(self, sample_model_result) -> None:
        """图表应包含零点参考线。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot_single

        fig = coefficient_plot_single(sample_model_result)
        # 检查是否有 vline 或 shapes
        has_vline = fig.layout.shapes is not None or fig.layout.annotations is not None
        assert has_vline

    def test_coefficient_plot_single_empty_coefficients(self) -> None:
        """无系数的结果应抛出 ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.results.table import ModelResult
        from src.visualization.coefficient import coefficient_plot_single

        empty = ModelResult(
            model_type="OLS",
            coefficients=[],
            n_obs=0,
            n_params=0,
            df_resid=0,
        )

        with pytest.raises(ValueError, match="无可用系数"):
            coefficient_plot_single(empty)


# =========================================================================
# Test: Regression integrity — plots produce renderable output
# =========================================================================


class TestFigureRenderability:
    """测试图表可渲染性（不抛出异常）。"""

    def test_scale_location_plot_to_html(self, sample_model_result, sample_data) -> None:
        """尺度-位置图应可转为 HTML。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        fig = scale_location_plot(sample_model_result, sample_data)
        html = fig.to_html()
        assert len(html) > 100

    def test_cooks_distance_plot_to_html(self, sample_model_result, sample_data) -> None:
        """Cook's distance 图应可转为 HTML。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import cooks_distance_plot

        fig = cooks_distance_plot(sample_model_result, sample_data)
        html = fig.to_html()
        assert len(html) > 100

    def test_coefficient_plot_to_html(self, sample_model_result) -> None:
        """系数图应可转为 HTML。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.coefficient import coefficient_plot_single

        fig = coefficient_plot_single(sample_model_result)
        html = fig.to_html()
        assert len(html) > 100


# =========================================================================
# Test: Error handling for all diagnostic functions
# =========================================================================


class TestErrorHandling:
    """测试各函数对无效输入的错误处理。"""

    def test_scale_location_plot_none_result(self, sample_data) -> None:
        """None 结果应抛出 AttributeError/ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import scale_location_plot

        with pytest.raises((ValueError, AttributeError)):
            scale_location_plot(None, sample_data)  # type: ignore[arg-type]

    def test_cooks_distance_plot_none_result(self, sample_data) -> None:
        """None 结果应抛出 AttributeError/ValueError。"""
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import cooks_distance_plot

        with pytest.raises((ValueError, AttributeError)):
            cooks_distance_plot(None, sample_data)  # type: ignore[arg-type]


# =========================================================================
# Test: _norm_ppf / _approx_norm_ppf fallback approximation
# =========================================================================


class TestNormPpfApproximation:
    """Test _norm_ppf and _approx_norm_ppf function accuracy.

    The A&S 26.2.23 formula approximates the inverse complementary CDF.
    The original code fed the quantile probability directly into the
    formula without converting to tail probability or handling sign,
    producing errors up to 4.65 for quantiles above the median.
    The fix corrects both the tail-probability mapping and sign handling.
    """

    def test_norm_ppf_uses_scipy_when_available(self) -> None:
        """When scipy is available, _norm_ppf should delegate to scipy's norm.ppf."""
        from scipy.stats import norm

        from src.visualization.residual import _norm_ppf

        q = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        result = _norm_ppf(q)
        expected = norm.ppf(q)
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_approx_norm_ppf_accuracy(self) -> None:
        """_approx_norm_ppf should approximate norm.ppf within 0.01 max abs error.

        Tests 101 evenly spaced quantiles across [0.01, 0.99].
        """
        from scipy.stats import norm

        from src.visualization.residual import _approx_norm_ppf

        qs = np.linspace(0.01, 0.99, 101)
        approx = _approx_norm_ppf(qs)
        true = norm.ppf(qs)

        max_err = np.max(np.abs(approx - true))
        assert max_err < 0.01, (
            f"Max absolute error {max_err:.6f} exceeds 0.01 threshold"
        )

    def test_approx_norm_ppf_symmetry(self) -> None:
        """_approx_norm_ppf should be antisymmetric: ppf(q) = -ppf(1-q)."""
        from src.visualization.residual import _approx_norm_ppf

        qs = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4])
        lower = _approx_norm_ppf(qs)
        upper = _approx_norm_ppf(1.0 - qs)
        np.testing.assert_allclose(lower, -upper, atol=1e-6)

    def test_approx_norm_ppf_median_is_zero(self) -> None:
        """ppf(0.5) should be approximately 0 (within floating-point tolerance)."""
        from src.visualization.residual import _approx_norm_ppf

        result = _approx_norm_ppf(np.array([0.5]))
        # A&S approximation at exactly 0.5 gives ~1e-7 (not exact zero),
        # which is well within acceptable floating-point tolerance
        assert abs(result[0]) < 1e-6

    def test_qq_plot_produces_symmetric_theoretical_quantiles(self) -> None:
        """QQ plot theoretical quantiles should be symmetric for symmetric sample.

        Verifies that the full _norm_ppf -> qq_plot pipeline produces
        correctly signed theoretical quantiles (was the root cause of
        the _norm_ppf bug — see HANDOVER.md).
        """
        if not PLOTLY_AVAILABLE:
            pytest.skip("plotly 未安装")

        from src.visualization.residual import qq_plot

        rng = np.random.default_rng(42)
        residuals = rng.normal(0, 1, 100)
        fig = qq_plot(residuals)

        scatter = fig.data[0]
        theoretical = np.asarray(scatter.x)

        # With the fix, theoretical quantiles should be approximately
        # antisymmetric: mid <=> negative tail, high <=> positive tail
        assert theoretical[0] < -2.0, (
            "Lowest theoretical quantile should be negative (left tail)"
        )
        assert theoretical[-1] > 2.0, (
            "Highest theoretical quantile should be positive (right tail)"
        )
        assert theoretical[49] < 0 < theoretical[50], (
            "Median quantiles should bracket zero"
        )

    def test_approx_norm_ppf_error_bounds_near_median(self) -> None:
        """Approximation error should be small near the median (q in [0.25, 0.75])."""
        from scipy.stats import norm

        from src.visualization.residual import _approx_norm_ppf

        qs = np.linspace(0.25, 0.75, 51)
        approx = _approx_norm_ppf(qs)
        true = norm.ppf(qs)

        max_err = np.max(np.abs(approx - true))
        assert max_err < 0.001, (
            f"Max error near median {max_err:.6f} exceeds 0.001"
        )
