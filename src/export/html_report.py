# encoding: utf-8
"""
Self-contained HTML report generator for regression analysis.

Produces a single-file HTML document with inline CSS and base64-embedded
images, suitable for sharing or archiving.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from jinja2 import Template

from src.results.table import ModelResult


class HtmlReportGenerator:
    """Generate a self-contained HTML regression report.

    The single public method ``generate_full_report`` accepts a data-summary
    DataFrame, a ``ModelResult``, an optional dict of chart figures, and an
    optional model-spec description, returning a complete ``<!DOCTYPE html>``
    document as a string.
    """

    _REPORT_TEMPLATE = Template(
        """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: "Helvetica Neue", Arial, "Microsoft YaHei", sans-serif;
    margin: 0; padding: 20px; color: #333; background: #f5f5f5;
  }
  .container {
    max-width: 960px; margin: 0 auto; background: #fff;
    padding: 30px 40px; border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.12);
  }
  h1 { color: #1a1a2e; font-size: 24px; border-bottom: 2px solid #1a1a2e; padding-bottom: 8px; }
  h2 { color: #16213e; font-size: 18px; margin-top: 28px; border-left: 4px solid #0f3460; padding-left: 10px; }
  h3 { color: #0f3460; font-size: 15px; margin-top: 20px; }
  .timestamp { color: #888; font-size: 13px; margin-bottom: 20px; }
  table {
    border-collapse: collapse; width: 100%; margin: 12px 0 16px;
    font-size: 13px;
  }
  th, td {
    border: 1px solid #ddd; padding: 6px 10px; text-align: right;
  }
  th { background: #e8eaf6; font-weight: 600; text-align: center; }
  td:first-child, th:first-child { text-align: left; font-weight: 500; }
  tr:nth-child(even) { background: #fafafa; }
  .fit-table th { background: #e0e0e0; }
  .chart-grid { display: flex; flex-wrap: wrap; gap: 16px; margin: 12px 0; }
  .chart-item { flex: 1 1 45%; min-width: 280px; }
  .chart-item img { width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; }
  .notes { background: #fffde7; border-left: 4px solid #f9a825; padding: 10px 14px; margin: 16px 0; font-size: 13px; }
  .footer { margin-top: 30px; padding-top: 12px; border-top: 1px solid #ddd; font-size: 12px; color: #999; text-align: center; }
  .spec-box { background: #f3f4f6; padding: 10px 14px; border-radius: 4px; font-family: "Consolas", "Courier New", monospace; font-size: 13px; margin: 8px 0; }
</style>
</head>
<body>
<div class="container">

<h1>{{ title }}</h1>
<div class="timestamp">生成时间: {{ timestamp }}</div>

<!-- 1. Model specification -->
<h2>模型设定</h2>
{% if model_spec %}
<div class="spec-box">{{ model_spec }}</div>
{% endif %}

<!-- 2. Descriptive statistics -->
<h2>描述性统计</h2>
{% if desc_stats is not none and not desc_stats.empty %}
<table>
<thead>
<tr>
  {% for col in desc_stats.columns %}
  <th>{{ col }}</th>
  {% endfor %}
</tr>
</thead>
<tbody>
{% for _, row in desc_stats.iterrows() %}
<tr>
  {% for col in desc_stats.columns %}
  <td>{{ row[col] }}</td>
  {% endfor %}
</tr>
{% endfor %}
</tbody>
</table>
{% else %}
<p style="color:#999;">无描述性统计数据。</p>
{% endif %}

<!-- 3. Regression coefficients -->
<h2>回归结果</h2>
{% if coef_table is not none and not coef_table.empty %}
<table>
<thead>
<tr>
  {% for col in coef_table.columns %}
  <th>{{ col }}</th>
  {% endfor %}
</tr>
</thead>
<tbody>
{% for _, row in coef_table.iterrows() %}
<tr>
  {% for col in coef_table.columns %}
  <td>{{ row[col] }}</td>
  {% endfor %}
</tr>
{% endfor %}
</tbody>
</table>
{% endif %}

<!-- 4. Model fit statistics -->
<h2>模型拟合统计量</h2>
{% if fit_stats %}
<table class="fit-table">
<thead><tr><th>统计量</th><th>值</th></tr></thead>
<tbody>
{% for key, val in fit_stats.items() %}
<tr><td>{{ key }}</td><td>{{ val }}</td></tr>
{% endfor %}
</tbody>
</table>
{% endif %}

<!-- 5. Diagnostic charts -->
<h2>诊断图</h2>
{% if charts %}
<div class="chart-grid">
  {% for label, b64 in charts %}
  <div class="chart-item">
    <h3>{{ label }}</h3>
    <img src="data:image/png;base64,{{ b64 }}" alt="{{ label }}">
  </div>
  {% endfor %}
</div>
{% else %}
<p style="color:#999;">无诊断图数据。</p>
{% endif %}

<!-- 6. Notes -->
<h2>注意事项</h2>
<div class="notes">
<ul>
  <li>因变量: <strong>{{ dep_var }}</strong></li>
  <li>观测数: <strong>{{ n_obs }}</strong></li>
  <li>显著性标记: *** p&lt;0.01, ** p&lt;0.05, * p&lt;0.1</li>
  <li>标准误区估计采用OLS标准误，如有异方差或自相关问题请使用稳健标准误。</li>
  <li>本报告由 Regression Analysis Tool 自动生成，仅供参考。</li>
</ul>
</div>

<div class="footer">
  Regression Analysis Tool &mdash; 自动生成报告
</div>

</div>
</body>
</html>
""".lstrip("\n")
    )

    @staticmethod
    def generate_full_report(
        data_summary: Optional[pd.DataFrame],
        model_result: ModelResult,
        charts_dict: Optional[Dict[str, Any]] = None,
        model_spec: str = "",
    ) -> str:
        """Generate a complete self-contained HTML report.

        Args:
            data_summary: DataFrame from ``descriptive_stats()``, or None.
            model_result: A ``ModelResult`` instance.
            charts_dict: Dict mapping chart labels to plotly/matplotlib
                Figure objects, or None.
            model_spec: Plain-text model specification description.

        Returns:
            A complete HTML document string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        title = "回归分析报告"
        dep_var = model_result.dep_var or "N/A"
        n_obs = str(model_result.n_obs)

        # Coefficient table
        coef_table: Optional[pd.DataFrame] = None
        try:
            coef_table = model_result.to_dataframe().reset_index()
        except Exception:
            pass

        # Model fit statistics
        fit_stats: Dict[str, str] = {}
        if model_result.r_squared is not None:
            fit_stats["R²"] = f"{model_result.r_squared:.4f}"
        if model_result.adj_r_squared is not None:
            fit_stats["调整 R²"] = f"{model_result.adj_r_squared:.4f}"
        if model_result.rmse:
            fit_stats["RMSE"] = f"{model_result.rmse:.4f}"
        if model_result.f_statistic is not None:
            fit_stats["F 统计量"] = f"{model_result.f_statistic[0]:.4f}"
            fit_stats["F 检验 p 值"] = f"{model_result.f_statistic[1]:.6e}"
        if model_result.aic:
            fit_stats["AIC"] = f"{model_result.aic:.2f}"
        if model_result.bic:
            fit_stats["BIC"] = f"{model_result.bic:.2f}"
        fit_stats["观测数"] = str(model_result.n_obs)
        fit_stats["参数数"] = str(model_result.n_params)

        # Encode charts as base64
        charts: List[tuple] = []
        if charts_dict:
            for label, fig in charts_dict.items():
                b64 = HtmlReportGenerator._figure_to_base64(fig)
                if b64:
                    charts.append((label, b64))

        return HtmlReportGenerator._REPORT_TEMPLATE.render(
            title=title,
            timestamp=timestamp,
            model_spec=model_spec,
            desc_stats=data_summary,
            coef_table=coef_table,
            fit_stats=fit_stats,
            charts=charts,
            dep_var=dep_var,
            n_obs=n_obs,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @staticmethod
    def _figure_to_base64(fig: Any) -> Optional[str]:
        """Convert a plotly or matplotlib figure to a base64 PNG string.

        Returns:
            Base64-encoded PNG string (without ``data:`` prefix), or None
            if conversion fails.
        """
        try:
            import plotly.graph_objects as go

            if isinstance(fig, go.Figure):
                buf = io.BytesIO()
                fig.write_image(buf, format="png", width=800, height=500, scale=2)
                buf.seek(0)
                return base64.b64encode(buf.read()).decode("utf-8")
        except Exception:
            pass

        try:
            import matplotlib.figure

            if isinstance(fig, matplotlib.figure.Figure):
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                return base64.b64encode(buf.read()).decode("utf-8")
        except Exception:
            pass

        return None
