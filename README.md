# Regression Analysis

A web-based regression analysis tool supporting OLS and logistic regression. Available in two versions.

## Quick Links
- Web App: https://qhwangantoneva.github.io/regression-analysis/
- Streamlit App: `uv run streamlit run app/app.py`
- GitHub: https://github.com/qhWangAntoneva/Regression-Analysis

## Feature Comparison: Streamlit vs Web

| Feature | Streamlit | Web |
|---------|-----------|-----|
| OLS Regression | Full support | Full support |
| Logit Regression | Full support | Full support |
| File Upload (CSV/Excel) | Full support | Full support (.csv/.tsv/.xlsx/.xls) |
| Sample Gallery | 5 pre-computed scenarios | 5 pre-computed scenarios |
| Variable Transformation (log/Z/center/square) | UI available | UI available |
| Interaction Terms | UI available | UI available |
| Multi-Model Comparison | Full support (coefficient chart + comparison table) | Full support |
| Diagnostic Charts (Residuals/Q-Q/Scale-Location/Cook's) | Full support | Full support |
| VIF & Residual Diagnostics | Full support | Full support |
| Scatter Plots with Regression Line | Full support | Full support |
| ROC Curve | Full support | Full support |
| Odds Ratio Forest Plot | Full support | Full support |
| Export: CSV | Supported | Supported |
| Export: Excel | Supported | Supported (Pyodide required) |
| Export: LaTeX | Supported | Not available |
| Export: HTML Report | Supported | Not available |
| Export: Word Report | Supported | Not available |
| Export: PNG Charts | Supported | Supported |
| Analysis Reproducibility Package | Supported | Not available |
| Session Persistence | Supported | Not available |
| Offline Use | Not available | Gallery mode works offline |
| Installation Required | Yes (Python + uv) | No (browser only) |

## Development

```bash
uv run streamlit run app/app.py        # Start Streamlit app
uv run python -m pytest tests/ -v      # Run 549 tests
bash web/deploy.sh                      # Deploy web version
```
