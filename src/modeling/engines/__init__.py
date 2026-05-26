"""Regression engine implementations.

Each module provides run_* (fit) and extract_* (results extraction) functions
for a specific model type.  Statsmodels engines: OLS, Logit, Probit, count
(Poisson/NegBin), MixedLM.  Linearmodels engine: panel (FE/RE).
"""

from src.modeling.engines.statsmodels_count_engine import extract_count_model, run_count_model
from src.modeling.engines.statsmodels_engine import extract_statsmodels, run_ols
from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit
from src.modeling.engines.statsmodels_mixedlm_engine import (
    extract_mixedlm,
    run_and_extract_mixedlm,
    run_mixedlm,
)
from src.modeling.engines.statsmodels_panel_engine import extract_panel, run_panel
from src.modeling.engines.statsmodels_probit_engine import extract_probit, run_probit

__all__ = [
    "run_ols",
    "extract_statsmodels",
    "run_logit",
    "extract_logit",
    "run_probit",
    "extract_probit",
    "run_count_model",
    "extract_count_model",
    "run_mixedlm",
    "extract_mixedlm",
    "run_and_extract_mixedlm",
    "run_panel",
    "extract_panel",
]
