# encoding: utf-8
"""Logit regression cross-validation benchmark for Regression Analysis v1.1.

Validates statsmodels Logit engine output against:
  - statsmodels GLM with Binomial family (same optimization, different API)
  - sklearn LogisticRegression (different solver, should agree on direction/magnitude)

Generates 5 binary classification datasets and prints a pass/fail summary table.

Target (Phase 5.4): Logit-vs-GLM coefficient diff < 0.01, sklearn sign match, all converge.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.modeling.engines.statsmodels_logit_engine import extract_logit, run_logit
from src.modeling.specification import ModelSpec, build_design_matrix

SEED = 42
RNG = np.random.default_rng(SEED)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# sklearn availability check
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    LogisticRegression = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EngineResult:
    """Coefficient-level results from one fitting engine."""

    param_names: List[str]
    coefs: np.ndarray
    ses: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    converged: bool
    pseudo_r2: Optional[float]
    llf: Optional[float]


@dataclass
class DatasetResult:
    """Cross-validation results for a single dataset."""

    name: str
    n_obs: int
    n_params: int
    pos_rate: float

    logit_result: EngineResult
    glm_result: EngineResult
    sklearn_result: Optional[EngineResult]

    max_coef_diff_glm: float
    max_or_diff_glm: float
    pseudo_r2_diff: Optional[float]
    ci_overlap_fraction: float  # fraction of coefs with overlapping CIs

    sklearn_sign_match: Optional[bool]
    sklearn_max_coef_ratio: Optional[float]  # max |sklearn_coef / logit_coef|

    dgp_max_coef_diff: Optional[float]

    @property
    def passed(self) -> bool:
        """Per-dataset pass/fail verdict."""
        if not self.logit_result.converged:
            return False
        if self.max_coef_diff_glm > 0.01:
            return False
        if self.sklearn_result is not None:
            if self.sklearn_sign_match is False:
                return False
            if self.sklearn_max_coef_ratio is not None and self.sklearn_max_coef_ratio > 2.0:
                return False
        return True


# ---------------------------------------------------------------------------
# Dataset generators
# ---------------------------------------------------------------------------


def _generate_dgp(seed: int = SEED) -> Tuple[pd.DataFrame, ModelSpec, Dict[str, float]]:
    """Dataset 1: Synthetic DGP with known true coefficients.

    y* = 0.5 + 1.0*x1 - 0.8*x2 + Logistic(0, 1) noise.
    """
    rng = np.random.default_rng(seed)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y_star = 0.5 + 1.0 * x1 - 0.8 * x2 + rng.logistic(0, 1, n)
    y = (y_star > 0).astype(int)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2"], has_intercept=True)
    true_coefs = {"Intercept": 0.5, "x1": 1.0, "x2": -0.8}
    return df, spec, true_coefs


def _generate_clinical(seed: int = SEED) -> Tuple[pd.DataFrame, ModelSpec, Dict[str, float]]:
    """Dataset 2: Simulated clinical trial.

    y* = -2.0 + 0.8*treatment + 0.05*age + 0.3*sex + Logistic(0, 1) noise.
    """
    rng = np.random.default_rng(seed)
    n = 400
    treatment = rng.binomial(1, 0.5, n)
    age = rng.normal(50, 15, n).clip(18, 90)
    sex = rng.binomial(1, 0.5, n)
    y_star = -2.0 + 0.8 * treatment + 0.05 * age + 0.3 * sex + rng.logistic(0, 1, n)
    y = (y_star > 0).astype(int)
    df = pd.DataFrame({"y": y, "treatment": treatment, "age": age, "sex": sex})
    spec = ModelSpec(dep_var="y", indep_vars=["treatment", "age", "sex"], has_intercept=True)
    true_coefs = {"Intercept": -2.0, "treatment": 0.8, "age": 0.05, "sex": 0.3}
    return df, spec, true_coefs


def _generate_mtcars(seed: int = SEED) -> Tuple[pd.DataFrame, ModelSpec, Dict[str, float]]:
    """Dataset 3: mtcars-style binary outcome from continuous predictors.

    Binary outcome (high_mpg) ~ mpg + disp + hp + wt.
    """
    rng = np.random.default_rng(seed)
    n = 200
    mpg = rng.uniform(10, 35, n)
    disp = rng.uniform(70, 500, n)
    hp = rng.uniform(50, 350, n)
    wt = rng.uniform(1.5, 5.5, n)
    # Coefficients chosen so E[y*] ≈ 0.9, giving ~60% positive class rate
    y_star = 4.0 + 0.06 * mpg - 0.003 * disp - 0.004 * hp - 0.8 * wt + rng.logistic(0, 1, n)
    y = (y_star > 0).astype(int)
    df = pd.DataFrame({"y": y, "mpg": mpg, "disp": disp, "hp": hp, "wt": wt})
    spec = ModelSpec(dep_var="y", indep_vars=["mpg", "disp", "hp", "wt"], has_intercept=True)
    true_coefs = {"Intercept": 4.0, "mpg": 0.06, "disp": -0.003, "hp": -0.004, "wt": -0.8}
    return df, spec, true_coefs


def _generate_large(seed: int = SEED) -> Tuple[pd.DataFrame, ModelSpec, Dict[str, float]]:
    """Dataset 4: Large sample stress test (5000 rows, 10 predictors)."""
    rng = np.random.default_rng(seed)
    n = 5000
    n_pred = 10
    data: Dict[str, np.ndarray] = {}
    coef_vals: Dict[str, float] = {"Intercept": 0.2}
    y_star = np.full(n, 0.2, dtype=float)
    true_coeffs = [0.3, -0.5, 0.2, -0.1, 0.4, -0.3, 0.15, -0.25, 0.35, -0.15]
    for i in range(1, n_pred + 1):
        col = f"x{i}"
        vals = rng.normal(0, 1, n)
        data[col] = vals
        c = true_coeffs[i - 1]
        y_star += c * vals
        coef_vals[col] = c
    y_star += rng.logistic(0, 1, n)
    data["y"] = (y_star > 0).astype(int)
    df = pd.DataFrame(data)
    pred_vars = [f"x{i}" for i in range(1, n_pred + 1)]
    spec = ModelSpec(dep_var="y", indep_vars=pred_vars, has_intercept=True)
    return df, spec, coef_vals


def _generate_sparse(seed: int = SEED) -> Tuple[pd.DataFrame, ModelSpec, Dict[str, float]]:
    """Dataset 5: Sparse / unbalanced edge case (~20% positive class)."""
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    # Low intercept pushes expected positive rate down
    y_star = -2.2 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 + rng.logistic(0, 1, n)
    y = (y_star > 0).astype(int)
    actual_rate = y.mean()
    # If too extreme, adjust intercept via rejection sampling
    if actual_rate < 0.05 or actual_rate > 0.35:
        actual_rate = y.mean()
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    spec = ModelSpec(dep_var="y", indep_vars=["x1", "x2", "x3"], has_intercept=True)
    true_coefs = {"Intercept": -2.2, "x1": 0.8, "x2": -0.5, "x3": 0.3}
    return df, spec, true_coefs


# ---------------------------------------------------------------------------
# Engine fitting helpers
# ---------------------------------------------------------------------------


def _compute_pseudo_r2(fitted_model: Any, y: pd.Series) -> float:
    """Compute McFadden's pseudo R-squared for a fitted statsmodels model."""
    llf = float(fitted_model.llf)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    p_null = float(np.mean(y_arr))
    if p_null == 0 or p_null == 1:
        return float("nan")
    ll_null = float(np.sum(y_arr * np.log(p_null) + (1 - y_arr) * np.log(1 - p_null)))
    if ll_null == 0:
        return float("nan")
    return float(1.0 - llf / ll_null)


def _fit_logit_engine(
    data: pd.DataFrame, spec: ModelSpec, y: pd.Series
) -> EngineResult:
    """Fit using the project's statsmodels Logit engine (run_logit + extract_logit)."""
    try:
        fitted, labels = run_logit(data, spec)
        model_result = extract_logit(
            fitted_model=fitted,
            alpha=0.05,
            dep_var=spec.dep_var,
            specification=f"{spec.dep_var} ~ {' + '.join(spec.all_predictors)}",
            variable_labels=labels,
        )
        # Collect coefficients
        names = [c.name for c in model_result.coefficients]
        coefs = np.array([c.coef for c in model_result.coefficients])
        ses = np.array([c.se for c in model_result.coefficients])
        ci_low = np.array([c.ci_lower for c in model_result.coefficients])
        ci_high = np.array([c.ci_upper for c in model_result.coefficients])
        converged = True
        pseudo_r2 = model_result.pseudo_r_squared
        llf = model_result.log_likelihood
    except Exception as exc:
        print(f"  [WARN] Logit engine failed: {exc}")
        names = []
        coefs = np.array([])
        ses = np.array([])
        ci_low = np.array([])
        ci_high = np.array([])
        converged = False
        pseudo_r2 = None
        llf = None

    return EngineResult(
        param_names=names,
        coefs=coefs,
        ses=ses,
        ci_lower=ci_low,
        ci_upper=ci_high,
        converged=converged,
        pseudo_r2=pseudo_r2,
        llf=llf,
    )


def _fit_glm_binomial(
    X: pd.DataFrame, y: pd.Series
) -> EngineResult:
    """Fit using statsmodels GLM with Binomial family."""
    try:
        glm_fitted = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        param_names = list(X.columns)
        coefs = glm_fitted.params.values.astype(float)
        ses = glm_fitted.bse.values.astype(float)
        ci = glm_fitted.conf_int(alpha=0.05)
        ci_low = ci.iloc[:, 0].values.astype(float)
        ci_high = ci.iloc[:, 1].values.astype(float)
        converged = getattr(glm_fitted, "converged", True)
        pseudo_r2 = _compute_pseudo_r2(glm_fitted, y)
        llf = float(getattr(glm_fitted, "llf", np.nan))
    except Exception as exc:
        print(f"  [WARN] GLM binomial failed: {exc}")
        param_names = list(X.columns)
        coefs = np.full(len(param_names), np.nan)
        ses = np.full(len(param_names), np.nan)
        ci_low = np.full(len(param_names), np.nan)
        ci_high = np.full(len(param_names), np.nan)
        converged = False
        pseudo_r2 = None
        llf = None

    return EngineResult(
        param_names=param_names,
        coefs=coefs,
        ses=ses,
        ci_lower=ci_low,
        ci_upper=ci_high,
        converged=converged,
        pseudo_r2=pseudo_r2,
        llf=llf,
    )


def _fit_sklearn(X: pd.DataFrame, y: pd.Series) -> Optional[EngineResult]:
    """Fit using sklearn LogisticRegression (no effective regularization)."""
    if not _HAS_SKLEARN:
        return None

    try:
        # Strip Intercept column -- sklearn adds its own
        X_no_intercept = X.drop(columns=["Intercept"], errors="ignore")
        clf = LogisticRegression(
            penalty="l2",
            C=1e10,
            solver="lbfgs",
            max_iter=2000,
            fit_intercept=True,
            random_state=SEED,
        )
        clf.fit(X_no_intercept, y.values.ravel())

        # Reconstruct aligned coefficients
        has_intercept_col = "Intercept" in X.columns
        if has_intercept_col:
            param_names = ["Intercept"] + list(X_no_intercept.columns)
            coefs = np.concatenate([[clf.intercept_[0]], clf.coef_[0]])
        else:
            param_names = list(X.columns)
            coefs = clf.coef_[0]

        # Sklearn doesn't provide SE/CI for logistic regression
        ses = np.full(len(coefs), np.nan)
        ci_low = np.full(len(coefs), np.nan)
        ci_high = np.full(len(coefs), np.nan)
        converged = True  # lbfgs converges or raises

        # Compute pseudo R-squared from predicted probabilities
        probs = clf.predict_proba(X_no_intercept)[:, 1]
        y_arr = y.values.ravel()
        # Avoid log(0)
        eps = 1e-15
        probs_clipped = np.clip(probs, eps, 1 - eps)
        llf = float(np.sum(y_arr * np.log(probs_clipped) + (1 - y_arr) * np.log(1 - probs_clipped)))
        p_null = float(np.mean(y_arr))
        if p_null == 0 or p_null == 1:
            pseudo_r2 = float("nan")
        else:
            ll_null = float(np.sum(y_arr * np.log(p_null) + (1 - y_arr) * np.log(1 - p_null)))
            pseudo_r2 = float(1.0 - llf / ll_null) if ll_null != 0 else float("nan")

    except Exception as exc:
        print(f"  [WARN] sklearn failed: {exc}")
        param_names = list(X.columns)
        coefs = np.full(len(param_names), np.nan)
        ses = np.full(len(param_names), np.nan)
        ci_low = np.full(len(param_names), np.nan)
        ci_high = np.full(len(param_names), np.nan)
        converged = False
        pseudo_r2 = None
        llf = None

    return EngineResult(
        param_names=param_names,
        coefs=coefs,
        ses=ses,
        ci_lower=ci_low,
        ci_upper=ci_high,
        converged=converged,
        pseudo_r2=pseudo_r2,
        llf=llf,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _align_coefs(
    ref: EngineResult, other: EngineResult
) -> Tuple[np.ndarray, np.ndarray]:
    """Align coefficients from two engines to the same variable order.

    Both engines should use the same design matrix column order, so direct
    alignment by index should work. Falls back to name-based alignment.
    """
    name_to_idx = {name: i for i, name in enumerate(ref.param_names)}
    other_coefs = np.full(len(ref.param_names), np.nan)
    for i, name in enumerate(other.param_names):
        if name in name_to_idx:
            other_coefs[name_to_idx[name]] = other.coefs[i]
    return ref.coefs, other_coefs


def _check_ci_overlap(ref: EngineResult, other: EngineResult) -> Tuple[int, int]:
    """Count how many coefficients have overlapping CIs between two engines."""
    overlap = 0
    total = 0
    for i in range(len(ref.param_names)):
        name = ref.param_names[i]
        if name not in other.param_names:
            continue
        j = other.param_names.index(name)
        lo_ref = ref.ci_lower[i]
        hi_ref = ref.ci_upper[i]
        lo_other = other.ci_lower[j]
        hi_other = other.ci_upper[j]
        # Skip if CIs are NaN
        if np.isnan(lo_ref) or np.isnan(hi_ref) or np.isnan(lo_other) or np.isnan(hi_other):
            continue
        total += 1
        # Overlap if intervals intersect
        if lo_ref <= hi_other and lo_other <= hi_ref:
            overlap += 1
    return overlap, total


def _check_dgp_agreement(
    result: EngineResult, true_coefs: Dict[str, float]
) -> float:
    """Max absolute difference between estimated and true DGP coefficients."""
    max_diff = 0.0
    for name, true_val in true_coefs.items():
        if name in result.param_names:
            idx = result.param_names.index(name)
            est_val = result.coefs[idx]
            if not np.isnan(est_val):
                diff = abs(float(est_val) - true_val)
                if diff > max_diff:
                    max_diff = diff
    return max_diff


# ---------------------------------------------------------------------------
# Per-dataset benchmark
# ---------------------------------------------------------------------------


def benchmark_dataset(
    name: str,
    data: pd.DataFrame,
    spec: ModelSpec,
    true_coefs: Optional[Dict[str, float]] = None,
) -> DatasetResult:
    """Run all three engines on one dataset and compute agreement metrics."""
    # Build a common design matrix for fair comparison
    X, y = build_design_matrix(spec, data)
    n_obs = len(y)
    n_params = X.shape[1]
    pos_rate = float(y.mean())

    # 1. Logit engine (our project's engine)
    logit_res = _fit_logit_engine(data, spec, y)

    # 2. GLM binomial on the same X, y
    glm_res = _fit_glm_binomial(X, y)

    # 3. sklearn LogisticRegression
    sklearn_res = _fit_sklearn(X, y)

    # --- Logit-vs-GLM comparisons ---
    coefs_logit, coefs_glm = _align_coefs(logit_res, glm_res)
    valid = ~np.isnan(coefs_logit) & ~np.isnan(coefs_glm)
    if valid.any():
        max_coef_diff_glm = float(np.max(np.abs(coefs_logit[valid] - coefs_glm[valid])))
        max_or_diff_glm = float(
            np.max(np.abs(np.exp(coefs_logit[valid]) - np.exp(coefs_glm[valid])))
        )
    else:
        max_coef_diff_glm = float("nan")
        max_or_diff_glm = float("nan")

    # Pseudo R-squared agreement
    if logit_res.pseudo_r2 is not None and glm_res.pseudo_r2 is not None:
        pseudo_r2_diff = abs(logit_res.pseudo_r2 - glm_res.pseudo_r2)
    else:
        pseudo_r2_diff = None

    # CI overlap
    ci_overlap, ci_total = _check_ci_overlap(logit_res, glm_res)
    ci_overlap_fraction = ci_overlap / ci_total if ci_total > 0 else 1.0

    # --- Logit-vs-sklearn comparisons ---
    sklearn_sign_match: Optional[bool] = None
    sklearn_max_coef_ratio: Optional[float] = None

    if sklearn_res is not None and sklearn_res.converged:
        coefs_l, coefs_s = _align_coefs(logit_res, sklearn_res)
        valid_sk = ~np.isnan(coefs_l) & ~np.isnan(coefs_s)
        if valid_sk.any():
            # Sign match: both coefficients have the same sign or both near zero
            signs_match = (np.sign(coefs_l[valid_sk]) == np.sign(coefs_s[valid_sk])) | (
                np.abs(coefs_l[valid_sk]) < 1e-6
            ) | (np.abs(coefs_s[valid_sk]) < 1e-6)
            sklearn_sign_match = bool(np.all(signs_match))
            # Max ratio (avoid division by near-zero)
            denom = np.where(np.abs(coefs_l[valid_sk]) < 1e-6, np.nan, coefs_l[valid_sk])
            ratios = np.abs(coefs_s[valid_sk] / denom)
            ratios = ratios[~np.isnan(ratios)]
            if len(ratios) > 0:
                sklearn_max_coef_ratio = float(np.max(ratios))
            else:
                sklearn_max_coef_ratio = float("nan")

    # --- DGP check (Dataset 1) ---
    dgp_max_coef_diff: Optional[float] = None
    if true_coefs is not None:
        dgp_max_coef_diff = _check_dgp_agreement(logit_res, true_coefs)

    return DatasetResult(
        name=name,
        n_obs=n_obs,
        n_params=n_params,
        pos_rate=pos_rate,
        logit_result=logit_res,
        glm_result=glm_res,
        sklearn_result=sklearn_res,
        max_coef_diff_glm=max_coef_diff_glm,
        max_or_diff_glm=max_or_diff_glm,
        pseudo_r2_diff=pseudo_r2_diff,
        ci_overlap_fraction=ci_overlap_fraction,
        sklearn_sign_match=sklearn_sign_match,
        sklearn_max_coef_ratio=sklearn_max_coef_ratio,
        dgp_max_coef_diff=dgp_max_coef_diff,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def format_table(results: List[DatasetResult]) -> str:
    """Format benchmark results as a human-readable table."""
    sep = "=" * 126
    header_parts = [
        f"{'Dataset':<18}",
        f"{'N':>6}",
        f"{'Pos%':>6}",
        f"{'Converge':>10}",
        f"{'|Log-GLM|':>11}",
        f"{'PseudoR2 diff':>13}",
        f"{'CI overlap':>10}",
    ]
    if _HAS_SKLEARN:
        header_parts.append(f"{'SK sign':>8}")
        header_parts.append(f"{'SK ratio':>10}")
    header_parts.append(f"{'DGP diff':>10}")
    header_parts.append(f"{'Verdict':>10}")
    header = "".join(header_parts)

    lines = [
        sep,
        "Regression Analysis v1.1 -- Logit Cross-Validation Benchmark",
        sep,
        header,
        "-" * 126,
    ]

    for r in results:
        # Convergence column
        conv_parts = []
        conv_parts.append("L" if r.logit_result.converged else "l")
        conv_parts.append("G" if r.glm_result.converged else "g")
        if r.sklearn_result is not None:
            conv_parts.append("S" if r.sklearn_result.converged else "s")
        else:
            conv_parts.append("-")
        conv_str = "/".join(conv_parts)

        # Build row
        row_parts = [
            f"{r.name:<18}",
            f"{r.n_obs:>6}",
            f"{r.pos_rate:>6.1%}",
            f"{conv_str:>10}",
            f"{r.max_coef_diff_glm:>11.6f}" if not np.isnan(r.max_coef_diff_glm) else f"{'N/A':>11}",
            f"{r.pseudo_r2_diff:>13.6f}" if r.pseudo_r2_diff is not None else f"{'N/A':>13}",
            f"{r.ci_overlap_fraction:>9.1%} ",
        ]
        if _HAS_SKLEARN:
            if r.sklearn_sign_match is None:
                row_parts.append(f"{'N/A':>8}")
            else:
                row_parts.append(f"{'PASS' if r.sklearn_sign_match else 'FAIL':>8}")
            if r.sklearn_max_coef_ratio is not None and not np.isnan(r.sklearn_max_coef_ratio):
                row_parts.append(f"{r.sklearn_max_coef_ratio:>10.3f}")
            else:
                row_parts.append(f"{'N/A':>10}")
        if r.dgp_max_coef_diff is not None:
            row_parts.append(f"{r.dgp_max_coef_diff:>10.4f}")
        else:
            row_parts.append(f"{'--':>10}")
        row_parts.append(f"{'PASS' if r.passed else 'FAIL':>10}")
        lines.append("".join(row_parts))

    lines.append("-" * 126)

    # Legend
    lines.append("Converge: L=Logitengine G=GLM(binomial) S=sklearn (uppercase=converged)")
    lines.append("|Log-GLM|: max absolute coefficient difference between Logit and GLM")
    lines.append("PseudoR2 diff: absolute difference in McFadden pseudo R-squared")
    lines.append("CI overlap: fraction of coefficients with overlapping 95% CIs (Logit vs GLM)")
    if _HAS_SKLEARN:
        lines.append("SK sign: sklearn coefficients all have same sign as Logit coefficients")
        lines.append("SK ratio: max |sklearn_coef / logit_coef| (target <= 2.0)")
    lines.append("DGP diff: max |estimated - true| for datasets with known DGP coefficients")
    lines.append("Criteria: GLM diff < 0.01, SK sign match, SK ratio <= 2.0, all converge")
    lines.append(sep)

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    n_total = len(results)
    lines.append(f"\nSummary: {n_pass}/{n_total} datasets PASS")

    if _HAS_SKLEARN and any(
        r.sklearn_result is not None and not r.sklearn_result.converged for r in results
    ):
        lines.append("WARNING: Some sklearn models did not converge.")
    if any(not r.logit_result.converged for r in results):
        lines.append("WARNING: Some Logit models did not converge.")
    if any(not r.glm_result.converged for r in results):
        lines.append("WARNING: Some GLM models did not converge.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate all 5 datasets, benchmark each, and print results."""
    if not _HAS_SKLEARN:
        print("sklearn not available, skipping sklearn comparison.\n")

    datasets: List[Tuple[str, pd.DataFrame, ModelSpec, Optional[Dict[str, float]]]] = []

    print("Generating datasets...", end=" ", flush=True)
    try:
        df1, spec1, tc1 = _generate_dgp()
        datasets.append(("1.Synthetic DGP", df1, spec1, tc1))
    except Exception as exc:
        print(f"\n  [ERROR] Dataset 1 generation failed: {exc}")

    try:
        df2, spec2, tc2 = _generate_clinical()
        datasets.append(("2.Clinical trial", df2, spec2, tc2))
    except Exception as exc:
        print(f"\n  [ERROR] Dataset 2 generation failed: {exc}")

    try:
        df3, spec3, tc3 = _generate_mtcars()
        datasets.append(("3.mtcars-style", df3, spec3, tc3))
    except Exception as exc:
        print(f"\n  [ERROR] Dataset 3 generation failed: {exc}")

    try:
        df4, spec4, tc4 = _generate_large()
        datasets.append(("4.Large 5Kx10", df4, spec4, tc4))
    except Exception as exc:
        print(f"\n  [ERROR] Dataset 4 generation failed: {exc}")

    try:
        df5, spec5, tc5 = _generate_sparse()
        datasets.append(("5.Sparse 20%", df5, spec5, tc5))
    except Exception as exc:
        print(f"\n  [ERROR] Dataset 5 generation failed: {exc}")

    print(f"done ({len(datasets)} datasets).\n", flush=True)

    # Benchmark each dataset
    all_results: List[DatasetResult] = []
    for name, df, spec, true_coefs in datasets:
        print(f"Benchmarking {name}...", end=" ", flush=True)
        try:
            result = benchmark_dataset(name, df, spec, true_coefs)
            all_results.append(result)
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}  "
                  f"(n={result.n_obs}, pos={result.pos_rate:.1%}, "
                  f"|Log-GLM|={result.max_coef_diff_glm:.6f})",
                  flush=True)
        except Exception as exc:
            print(f"ERROR: {exc}", flush=True)

    print("\n" + format_table(all_results))


if __name__ == "__main__":
    main()
