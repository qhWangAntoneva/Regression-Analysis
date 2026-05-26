"""Performance benchmark suite for Regression Analysis v1.0.

Times OLS fit and other operations across increasing data sizes.
Prints a formatted results table to stdout.

Target (Phase 4.3): 100K rows x 20 variables OLS fit <= 3 seconds.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

ROW_SIZES = [1_000, 10_000, 50_000, 100_000]
VAR_COUNTS = [10, 20]
SEED = 42
MAX_VIZ = 5_000  # cap data for plotly performance

warnings.filterwarnings("ignore")


def generate_dataset(n_rows: int, n_vars: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic data: y, x1..xN, cat."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for i in range(1, n_vars + 1):
        data[f"x{i}"] = rng.normal(loc=float(i * 10), scale=5.0, size=n_rows)
    data["cat"] = rng.choice(["A", "B", "C"], size=n_rows, p=[0.4, 0.35, 0.25])
    coeffs = np.linspace(0.5, 1.5, min(5, n_vars))
    y = np.ones(n_rows) * 5.0
    for idx, coef in enumerate(coeffs, start=1):
        y += coef * data[f"x{idx}"]
    y += 0.5 * pd.Series(data["cat"]).map({"A": 1, "B": 0, "C": -1}).to_numpy()
    y += rng.normal(loc=0.0, scale=2.0, size=n_rows)
    data["y"] = y
    return pd.DataFrame(data)


def time_ols_fit(df: pd.DataFrame, n_vars: int) -> tuple[float, object]:
    """Time one OLS fit: design matrix + statsmodels OLS + extract.

    Attaches residuals/fitted_values to the returned ModelResult.
    """
    from statsmodels.regression.linear_model import OLS

    from src.modeling.engines.statsmodels_engine import extract_statsmodels
    from src.modeling.specification import ModelSpec, build_design_matrix

    pred_vars = [f"x{i}" for i in range(1, n_vars + 1)]
    spec = ModelSpec(dep_var="y", indep_vars=pred_vars, has_intercept=True)
    X, y_vec = build_design_matrix(spec, df)

    t0 = time.perf_counter()
    fitted = OLS(y_vec, X).fit()
    result = extract_statsmodels(
        fitted, model_type="OLS", dep_var="y",
        specification=f"y ~ {' + '.join(pred_vars)}",
    )
    result.residuals = fitted.resid  # type: ignore[attr-defined]
    result.fitted_values = fitted.fittedvalues  # type: ignore[attr-defined]
    return time.perf_counter() - t0, result


def time_visualization(df: pd.DataFrame, result: object) -> dict[str, float]:
    """Time 4 chart types.  Data is capped at ``MAX_VIZ`` points."""
    from src.visualization.coefficient import coefficient_plot_single
    from src.visualization.residual import qq_plot, residual_vs_fitted_plot
    from src.visualization.scatter import scatter_with_regression

    # Cap data for all plots -- residual plot uses result.residuals/fitted_values
    # which are full-length, so we cap those too on a temporary copy.
    n_full = len(df)
    if n_full > MAX_VIZ:
        df_viz = df.sample(MAX_VIZ, random_state=42)
        # Cap residuals on a temp wrapper so residual plot is fast
        full_resid = np.asarray(result.residuals).flatten()
        full_fitted = np.asarray(result.fitted_values).flatten()
        indices = list(df_viz.index)
        result.residuals = full_resid[indices]  # type: ignore[attr-defined]
        result.fitted_values = full_fitted[indices]  # type: ignore[attr-defined]
        raw_resid = full_resid[indices]
    else:
        df_viz = df
        raw_resid = np.asarray(result.residuals).flatten()

    timings: dict[str, float] = {}

    t0 = time.perf_counter()
    _ = scatter_with_regression(df_viz, x_col="x1", y_col="y")
    timings["scatter"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = residual_vs_fitted_plot(result, df_viz)
    timings["residual"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = qq_plot(raw_resid)
    timings["qq"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = coefficient_plot_single(result)
    timings["coef_plot"] = time.perf_counter() - t0

    # Restore full residuals
    if n_full > MAX_VIZ:
        result.residuals = full_resid  # type: ignore[attr-defined]
        result.fitted_values = full_fitted  # type: ignore[attr-defined]

    return timings


def format_table(results: list[tuple[int, int, dict[str, float]]]) -> str:
    sep = "=" * 105
    header = (f"{'Size':>8}  {'Vars':>4}  {'Generate':>10}  {'OLS Fit':>10}  "
              f"{'Scatter':>10}  {'Resid':>10}  {'QQ':>10}  {'Coef':>10}  {'Total':>10}")
    lines = [sep, "Regression Analysis v1.0 -- Performance Benchmark", sep,
             header, "-" * 105]

    for n_rows, n_vars, m in results:
        total = m["generate"] + m["ols_fit"] + m["scatter"] + m["residual"] + m["qq"] + m["coef_plot"]
        lines.append(
            f"{n_rows:>8,}  {n_vars:>4}  "
            f"{m['generate']:>10.4f}  {m['ols_fit']:>10.4f}  "
            f"{m['scatter']:>10.4f}  {m['residual']:>10.4f}  "
            f"{m['qq']:>10.4f}  {m['coef_plot']:>10.4f}  {total:>10.4f}"
        )

    lines.append("-" * 105)
    lines.append("All times in seconds.  Data capped at 5K points for plotly performance.")
    lines.append("Target: 100K rows x 20 vars OLS fit <= 3.0 s")

    target = [(r, v, m) for r, v, m in results if r == 100_000 and v == 20]
    if target:
        ols = target[0][2]["ols_fit"]
        lines.append(f"\nTarget check (100Kx20 OLS): {ols:.4f}s -- "
                     f"{'PASS' if ols <= 3.0 else 'FAIL'}")

    return "\n".join(lines)


def main() -> None:
    print("Warming up...", end=" ", flush=True)
    df_w = generate_dataset(100, 5, seed=0)
    time_ols_fit(df_w, 5)
    print("done.\n", flush=True)

    all_results: list[tuple[int, int, dict[str, float]]] = []

    for n_vars in VAR_COUNTS:
        for n_rows in ROW_SIZES:
            label = f"{n_rows:,} x {n_vars}"
            print(f"Benchmarking {label}...", end=" ", flush=True)

            # 1. Generate
            t0 = time.perf_counter()
            df = generate_dataset(n_rows, n_vars, seed=SEED)
            t_gen = time.perf_counter() - t0

            # 2. OLS fit
            t_ols, result = time_ols_fit(df, n_vars)

            # 3. Visualization
            vis = time_visualization(df, result)

            metrics = {"generate": t_gen, "ols_fit": t_ols, **vis}
            all_results.append((n_rows, n_vars, metrics))
            print(f"OLS: {t_ols:.4f}s", flush=True)

    print("\n" + format_table(all_results))


if __name__ == "__main__":
    main()
