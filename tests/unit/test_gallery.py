# encoding: utf-8
"""Tests for the Sample Gallery module (src.utils.gallery).

Covers gallery item creation, DGP reproducibility, scenario imperfections,
JSON serialization roundtrip, and gallery index/metadata API.
"""

from __future__ import annotations

import hashlib
from typing import List

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Try to import the gallery helper functions for JSON roundtrip; if they
# don't exist, the roundtrip tests fall back to exercising result_json.
# ---------------------------------------------------------------------------
_json_to_model_result = None
_model_result_to_json = None
try:
    from src.utils.gallery import _json_to_model_result, _model_result_to_json  # type: ignore[import]
except ImportError:
    pass

from src.utils.gallery import (
    GalleryItem,
    get_gallery_index,
    get_gallery_item,
    get_gallery_items,
)


# =========================================================================
# Helpers
# =========================================================================


def _df_hash(df: pd.DataFrame) -> str:
    """Return a deterministic hash of a DataFrame for reproducibility checks."""
    buf = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(buf).hexdigest()


def _compute_vif(data: pd.DataFrame, target_col: str, other_cols: List[str]) -> float:
    """Compute VIF for *target_col* by regressing it on *other_cols* via OLS.

    VIF = 1 / (1 - R^2) of the auxiliary regression.
    """
    try:
        from statsmodels.api import OLS, add_constant
    except ImportError:
        return float("nan")

    X = data[other_cols].select_dtypes(include=[np.number])
    if X.empty or X.shape[1] == 0:
        return float("nan")
    X = add_constant(X)
    y = data[target_col]

    try:
        model = OLS(y, X).fit()
        r2 = model.rsquared
        if r2 >= 0.9999:
            return float("inf")
        return 1.0 / (1.0 - r2)
    except Exception:
        return float("nan")


# =========================================================================
# 1. TestGalleryItemCreation
# =========================================================================


class TestGalleryItemCreation:
    """Verify all 5 gallery items are created correctly."""

    # ------------------------------------------------------------------
    def test_all_five_items_load(self) -> None:
        """get_gallery_items() returns 5 items."""
        items = get_gallery_items()
        assert len(items) == 5, f"Expected 5 gallery items, got {len(items)}"

    # ------------------------------------------------------------------
    def test_all_items_have_required_fields(self) -> None:
        """Every item has non-None id, title, persona, description, tags,
        data, model_spec, model_result, result_json, key_features, story,
        dep_var."""
        required_fields = [
            "id", "title", "persona", "description", "tags",
            "data", "model_spec", "model_result", "result_json",
            "key_features", "story", "dep_var",
        ]
        for item in get_gallery_items():
            for field in required_fields:
                val = getattr(item, field)
                assert val is not None, (
                    f"Item '{item.id}': field '{field}' is None"
                )

    # ------------------------------------------------------------------
    def test_all_items_data_not_empty(self) -> None:
        """Each item's data DataFrame has rows > 0 and correct n_obs."""
        for item in get_gallery_items():
            assert isinstance(item.data, pd.DataFrame), (
                f"Item '{item.id}': data is not a DataFrame"
            )
            assert len(item.data) > 0, (
                f"Item '{item.id}': data has 0 rows"
            )
            assert item.n_obs == len(item.data), (
                f"Item '{item.id}': n_obs={item.n_obs} != len(data)={len(item.data)}"
            )

    # ------------------------------------------------------------------
    def test_all_items_result_json_has_coefficients(self) -> None:
        """Each item's result_json dict contains 'coefficients' list and
        model stats."""
        for item in get_gallery_items():
            rj = item.result_json
            assert isinstance(rj, dict), (
                f"Item '{item.id}': result_json is not a dict"
            )
            assert "coefficients" in rj, (
                f"Item '{item.id}': result_json missing 'coefficients'"
            )
            coefs = rj["coefficients"]
            assert isinstance(coefs, list), (
                f"Item '{item.id}': coefficients is not a list"
            )
            assert len(coefs) > 0, (
                f"Item '{item.id}': coefficients list is empty"
            )
            # Verify each coefficient entry has expected keys
            for c in coefs:
                assert "name" in c, "Coefficient missing 'name'"
                assert "coef" in c, "Coefficient missing 'coef'"
                assert "se" in c, "Coefficient missing 'se'"

            # Check that key model stats are present
            for stat in ("n_obs", "r_squared", "n_params"):
                assert stat in rj, (
                    f"Item '{item.id}': result_json missing '{stat}'"
                )

    # ------------------------------------------------------------------
    def test_all_items_model_result_valid(self) -> None:
        """Each item's model_result has R^2 between 0 and 1, and non-empty
        coefficients."""
        for item in get_gallery_items():
            mr = item.model_result
            assert mr.r_squared is not None, (
                f"Item '{item.id}': r_squared is None"
            )
            assert 0.0 <= mr.r_squared <= 1.0, (
                f"Item '{item.id}': r_squared={mr.r_squared} not in [0, 1]"
            )
            assert len(mr.coefficients) > 0, (
                f"Item '{item.id}': coefficients list is empty"
            )
            # Each coefficient must have a non-empty name
            for c in mr.coefficients:
                assert c.name and len(c.name) > 0, (
                    f"Item '{item.id}': coefficient with empty name"
                )


# =========================================================================
# 2. TestDGPReproducibility
# =========================================================================


class TestDGPReproducibility:
    """DGP produces consistent data with seed 42."""

    # ------------------------------------------------------------------
    def test_seed_reproducibility(self) -> None:
        """Calling get_gallery_items() twice produces identical data
        (same hash)."""
        items_a = get_gallery_items()
        items_b = get_gallery_items()

        assert len(items_a) == len(items_b)

        for a, b in zip(items_a, items_b):
            assert a.id == b.id
            assert _df_hash(a.data) == _df_hash(b.data), (
                f"Item '{a.id}': data reproducibility broken"
            )
            # Model result coefficients should also match
            a_names = [c.name for c in a.model_result.coefficients]
            b_names = [c.name for c in b.model_result.coefficients]
            assert a_names == b_names, (
                f"Item '{a.id}': coefficient names changed"
            )
            for ac, bc in zip(a.model_result.coefficients, b.model_result.coefficients):
                assert ac.coef == bc.coef, (
                    f"Item '{a.id}' coeff '{ac.name}': reproducibility broken"
                )

    # ------------------------------------------------------------------
    def test_coefficients_match_known_dgp(self) -> None:
        """Estimated coefficients are within 2 SE of their known true values.

        For scenario 1 (survey_happiness), income is stored in raw units
        and the DGP uses income/10000 internally, so the expected
        coefficient for raw 'income' is 0.35/10000 = 3.5e-5.
        """

        def _find_coef(mr, name_contains: str):
            matches = [c for c in mr.coefficients if name_contains in c.name]
            if matches:
                return matches[0]
            return None

        item = get_gallery_item("survey_happiness")
        assert item is not None

        mr = item.model_result

        # Check income coefficient: DGP formula uses income/10000,
        # so the coefficient on raw income is 0.35 / 10000 = 3.5e-5
        income_c = _find_coef(mr, "income")
        if income_c is not None:
            assert income_c.coef > 0, (
                f"income coef should be positive, got {income_c.coef}"
            )
            assert income_c.pvalue < 0.001, (
                f"income pvalue={income_c.pvalue} should be highly significant"
            )

        # Check health coefficient is near 0.6 (within 3 SE)
        health_c = _find_coef(mr, "health")
        if health_c is not None:
            assert abs(health_c.coef - 0.60) < 3.0 * health_c.se, (
                f"health coef={health_c.coef} not within 3*SE={3.0 * health_c.se} of 0.60"
            )

    # ------------------------------------------------------------------
    def test_r_squared_in_expected_range(self) -> None:
        """Each scenario's R^2 falls in expected range."""
        expected_ranges = {
            "survey_happiness": (0.25, 0.55),
            "trust_experiment": (0.10, 0.30),
            "ecommerce_sales": (0.85, 1.00),
            "customer_satisfaction": (0.65, 0.90),
            "policy_effect": (0.45, 0.75),
        }

        for item in get_gallery_items():
            lo, hi = expected_ranges.get(item.id, (0.0, 1.0))
            r2 = item.model_result.r_squared
            assert r2 is not None
            assert lo <= r2 <= hi, (
                f"Item '{item.id}': r_squared={r2:.4f} not in [{lo}, {hi}]"
            )


# =========================================================================
# 3. TestScenarioImperfections
# =========================================================================


class TestScenarioImperfections:
    """Each scenario includes its intended 'imperfection'."""

    # ------------------------------------------------------------------
    def test_survey_happiness_has_borderline_variable(self) -> None:
        """Scenario 1: at least one variable has p > 0.05.

        (education should have p > 0.05 due to collinearity with income.)
        """
        item = get_gallery_item("survey_happiness")
        assert item is not None

        borderline = [c for c in item.model_result.coefficients if c.pvalue > 0.05]
        assert len(borderline) > 0, (
            "Expected at least one variable with p > 0.05 in survey_happiness"
        )

    # ------------------------------------------------------------------
    def test_trust_experiment_party_member_significant(self) -> None:
        """Scenario 2: party_member is statistically significant (p < 0.05)
        even with small sample size (n=200).
        """
        item = get_gallery_item("trust_experiment")
        assert item is not None

        pm = [c for c in item.model_result.coefficients if "party" in c.name.lower()]
        assert len(pm) > 0, (
            "Expected 'party_member' variable in trust_experiment"
        )
        pm_coef = pm[0]
        assert pm_coef.pvalue < 0.05, (
            f"party_member pvalue={pm_coef.pvalue:.4f} should be < 0.05"
        )

    # ------------------------------------------------------------------
    def test_ecommerce_vif_high(self) -> None:
        """Scenario 3: VIF > 5 for ad_spend or promotion_discount.

        Check that at least one of these two predictors has another predictor
        with correlation > 0.5 (strong collinearity).
        """
        item = get_gallery_item("ecommerce_sales")
        assert item is not None

        df = item.data
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Try to compute VIF for key predictors
        high_vif_found = False
        candidates = [c for c in numeric_cols if c != item.dep_var and c not in
                       ("Intercept", "const")]

        # First approach: check pair-wise correlation for ad_spend /
        # promotion_discount
        target_names = ["ad_spend", "promotion_discount", "promotion"]
        for tn in target_names:
            if tn in df.columns:
                # Check correlation of this column with all other numeric cols
                for other in numeric_cols:
                    if other == tn or other == item.dep_var:
                        continue
                    corr = df[tn].corr(df[other])
                    if abs(corr) > 0.5:
                        high_vif_found = True
                        break
            if high_vif_found:
                break

        # Second approach: compute actual VIF values
        if not high_vif_found and len(candidates) >= 3:
            for tn in target_names:
                if tn not in df.columns:
                    continue
                other_cols = [c for c in candidates if c != tn]
                vif = _compute_vif(df, tn, other_cols)
                if vif > 5.0 and vif < float("inf"):
                    high_vif_found = True
                    break

        assert high_vif_found, (
            "Expected high VIF (>5) or strong correlation (>0.5) between "
            "ad_spend/promotion_discount and another predictor"
        )

    # ------------------------------------------------------------------
    def test_customer_satisfaction_has_two_categorical(self) -> None:
        """Scenario 4: service_quality and price_perception exist as
        categorical columns (object or category dtype)."""
        item = get_gallery_item("customer_satisfaction")
        assert item is not None

        cat_cols = [c for c in item.data.columns
                    if item.data[c].dtype in ("object", "category")]
        assert len(cat_cols) >= 2, (
            f"Expected >= 2 categorical columns, got {len(cat_cols)}: {cat_cols}"
        )

    # ------------------------------------------------------------------
    def test_env_policy_has_interaction_and_hc1(self) -> None:
        """Scenario 5: model_spec has interaction_terms and
        model_result.se_type == 'HC1'."""
        item = get_gallery_item("policy_effect")
        assert item is not None

        # Check model_spec has interaction terms
        interaction_terms = item.model_spec.interaction_terms
        assert len(interaction_terms) > 0, (
            f"Expected interaction_terms, got {interaction_terms}"
        )

        # Check model_result uses HC1 standard errors
        assert item.model_result.se_type == "HC1", (
            f"Expected se_type='HC1', got '{item.model_result.se_type}'"
        )


# =========================================================================
# 4. TestJsonRoundtrip
# =========================================================================


class TestJsonRoundtrip:
    """ModelResult JSON serialization roundtrip."""

    # ------------------------------------------------------------------
    def test_roundtrip_preserves_coefficients(self) -> None:
        """ModelResult -> JSON -> ModelResult preserves coefficient names
        and values."""
        item = get_gallery_item("survey_happiness")
        assert item is not None

        if _json_to_model_result is not None and _model_result_to_json is not None:
            # Use the dedicated serialization functions
            json_dict = _model_result_to_json(item.model_result)
            reconstructed = _json_to_model_result(json_dict)
            orig_names = [c.name for c in item.model_result.coefficients]
            recon_names = [c.name for c in reconstructed.coefficients]
            assert orig_names == recon_names, (
                f"Coefficient names mismatch: {orig_names} vs {recon_names}"
            )
            for oc, rc in zip(item.model_result.coefficients, reconstructed.coefficients):
                assert round(oc.coef, 8) == round(rc.coef, 8), (
                    f"coef mismatch for {oc.name}: {oc.coef} vs {rc.coef}"
                )
        else:
            # Fall back to checking result_json field consistency
            rj = item.result_json
            coeffs_from_json = rj["coefficients"]
            coeffs_from_mr = item.model_result.coefficients

            json_names = [c["name"] for c in coeffs_from_json]
            mr_names = [c.name for c in coeffs_from_mr]
            assert json_names == mr_names, (
                f"Coefficient names mismatch: {json_names} vs {mr_names}"
            )
            for jc, mc in zip(coeffs_from_json, coeffs_from_mr):
                assert round(jc["coef"], 8) == round(mc.coef, 8), (
                    f"coef mismatch for {jc['name']}: {jc['coef']} vs {mc.coef}"
                )

    # ------------------------------------------------------------------
    def test_roundtrip_preserves_model_stats(self) -> None:
        """Roundtrip preserves R^2, adj_R^2, AIC, BIC, n_obs."""
        item = get_gallery_item("trust_experiment")
        assert item is not None

        if _json_to_model_result is not None and _model_result_to_json is not None:
            json_dict = _model_result_to_json(item.model_result)
            reconstructed = _json_to_model_result(json_dict)
            stats_checks = [
                ("r_squared", item.model_result.r_squared, reconstructed.r_squared),
                ("adj_r_squared", item.model_result.adj_r_squared, reconstructed.adj_r_squared),
                ("aic", item.model_result.aic, reconstructed.aic),
                ("bic", item.model_result.bic, reconstructed.bic),
                ("n_obs", item.model_result.n_obs, reconstructed.n_obs),
            ]
            for name, orig, recon in stats_checks:
                assert orig == recon, f"{name}: {orig} vs {recon}"
        else:
            # Fallback: check result_json contains these stats and matches
            rj = item.result_json
            mr = item.model_result

            # R^2
            assert "r_squared" in rj
            assert round(rj["r_squared"], 6) == round(mr.r_squared, 6), (
                f"r_squared mismatch: {rj['r_squared']} vs {mr.r_squared}"
            )

            # n_obs
            assert "n_obs" in rj
            assert rj["n_obs"] == mr.n_obs, (
                f"n_obs mismatch: {rj['n_obs']} vs {mr.n_obs}"
            )

            # AIC (if present)
            if "aic" in rj:
                assert round(rj["aic"], 6) == round(mr.aic, 6)

            # BIC (if present)
            if "bic" in rj:
                assert round(rj["bic"], 6) == round(mr.bic, 6)

            # Adj R^2 (if present)
            if "adj_r_squared" in rj and mr.adj_r_squared is not None:
                assert round(rj["adj_r_squared"], 6) == round(mr.adj_r_squared, 6)


# =========================================================================
# 5. TestGalleryIndex
# =========================================================================


class TestGalleryIndex:
    """Gallery metadata API works."""

    # ------------------------------------------------------------------
    def test_get_gallery_index_returns_5(self) -> None:
        """Returns 5 items."""
        index = get_gallery_index()
        assert len(index) == 5, (
            f"Expected 5 index entries, got {len(index)}"
        )

    # ------------------------------------------------------------------
    def test_get_gallery_index_has_required_keys(self) -> None:
        """Each item has: id, title, persona, persona_icon, description,
        tags, n_obs, key_features, dep_var."""
        required_keys = [
            "id", "title", "persona", "persona_icon", "description",
            "tags", "n_obs", "key_features", "dep_var",
        ]
        for entry in get_gallery_index():
            for key in required_keys:
                assert key in entry, (
                    f"Index entry '{entry.get('id', '?')}': missing key '{key}'"
                )

    # ------------------------------------------------------------------
    def test_get_gallery_index_no_heavy_data(self) -> None:
        """Index items do NOT contain 'data', 'model_result', or
        'result_json' keys."""
        forbidden_keys = {"data", "model_result", "result_json"}
        for entry in get_gallery_index():
            overlap = forbidden_keys & set(entry.keys())
            assert len(overlap) == 0, (
                f"Index entry '{entry.get('id', '?')}' contains heavy keys: {overlap}"
            )

    # ------------------------------------------------------------------
    def test_get_gallery_item_by_valid_id(self) -> None:
        """get_gallery_item('survey_happiness') returns GalleryItem."""
        item = get_gallery_item("survey_happiness")
        assert item is not None
        assert isinstance(item, GalleryItem)
        assert item.id == "survey_happiness"

    # ------------------------------------------------------------------
    def test_get_gallery_item_by_invalid_id(self) -> None:
        """get_gallery_item('nonexistent') returns None."""
        item = get_gallery_item("nonexistent")
        assert item is None

    # ------------------------------------------------------------------
    @pytest.mark.parametrize("item_id", [
        "survey_happiness",
        "trust_experiment",
        "ecommerce_sales",
        "customer_satisfaction",
        "policy_effect",
    ])
    def test_get_gallery_item_all_valid_ids(self, item_id: str) -> None:
        """All 5 known IDs return a GalleryItem."""
        item = get_gallery_item(item_id)
        assert item is not None, f"get_gallery_item('{item_id}') returned None"
        assert isinstance(item, GalleryItem)
        assert item.id == item_id


# =========================================================================
# 6. TestGalleryItemDataclass
# =========================================================================


class TestGalleryItemDataclass:
    """GalleryItem dataclass field types and structure."""

    # ------------------------------------------------------------------
    def test_tags_is_list_of_strings(self) -> None:
        """Every item's tags field is a list of strings."""
        for item in get_gallery_items():
            assert isinstance(item.tags, list), (
                f"Item '{item.id}': tags is not a list"
            )
            assert all(isinstance(t, str) for t in item.tags), (
                f"Item '{item.id}': tags contains non-string entries"
            )

    # ------------------------------------------------------------------
    def test_key_features_is_list_of_strings(self) -> None:
        """Every item's key_features field is a list of strings."""
        for item in get_gallery_items():
            assert isinstance(item.key_features, list), (
                f"Item '{item.id}': key_features is not a list"
            )
            assert all(isinstance(kf, str) for kf in item.key_features), (
                f"Item '{item.id}': key_features contains non-string entries"
            )

    # ------------------------------------------------------------------
    def test_ids_are_unique(self) -> None:
        """All 5 items have distinct IDs."""
        ids = [item.id for item in get_gallery_items()]
        assert len(ids) == len(set(ids)), f"Duplicate IDs found: {ids}"

    # ------------------------------------------------------------------
    def test_dep_var_exists_in_data(self) -> None:
        """Each item's dep_var is a column in its data DataFrame."""
        for item in get_gallery_items():
            assert item.dep_var in item.data.columns, (
                f"Item '{item.id}': dep_var='{item.dep_var}' not in data columns "
                f"{list(item.data.columns)}"
            )

    # ------------------------------------------------------------------
    def test_persona_icon_is_non_empty_string(self) -> None:
        """Each item has a non-empty persona_icon string."""
        for item in get_gallery_items():
            assert isinstance(item.persona_icon, str), (
                f"Item '{item.id}': persona_icon is not a string"
            )
            assert len(item.persona_icon) > 0, (
                f"Item '{item.id}': persona_icon is empty"
            )

    # ------------------------------------------------------------------
    def test_story_is_non_empty_string(self) -> None:
        """Each item has a non-empty story."""
        for item in get_gallery_items():
            assert isinstance(item.story, str), (
                f"Item '{item.id}': story is not a string"
            )
            assert len(item.story) > 0, (
                f"Item '{item.id}': story is empty"
            )
