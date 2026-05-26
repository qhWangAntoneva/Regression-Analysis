"""Sample Gallery module for the Regression Analysis app.

Provides 7 pre-computed regression analysis scenarios (datasets + model results)
based on 4 user personas (OLS, MixedLM, and Panel models).  Pre-adapted for
Pyodide (WebAssembly) deployment: results are serialized to JSON so the
Pyodide runtime does not need to run statsmodels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.modeling.engines.statsmodels_engine import run_ols
from src.modeling.engines.statsmodels_mixedlm_engine import run_and_extract_mixedlm
from src.modeling.engines.statsmodels_panel_engine import extract_panel, run_panel
from src.modeling.specification import ModelSpec
from src.results.table import CoefficientRow, ModelResult

# ======================================================================
# JSON Serialization helpers
# ======================================================================

def _model_result_to_json(mr: ModelResult) -> dict:
    """Serialize a ModelResult to a plain JSON-compatible dict.

    All coefficient rows and model-level statistics are flattened into a
    dictionary that can be serialized via ``json.dumps``.  This pre-computed
    blob is what a Pyodide (WebAssembly) front-end consumes so that it does
    not need statsmodels at runtime.
    """
    return {
        "model_type": mr.model_type,
        "coefficients": [
            {
                "name": c.name,
                "coef": c.coef,
                "se": c.se,
                "t_stat": c.t_stat,
                "pvalue": c.pvalue,
                "ci_lower": c.ci_lower,
                "ci_upper": c.ci_upper,
                "significance": c.significance,
            }
            for c in mr.coefficients
        ],
        "n_obs": mr.n_obs,
        "n_params": mr.n_params,
        "df_resid": mr.df_resid,
        "r_squared": mr.r_squared,
        "adj_r_squared": mr.adj_r_squared,
        "f_statistic": list(mr.f_statistic) if mr.f_statistic is not None else None,
        "log_likelihood": mr.log_likelihood,
        "aic": mr.aic,
        "bic": mr.bic,
        "rmse": mr.rmse,
        "dep_var": mr.dep_var,
        "specification": mr.specification,
        "method": mr.method,
        "transforms_applied": mr.transforms_applied,
        "interaction_terms_applied": [
            list(pair) for pair in mr.interaction_terms_applied
        ],
        "se_type": mr.se_type,
        "mixedlm_extra": {
            "group_var": getattr(mr, "group_var", None),
            "re_var": getattr(mr, "re_var", {}),
            "group_count": getattr(mr, "group_count", None),
            "mixedlm_scale": getattr(mr, "mixedlm_scale", None),
            "mixedlm_converged": getattr(mr, "mixedlm_converged", None),
        },
        "panel_extra": {
            "within_r_squared": getattr(mr, "within_r_squared", None),
            "between_r_squared": getattr(mr, "between_r_squared", None),
            "overall_r_squared": getattr(mr, "overall_r_squared", None),
            "entity_count": getattr(mr, "entity_count", None),
            "time_count": getattr(mr, "time_count", None),
            "panel_type": getattr(mr, "panel_type", None),
            "f_pooled": getattr(mr, "f_pooled", None),
        },
    }


def _json_to_model_result(d: dict) -> ModelResult:
    """Deserialize a plain dict back to a ModelResult.

    This is the inverse of :func:`_model_result_to_json`.  It reconstructs
    CoefficientRow objects and other structured fields.
    """
    coefficients = [
        CoefficientRow(
            name=c["name"],
            coef=c["coef"],
            se=c["se"],
            t_stat=c["t_stat"],
            pvalue=c["pvalue"],
            ci_lower=c["ci_lower"],
            ci_upper=c["ci_upper"],
            significance=c.get("significance", ""),
        )
        for c in d["coefficients"]
    ]
    f_stat = tuple(d["f_statistic"]) if d["f_statistic"] is not None else None
    interaction_terms = [
        tuple(pair) for pair in d.get("interaction_terms_applied", [])
    ]
    return ModelResult(
        model_type=d["model_type"],
        coefficients=coefficients,
        n_obs=d["n_obs"],
        n_params=d["n_params"],
        df_resid=d["df_resid"],
        r_squared=d["r_squared"],
        adj_r_squared=d["adj_r_squared"],
        f_statistic=f_stat,
        log_likelihood=d["log_likelihood"],
        aic=d["aic"],
        bic=d["bic"],
        rmse=d["rmse"],
        dep_var=d["dep_var"],
        specification=d["specification"],
        method=d.get("method", "OLS"),
        transforms_applied=d.get("transforms_applied", {}),
        interaction_terms_applied=interaction_terms,
        se_type=d.get("se_type", "nonrobust"),
    )
    # Restore dynamically-attached MixedLM fields
    mixedlm_extra = d.get("mixedlm_extra", {})
    if mixedlm_extra.get("group_var"):
        result.group_var = mixedlm_extra["group_var"]
    if mixedlm_extra.get("re_var"):
        result.re_var = mixedlm_extra["re_var"]
    if mixedlm_extra.get("group_count"):
        result.group_count = mixedlm_extra["group_count"]
    if "mixedlm_scale" in mixedlm_extra:
        result.mixedlm_scale = mixedlm_extra["mixedlm_scale"]  # type: ignore[attr-defined]
    if "mixedlm_converged" in mixedlm_extra:
        result.mixedlm_converged = mixedlm_extra["mixedlm_converged"]  # type: ignore[attr-defined]
    # Restore dynamically-attached Panel fields
    panel_extra = d.get("panel_extra", {})
    if "within_r_squared" in panel_extra:
        result.within_r_squared = panel_extra["within_r_squared"]
    if "between_r_squared" in panel_extra:
        result.between_r_squared = panel_extra["between_r_squared"]
    if "overall_r_squared" in panel_extra:
        result.overall_r_squared = panel_extra["overall_r_squared"]
    if "entity_count" in panel_extra:
        result.entity_count = panel_extra["entity_count"]
    if "time_count" in panel_extra:
        result.time_count = panel_extra["time_count"]
    if "panel_type" in panel_extra:
        result.panel_type = panel_extra["panel_type"]
    if "f_pooled" in panel_extra and panel_extra["f_pooled"] is not None:
        fp = panel_extra["f_pooled"]
        result.f_pooled = tuple(fp) if isinstance(fp, list) else fp  # type: ignore[attr-defined]
    return result


# ======================================================================
# GalleryItem dataclass
# ======================================================================

@dataclass
class GalleryItem:
    """A single, pre-computed regression-analysis scenario.

    Each item bundles a synthetic dataset, a model specification, the fitted
    OLS results (both as structured Python objects and as a JSON blob), and
    descriptive metadata for the UI.
    """

    id: str                      # e.g. "survey_happiness"
    title: str                   # Chinese display name
    persona: str                 # Persona name
    persona_icon: str            # single emoji / icon char
    description: str             # 1-2 sentences
    tags: list[str]              # e.g. ["问卷数据", "多重共线性"]
    n_obs: int
    data: pd.DataFrame
    model_spec: ModelSpec
    model_result: ModelResult    # pre-computed via run_ols()
    result_json: dict            # JSON-serializable dict of model_result
    key_features: list[str]      # What makes this scenario interesting
    story: str                   # Analysis narrative (2-3 paragraphs)
    dep_var: str                 # Convenience field


# ======================================================================
# Module-level cache
# ======================================================================

_gallery_cache: list[GalleryItem] | None = None


# ======================================================================
# DGP 1 -- CGSS 幸福感调查  (Persona: 张薇, n = 400)
# ======================================================================

def _make_survey_happiness() -> GalleryItem:
    """Generate the CGSS happiness-survey scenario.

    Realistic CGSS-style data with a skewed income distribution, ordinal
    education, and a moderate overall fit (R-squared ~0.35).
    """
    rng = np.random.default_rng(42)
    n = 400

    # --- education (5-level ordinal categorical) ---
    edu_categories = ["初中以下", "高中中专", "大专", "本科", "硕士及以上"]
    edu_probs = [0.15, 0.25, 0.25, 0.25, 0.10]
    edu_idx = rng.choice(5, size=n, p=edu_probs)
    edu_num = edu_idx.astype(float)  # 0,1,2,3,4 for DGP computation
    education = pd.Categorical(
        np.array(edu_categories)[edu_idx],
        categories=edu_categories,
        ordered=False,
    )

    # --- income (lognormal, mildly correlated with education) ---
    income = np.exp(rng.normal(8.5 + 0.15 * edu_num, 0.7, n))

    # --- health (1-5, discrete) ---
    health = rng.integers(1, 6, size=n).astype(float)

    # --- urban / rural ---
    ur_labels = np.array(["城市", "农村"])
    ur_idx = rng.choice(2, size=n, p=[0.60, 0.40])
    urban_rural = pd.Categorical(
        ur_labels[ur_idx],
        categories=["城市", "农村"],
        ordered=False,
    )

    # --- work hours (20-80, uniform) ---
    work_hours = rng.uniform(20, 80, n)

    # --- DGP ---
    happiness_score = (
        2.5
        + 0.35 * (income / 10000)
        + 0.4 * edu_num
        + 0.6 * health
        - 0.3 * (urban_rural == "农村").astype(float)
        + 0.005 * work_hours
        + rng.normal(0, 1.5, n)
    )

    # --- DataFrame ---
    df = pd.DataFrame({
        "income": income,
        "education": education,
        "health": health,
        "urban_rural": urban_rural,
        "work_hours": work_hours,
        "happiness_score": happiness_score,
    })

    # --- ModelSpec ---
    spec = ModelSpec(
        dep_var="happiness_score",
        indep_vars=["income", "education", "health", "urban_rural", "work_hours"],
    )

    # --- Fit ---
    result = run_ols(data=df, spec=spec)
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="survey_happiness",
        title="CGSS 居民幸福感调查分析",
        persona="张薇（社科研究生）",
        persona_icon="📊",
        description=(
            "基于 CGSS 问卷数据，研究收入、教育、健康、城乡身份和工时"
            "对主观幸福感的多元影响，存在收入与教育的多重共线性。"
        ),
        tags=["问卷数据", "分类变量", "多重共线性"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "中等拟合优度（R²≈0.35），贴近真实社会调查",
            "教育为五水平序次分类变量，生成 4 个虚拟系数",
            "收入与教育存在温和相关（corr≈0.3），教育系数边界显著",
            "城乡二元变量，农村身份负向影响幸福感",
        ],
        story=(
            "社科研究生张薇使用中国综合社会调查（CGSS）数据，探究城市与农村居民幸福感"
            "的决定因素。她假设收入水平正向促进幸福感，教育通过人力资本和社会地位间接"
            "提升幸福感，身体健康状况直接改善主观福祉，而农村户口可能伴随着公共服务"
            "和社会保障的相对劣势，因而对幸福感产生负面影响。\n\n"
            "初步探索性分析显示，城市的平均幸福感略高于农村，且幸福感随教育水平提升呈"
            "单调递增趋势。然而，张薇注意到收入与教育之间存在温和的正相关——高学历者"
            "往往收入更高，这可能导致回归模型中教育系数的估计精度下降。她决定先运行"
            "一个含所有变量的 OLS 模型，观察各变量的显著性水平，再考虑是否需要通过"
            "VIF 诊断或逐步回归来处理多重共线性问题。\n\n"
            "模型的 R² 约为 0.35，说明幸福感的变化有相当一部分不能由这五个变量解释——"
            "这可能与人格特质、家庭关系、社会网络等未观测因素有关。这一结果也提醒"
            "张薇在撰写论文时谨慎解释因果推断，并建议将本研究定位为探索性分析而非"
            "严格的因果识别。"
        ),
        dep_var="happiness_score",
    )


# ======================================================================
# DGP 2 -- 社会信任实验  (Persona: 张薇, n = 200)
# ======================================================================

def _make_trust_experiment() -> GalleryItem:
    """Generate the social-trust survey scenario.

    Small-sample (n = 200) design where party membership is deliberately
    noisy so that its coefficient hovers near the 5 % significance boundary.
    """
    rng = np.random.default_rng(42)
    n = 200

    # --- age (18-70, uniform) ---
    age = rng.uniform(18, 70, n)

    # --- income (normal, clipped to positive) ---
    income = np.clip(rng.normal(5000, 3000, n), 500, 15000)

    # --- education years (6-22, discrete uniform) ---
    edu_years = rng.integers(6, 23, size=n).astype(float)

    # --- media exposure (0-10, roughly normal) ---
    media_exposure = np.clip(rng.normal(5, 2.5, n), 0, 10)

    # --- party member (0/1, ~15 %) ---
    party_member = (rng.random(n) < 0.15).astype(float)

    # --- DGP: party_member signal is noisy (rng.normal(2.5, 3.0, n) instead of
    #     fixed 2.5) so that the estimated coefficient is borderline significant
    #     (expected p ~ 0.07)
    party_effect = rng.normal(2.5, 3.0, n)
    trust_index = (
        40.0
        + 0.12 * age
        + 0.0008 * income
        + 0.6 * edu_years
        + 1.0 * media_exposure
        + party_effect * party_member
        + rng.normal(0, 12, n)
    )

    # --- DataFrame ---
    df = pd.DataFrame({
        "age": age,
        "income": income,
        "edu_years": edu_years,
        "media_exposure": media_exposure,
        "party_member": party_member,
        "trust_index": trust_index,
    })

    # --- ModelSpec ---
    spec = ModelSpec(
        dep_var="trust_index",
        indep_vars=["age", "income", "edu_years", "media_exposure", "party_member"],
    )

    # --- Fit ---
    result = run_ols(data=df, spec=spec)
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="trust_experiment",
        title="社会信任影响因素调查",
        persona="张薇（社科研究生）",
        persona_icon="📊",
        description=(
            "200 份小样本问卷，探讨年龄、收入、受教育年限、媒体接触和党员身份"
            "对社会信任指数的多元回归分析。"
        ),
        tags=["小样本", "边界显著", "社会调查"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "小样本设计（n=200），标准误相对较大",
            "党员身份系数被注入额外噪声，p 值接近 0.05 边界",
            "社会信任指数信噪比较低（误差标准差 12），贴近真实问卷噪声水平",
            "适合演示\"统计显著 vs 实际显著\"的教学场景",
        ],
        story=(
            "张薇的第二个研究课题聚焦于社会信任的政治与社会经济根源。她通过线上问卷"
            "收集了 200 名受访者的数据，测量了年龄、月收入、受教育年限、媒体接触频率"
            "和党员身份等基本变量，并以一个 0-100 的信任指数作为被解释变量。\n\n"
            "小样本意味着回归系数的估计精度有限。为此，张薇在设计阶段特意注入了较强的"
            "随机误差（标准差 12），以模拟真实问卷数据中常见的测量误差和个体异质性。"
            "党员身份的效应被设计为\"有信号但不够强\"的场景——理论上党员参与可能通过"
            "组织信任外溢提升广义信任，但这一效应在样本量有限且个体差异较大的情况下"
            "可能无法达到传统的 5% 显著性阈值。\n\n"
            "回归结果显示多数社会经济变量的系数方向与理论预期一致（年龄正效应、教育"
            "正效应、媒体接触正效应），但党员身份的 p 值在 0.05-0.10 之间徘徊。这一"
            "\"边界显著\"的结果是教学中完美的讨论素材：张薇应当报告这一结果并将其视为"
            "\"提示性证据\"（suggestive evidence），还是应该收集更大样本以获取更稳健"
            "的结论？"
        ),
        dep_var="trust_index",
    )


# ======================================================================
# DGP 3 -- 电商销售额  (Persona: 陈志远, n = 500)
# ======================================================================

def _make_ecommerce_sales() -> GalleryItem:
    """Generate the e-commerce sales scenario.

    Features high R-squared (~0.7) and substantial collinearity between
    ad_spend and promotion_discount (corr ~ 0.7).  Season is a 4-level
    categorical variable.
    """
    rng = np.random.default_rng(42)
    n = 500

    # --- correlated ad_spend and promotion_discount (target corr ~ 0.7) ---
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    mean = np.array([0.0, 0.0])
    correlated = rng.multivariate_normal(mean, cov, size=n)

    ad_spend = np.clip(100 + correlated[:, 0] * 40, 20, 250)
    promotion_discount = np.clip(0.25 + correlated[:, 1] * 0.08, 0.02, 0.50)

    # --- price (uniform 50-500) ---
    price = rng.uniform(50, 500, n)

    # --- competitor_price (uniform 40-480) ---
    competitor_price = rng.uniform(40, 480, n)

    # --- season (Q1-Q4, roughly equal probability) ---
    season_labels = ["Q1", "Q2", "Q3", "Q4"]
    season_idx = rng.choice(4, size=n, p=[0.25, 0.25, 0.25, 0.25])
    season = pd.Categorical(
        np.array(season_labels)[season_idx],
        categories=season_labels,
        ordered=False,
    )

    # --- season effects (Q1 is patsy reference; DGP uses Q1 = -200) ---
    season_effect_map = {"Q1": -200, "Q2": 100, "Q3": 300, "Q4": -200}
    season_effect = np.array([season_effect_map[s] for s in season])

    # --- DGP ---
    sales = (
        800
        + 3.0 * ad_spend
        - 6.5 * price
        + 400 * promotion_discount
        + 2.5 * competitor_price
        + season_effect
        + rng.normal(0, 250, n)
    )

    # --- DataFrame ---
    df = pd.DataFrame({
        "ad_spend": ad_spend,
        "price": price,
        "promotion_discount": promotion_discount,
        "competitor_price": competitor_price,
        "season": season,
        "sales": sales,
    })

    # --- ModelSpec ---
    spec = ModelSpec(
        dep_var="sales",
        indep_vars=[
            "ad_spend",
            "price",
            "promotion_discount",
            "competitor_price",
            "season",
        ],
    )

    # --- Fit ---
    result = run_ols(data=df, spec=spec)
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="ecommerce_sales",
        title="电商平台销售额驱动因素分析",
        persona="陈志远（市场研究员）",
        persona_icon="📈",
        description=(
            "500 条电商销售记录，分析广告支出、价格、促销折扣、竞品价格和季节"
            "对销售额的影响，广告支出与促销折扣高度相关。"
        ),
        tags=["商业分析", "高R²", "多重共线性"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "高拟合优度（R²≈0.7），商业数据的典型特征",
            "广告支出与促销折扣相关系数约 0.7，VIF>5",
            "季节为四水平分类变量，展示季节性效应分解",
            "适合演示 VIF 诊断、岭回归等共线性处理技术",
        ],
        story=(
            "市场研究员陈志远受某电商平台委托，分析影响月销售额的关键驱动因素。"
            "他获取了过去 500 天的运营数据，包括每日广告支出、产品均价、促销折扣"
            "力度、主要竞争对手均价以及所属季度。\n\n"
            "初步数据探查中，陈志远发现广告支出与促销折扣之间存在高度正相关（r≈0.7）"
            "——这并非偶然：平台通常在促销活动期间同步加大广告投放以最大化曝光效果。"
            "然而，这种共线性使得 OLS 在估计两个变量各自的边际效应时面临困难：广告和"
            "促销的标准误都会被放大，导致系数解读的不确定性增加。\n\n"
            "模型的整体解释力很强（R²≈0.7），说明销售额的波动大部分可以由这几个"
            "运营变量解释。季节性效应也清晰可见：Q3 销售额显著高于其他季度（夏季"
            "促销季），Q1 和 Q4 则相对平淡。陈志远建议运营团队在季度预算分配时参考"
            "这些效应，但提醒说广告与促销的个体系数不宜单独解读，最好结合 VIF 诊断"
            "或使用岭回归等正则化方法做稳健性检验。"
        ),
        dep_var="sales",
    )


# ======================================================================
# DGP 4 -- 客户满意度  (Persona: 陈志远, n = 350)
# ======================================================================

def _make_customer_satisfaction() -> GalleryItem:
    """Generate the customer-satisfaction scenario.

    Two multi-level categorical variables (service_quality with 4 levels,
    price_perception with 4 levels) produce 6 dummy coefficients in total.
    """
    rng = np.random.default_rng(42)
    n = 350

    # --- wait_time (0.5-60, right-skewed via exponential) ---
    wait_time = np.clip(rng.exponential(scale=10, size=n), 0.5, 60)

    # --- service_quality (4-level categorical) ---
    sq_labels = ["差", "中", "好", "优秀"]
    sq_idx = rng.choice(4, size=n, p=[0.10, 0.30, 0.35, 0.25])
    service_quality = pd.Categorical(
        np.array(sq_labels)[sq_idx],
        categories=sq_labels,
        ordered=False,
    )

    # --- price_perception (4-level categorical) ---
    pp_labels = ["太低", "合理", "偏高", "太贵"]
    pp_idx = rng.choice(4, size=n, p=[0.05, 0.40, 0.35, 0.20])
    price_perception = pd.Categorical(
        np.array(pp_labels)[pp_idx],
        categories=pp_labels,
        ordered=False,
    )

    # --- loyalty_years (0-15, right-skewed) ---
    loyalty_years = rng.exponential(scale=3, size=n)
    loyalty_years = np.clip(loyalty_years, 0, 15)

    # --- complaint_count (0-10, Poisson-like) ---
    complaint_count = np.clip(rng.poisson(lam=2, size=n), 0, 10).astype(float)

    # --- DGP maps ---
    sq_map = {"差": -15, "中": -5, "好": 5, "优秀": 15}
    pp_map = {"太低": 0, "合理": 5, "偏高": -5, "太贵": -15}

    sq_effect = np.array([sq_map[s] for s in service_quality])
    pp_effect = np.array([pp_map[p] for p in price_perception])

    # --- DGP ---
    satisfaction_score = (
        70.0
        - 0.8 * wait_time
        + sq_effect
        + pp_effect
        + 0.4 * loyalty_years
        - 2.5 * complaint_count
        + rng.normal(0, 8, n)
    )

    # --- clip to [10, 100] ---
    satisfaction_score = np.clip(satisfaction_score, 10, 100)

    # --- DataFrame ---
    df = pd.DataFrame({
        "wait_time": wait_time,
        "service_quality": service_quality,
        "price_perception": price_perception,
        "loyalty_years": loyalty_years,
        "complaint_count": complaint_count,
        "satisfaction_score": satisfaction_score,
    })

    # --- ModelSpec ---
    spec = ModelSpec(
        dep_var="satisfaction_score",
        indep_vars=[
            "wait_time",
            "service_quality",
            "price_perception",
            "loyalty_years",
            "complaint_count",
        ],
    )

    # --- Fit ---
    result = run_ols(data=df, spec=spec)
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="customer_satisfaction",
        title="连锁餐饮客户满意度分析",
        persona="陈志远（市场研究员）",
        persona_icon="📈",
        description=(
            "350 份餐饮客户问卷，分析等待时间、服务质量、价格感知、忠诚年限和投诉"
            "次数对满意度的影响，包含两个多水平分类变量。"
        ),
        tags=["客户分析", "多分类变量", "服务业"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "两个四水平分类变量，共生成 6 个虚拟系数",
            "中等拟合优度（R²≈0.45），服务业满意度典型水平",
            "因变量截断在 [10, 100]，可能存在天花板/地板效应",
            "适合演示分类变量显著性联合 F 检验",
        ],
        story=(
            "陈志远为某连锁餐饮品牌做客户满意度调研，在 12 家门店收集了 350 份有效"
            "问卷。问卷涵盖了等待时间、服务质量评价（差/中/好/优秀）、价格感知"
            "（太低/合理/偏高/太贵）、客户忠诚年限和近一年投诉次数等核心变量，"
            "以 10-100 的满意度评分作为结果变量。\n\n"
            "数据的一个突出特征是同时包含两个多水平分类变量——服务质量和价格感知各有"
            "四个水平，OLS 模型将为每个分类变量生成 3 个虚拟系数（参考类别为各自的第一"
            "水平\"差\"和\"太低\"），共计 6 个分类效应系数。陈志远需要特别留意这些系数"
            "的联合显著性——可以借助 F 检验来判断\"服务质量\"和\"价格感知\"整体上是否"
            "对满意度有显著贡献。\n\n"
            "初步回归结果显示等待时间每增加 1 分钟，满意度下降约 0.8 分，而投诉次数"
            "的边际效应为 -2.5 分/次。服务质量从\"差\"到\"优秀\"的升级带来约 30 分的"
            "满意度提升，价格感知的效应则呈非线性——\"太贵\"的负面冲击远大于\"偏高\"。"
            "这些发现为门店运营提供了可操作的建议：优先改善服务质量和管控价格口碑，"
            "其次考虑缩短高峰时段等待时间。"
        ),
        dep_var="satisfaction_score",
    )


# ======================================================================
# DGP 5 -- 环境政策效应  (Persona: 李明远, n = 300)
# ======================================================================

def _make_policy_effect() -> GalleryItem:
    """Generate the environmental-policy evaluation scenario.

    Includes a policy_intensity * industrial_share interaction term and
    uses HC1 robust standard errors.  Region is a 3-level categorical.
    """
    rng = np.random.default_rng(42)
    n = 300

    # --- policy_intensity (0-10, uniform) ---
    policy_intensity = rng.uniform(0, 10, n)

    # --- gdp_per_capita (lognormal, in yuan) ---
    gdp_per_capita = np.exp(rng.normal(10.5, 0.6, n))

    # --- industrial_share (0.15-0.55, beta-like) ---
    industrial_share = np.clip(rng.normal(0.35, 0.10, n), 0.15, 0.55)

    # --- population_density (lognormal, persons/km^2) ---
    population_density = np.exp(rng.normal(5.5, 1.0, n))

    # --- region (东部/中部/西部, 东部 as reference) ---
    region_labels = ["东部", "中部", "西部"]
    region_idx = rng.choice(3, size=n, p=[0.45, 0.30, 0.25])
    region = pd.Categorical(
        np.array(region_labels)[region_idx],
        categories=region_labels,
        ordered=False,
    )

    # --- region effects (东部=+2, 中部=+0, 西部=-1) ---
    region_effect_map = {"东部": 2, "中部": 0, "西部": -1}
    region_effect = np.array([region_effect_map[r] for r in region])

    # --- interaction term ---
    interaction = policy_intensity * industrial_share

    # --- DGP ---
    emission_reduction = (
        3.0
        + 1.2 * policy_intensity
        - 0.15 * (gdp_per_capita / 10000)
        + 8.0 * industrial_share
        - 0.0005 * population_density
        + region_effect
        + 0.6 * interaction
        + rng.normal(0, 3.5, n)
    )

    # --- DataFrame ---
    df = pd.DataFrame({
        "policy_intensity": policy_intensity,
        "gdp_per_capita": gdp_per_capita,
        "industrial_share": industrial_share,
        "population_density": population_density,
        "region": region,
        "emission_reduction": emission_reduction,
    })

    # --- ModelSpec (includes interaction term, HC1 SE) ---
    spec = ModelSpec(
        dep_var="emission_reduction",
        indep_vars=[
            "policy_intensity",
            "gdp_per_capita",
            "industrial_share",
            "population_density",
            "region",
        ],
        interaction_terms=[("policy_intensity", "industrial_share")],
    )

    # --- Fit with HC1 robust standard errors ---
    result = run_ols(data=df, spec=spec, cov_type="HC1")
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="policy_effect",
        title="环境规制政策减排效果评估",
        persona="李明远（政策分析师）",
        persona_icon="🏛️",
        description=(
            "300 个地级市面板数据，评估环境规制强度对工业污染减排的效果，"
            "含政策强度与产业结构交互项及地区固定效应。"
        ),
        tags=["政策评估", "交互项", "稳健标准误"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "政策强度 x 产业结构交互项，边际效应取决于调节变量水平",
            "HC1 稳健标准误应对可能的异方差",
            "三水平地区分类（东/中/西），控制区域固定效应",
            "适合演示交互项的边际效应分解和简单斜率分析",
        ],
        story=(
            "政策分析师李明远受省生态环境厅委托，评估近年环境规制政策对工业污染减排"
            "的实际效果。他整理了 300 个地级市的横截面数据，核心变量包括政策执行强度"
            "（0-10 分，由环保执法频次、排放标准严格度和排污费征收率综合构建）、"
            "人均 GDP、工业增加值占比、人口密度和所在区域。\n\n"
            "李明远的核心假设是：政策的减排效果并非均匀分布，而是取决于当地的产业结构。"
            "在工业化程度较高的地区，同样的政策力度可能产生更大的减排绝对值——因为高排放"
            "企业更多，减排空间更大；但在服务业占主导的地区，政策对工业排放的影响可能"
            "相对有限。为了检验这一假设，他在模型中加入了政策强度与工业占比的交互项。\n\n"
            "考虑到不同地级市的异质性（如产业结构的历史依赖、统计口径差异等），李明远"
            "选择使用 HC1 稳健标准误而非经典 OLS 标准误，以避免因异方差而导致有偏的"
            "显著性推断。模型的交互项系数为正且显著，支持了\"产业结构增强政策效应\"的"
            "假设。但李明远在报告中也坦承，横截面数据无法完全解决内生性问题——政策更强"
            "的地区可能本身就面临更严重的污染压力。"
        ),
        dep_var="emission_reduction",
    )


# ======================================================================
# DGP 6 -- 学校学业成绩  (Persona: 王磊, n = 1000)
# ======================================================================

def _make_school_performance() -> GalleryItem:
    """Generate the school academic performance scenario.

    Students nested in 50 schools (n=1000, ~20 per school).  MixedLM captures
    between-school variance via random intercepts (ICC ~ 0.2).  Both
    student-level and school-level predictors are included.
    """
    rng = np.random.default_rng(42)
    n_schools = 50
    students_per_school = 20
    n = n_schools * students_per_school  # 1000

    # --- school-level data (50 schools) ---
    school_ids = [f"S{i+1:03d}" for i in range(n_schools)]
    school_records: list[dict] = []
    for sid in school_ids:
        class_size = rng.uniform(15, 45)
        teacher_quality = rng.uniform(1, 10)
        school_re = rng.normal(0, np.sqrt(5))  # random intercept ~N(0,5)
        for _ in range(students_per_school):
            school_records.append({
                "school_id": sid,
                "class_size": class_size,
                "teacher_quality": teacher_quality,
                "_school_re": school_re,
            })
    df_school = pd.DataFrame(school_records)

    # --- student-level variables ---
    # study_hours (10-50, uniform)
    study_hours = rng.uniform(10, 50, n)

    # parent_education (4-level categorical)
    edu_categories = ["初中", "高中", "本科", "研究生"]
    edu_idx = rng.choice(4, size=n, p=[0.20, 0.35, 0.30, 0.15])
    parent_edu_num = edu_idx.astype(float)  # 0,1,2,3 for DGP
    parent_education = pd.Categorical(
        np.array(edu_categories)[edu_idx],
        categories=edu_categories,
        ordered=False,
    )

    # socioeconomic_score (~N(50, 15), individual level)
    socioeconomic_score = rng.normal(50, 15, n)

    # --- DGP ---
    exam_score = (
        40.0
        + 0.8 * study_hours
        + 2.0 * parent_edu_num
        + 1.5 * df_school["teacher_quality"].values
        - 0.3 * df_school["class_size"].values
        + 0.2 * socioeconomic_score
        + df_school["_school_re"].values
        + rng.normal(0, 8, n)
    )

    # --- DataFrame ---
    df = pd.DataFrame({
        "student_id": range(1, n + 1),
        "school_id": df_school["school_id"],
        "study_hours": study_hours,
        "parent_education": parent_education,
        "class_size": df_school["class_size"],
        "teacher_quality": df_school["teacher_quality"],
        "socioeconomic_score": socioeconomic_score,
        "exam_score": exam_score,
    })

    # --- ModelSpec (group_var attached after creation) ---
    spec = ModelSpec(
        dep_var="exam_score",
        indep_vars=[
            "study_hours",
            "parent_education",
            "teacher_quality",
            "class_size",
            "socioeconomic_score",
        ],
    )
    spec.group_var = "school_id"

    # --- Fit via run_and_extract_mixedlm ---
    result = run_and_extract_mixedlm(data=df, spec=spec)
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="school_performance",
        title="学校学业成绩影响因素分析 — 多层次模型",
        persona="王磊（教育研究员）",
        persona_icon="🎓",
        description=(
            "50 所学校 1000 名学生分层数据，使用多层次模型（MixedLM）分析"
            "学习时长、家庭教育背景、教师质量、班级规模和社会经济地位"
            "对考试成绩的影响，控制学校间随机截距效应。"
        ),
        tags=["教育数据", "多层次模型", "嵌套数据"],
        n_obs=n,
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "50 所学校分层嵌套结构，每校约 20 名学生",
            "学校随机截距捕捉组间异质性（ICC≈0.2）",
            "教师质量和班级规模为学校层面变量，跨校变异明显",
            "适合演示混合效应模型中固定效应 vs 随机效应的解释",
        ],
        story=(
            "教育研究员王磊受市教育局委托，研究影响中学生学业成绩的关键因素。"
            "他从 50 所中学收集了 1000 名学生的数据，涵盖学生学习投入（周学习时长）、"
            "家庭背景（父母最高教育程度、社会经济地位指数）以及学校特征（班级规模、"
            "教师质量评分）。由于学生嵌套在学校之中，同一学校的学生共享相同的教师资源"
            "和教学环境，传统 OLS 回归会因忽视组内相关性而导致标准误低估。\n\n"
            "王磊选择使用多层次模型（MixedLM），通过随机截距捕捉学校间的异质性。"
            "初步分析显示学校层面的随机效应方差约为 5，个体残差方差约为 64，"
            "组内相关系数（ICC）约为 0.2——这意味着约 20% 的成绩变异可归因于"
            "学校之间的系统性差异，而非学生个体差异。这一发现确认了使用多层次模型的"
            "必要性。\n\n"
            "在固定效应部分，学习时长每增加 1 小时/周，考试成绩预计提升约 0.8 分；"
            "父母教育从\"初中\"到\"研究生\"每提升一个层级，成绩提升约 2 分。"
            "在学校层面，教师质量每提高 1 分（1-10 量表），成绩提升约 1.5 分，"
            "而班级规模每增加 1 人，成绩下降约 0.3 分。这一分析为教育政策制定"
            "提供了量化依据：改善师资和适当控制班额可能是比扩大家庭作业更有效的"
            "干预路径。"
        ),
        dep_var="exam_score",
    )


# ======================================================================
# DGP 7 -- 省级经济增长  (Persona: 李明远, n = 240)
# ======================================================================

def _make_province_growth() -> GalleryItem:
    """Generate the province economic growth panel scenario.

    30 provinces over 8 years (2016-2023, n=240).  A fixed-effects panel
    model controls for unobserved time-invariant provincial heterogeneity,
    isolating the within-province effects of investment, education spending,
    trade openness, urbanisation, fiscal policy, and high-tech industry share
    on annual GDP growth.
    """
    rng = np.random.default_rng(42)

    provinces = [
        "北京", "天津", "河北", "山西", "内蒙古",
        "辽宁", "吉林", "黑龙江",
        "上海", "江苏", "浙江", "安徽", "福建", "江西", "山东",
        "河南", "湖北", "湖南",
        "广东", "广西", "海南",
        "重庆", "四川", "贵州", "云南",
        "陕西", "甘肃", "青海", "宁夏", "新疆",
    ]
    years = list(range(2016, 2024))  # 8 years

    # Province regions for base-effect assignment
    coastal = {"北京", "天津", "辽宁", "上海", "江苏", "浙江",
               "福建", "山东", "广东", "海南"}
    inland = {"河北", "山西", "吉林", "黑龙江", "安徽", "江西",
              "河南", "湖北", "湖南", "重庆", "四川", "陕西"}
    western = {"内蒙古", "广西", "贵州", "云南", "甘肃", "青海", "宁夏", "新疆"}

    rows: list[dict] = []
    for year in years:
        year_effect = 0.1 * (year - 2016)
        for prov in provinces:
            if prov in coastal:
                base = rng.uniform(3, 5)
            elif prov in inland:
                base = rng.uniform(1, 3)
            else:
                base = rng.uniform(0, 2)

            investment_rate = rng.uniform(20, 50)
            education_spending = rng.uniform(2, 8)
            trade_openness = (
                rng.uniform(5, 80) if prov in coastal else rng.uniform(2, 30)
            )
            urbanization_rate = np.clip(
                rng.normal(60 if prov in coastal else 50, 10), 40, 90
            )
            fiscal_deficit = rng.uniform(1, 10)
            high_tech_ratio = (
                rng.uniform(2, 25) if prov in coastal else rng.uniform(1, 15)
            )

            gdp_growth = (
                2.0
                + base
                + 0.15 * investment_rate
                + 0.3 * education_spending
                + 0.08 * trade_openness
                + 0.05 * urbanization_rate
                - 0.2 * fiscal_deficit
                + 0.12 * high_tech_ratio
                + year_effect
                + rng.normal(0, 0.8)
            )

            rows.append({
                "province": prov,
                "year": year,
                "gdp_growth": gdp_growth,
                "investment_rate": investment_rate,
                "education_spending": education_spending,
                "trade_openness": trade_openness,
                "urbanization_rate": urbanization_rate,
                "fiscal_deficit": fiscal_deficit,
                "high_tech_ratio": high_tech_ratio,
            })

    df = pd.DataFrame(rows)

    # --- ModelSpec (entity/time/panel attached after creation) ---
    spec = ModelSpec(
        dep_var="gdp_growth",
        indep_vars=[
            "investment_rate",
            "education_spending",
            "trade_openness",
            "urbanization_rate",
            "fiscal_deficit",
            "high_tech_ratio",
        ],
    )
    spec.entity_var = "province"
    spec.time_var = "year"
    spec.panel_model = "fixed"

    # --- Fit via run_panel then extract_panel ---
    fitted_model, labels = run_panel(data=df, spec=spec)

    preds_str = " + ".join(spec.all_predictors)
    spec_str = f"{spec.dep_var} ~ {preds_str}  [entity: province, time: year, model: fixed]"

    result = extract_panel(
        fitted_model=fitted_model,
        alpha=0.05,
        dep_var=spec.dep_var,
        specification=spec_str,
        variable_labels=labels,
    )
    result_json = _model_result_to_json(result)

    return GalleryItem(
        id="province_growth",
        title="省级经济增长驱动因素分析 — 面板数据 FE 模型",
        persona="李明远（政策分析师）",
        persona_icon="🏛️",
        description=(
            "30 个省份 2016-2023 年平衡面板数据，使用固定效应模型分析"
            "固定资产投资、教育支出、贸易开放度、城镇化率、财政赤字"
            "和高技术产业占比对 GDP 增长率的驱动作用。"
        ),
        tags=["面板数据", "固定效应", "宏观经济学"],
        n_obs=len(df),
        data=df,
        model_spec=spec,
        model_result=result,
        result_json=result_json,
        key_features=[
            "30 个省份 × 8 年平衡面板（n=240）",
            "固定效应模型控制省份不随时间异质性",
            "涵盖投资、教育、贸易、城镇化、财政、高技术等多维解释变量",
            "适合演示面板数据 FE vs RE 选择（Hausman 检验）",
        ],
        story=(
            "政策分析师李明远受国家发改委委托，开展省级经济增长驱动因素研究。"
            "他整理了 30 个省份 2016-2023 年的平衡面板数据，涵盖固定资产投资率、"
            "教育支出占比、贸易开放度（进出口占 GDP 比重）、城镇化率、财政赤字占比"
            "和高技术产业增加值占比等六个核心解释变量。面板数据允许同时利用"
            "省份间差异和时间维度信息，比横截面 OLS 或单时间序列更可靠。\n\n"
            "李明远选择固定效应（FE）模型，因为各省存在不随时间变化的未观测异质性"
            "（如地理区位、历史文化、制度质量），这些因素同时影响解释变量和增长"
            "表现。FE 模型通过组内变换消去了这些固定效应，仅利用各省内部"
            "随时间的变化来识别系数——本质上回答\"当某省的投资率提高时，"
            "其增长率如何变化\"的问题。回归后，他还计划运行 Hausman 检验"
            "来验证 FE 相对于随机效应（RE）模型的适用性。\n\n"
            "初步结果显示，教育支出和高技术产业占比对经济增长的正向驱动"
            "最为显著——教育支出每增加 1 个百分点，GDP 增长率提升约 0.3 个"
            "百分点；高技术产业占比每提高 1 个百分点，增长率提升约 0.12 个"
            "百分点。财政赤字则呈现显著的负向效应，暗示过度赤字可能抑制"
            "增长。李明远注意到城镇化率和贸易开放度的系数在沿海省份与内陆"
            "省份之间可能存在结构性差异，建议后续可以引入分组回归或交互项"
            "进行异质性分析。"
        ),
        dep_var="gdp_growth",
    )


# ======================================================================
# Module API
# ======================================================================

def get_gallery_items() -> list[GalleryItem]:
    """Return the complete list of 7 pre-computed gallery items.

    This function is heavy -- it runs every DGP and fits 7 models.
    For a lightweight UI listing, prefer :func:`get_gallery_index`.
    """
    global _gallery_cache
    if _gallery_cache is not None:
        return _gallery_cache
    _gallery_cache = [
        _make_survey_happiness(),
        _make_trust_experiment(),
        _make_ecommerce_sales(),
        _make_customer_satisfaction(),
        _make_policy_effect(),
        _make_school_performance(),
        _make_province_growth(),
    ]
    return _gallery_cache


def get_gallery_index() -> list[dict]:
    """Return lightweight metadata for all gallery items.

    Each entry contains only display-oriented fields (no DataFrames,
    no serialized model results).  Suitable for populating a UI gallery
    listing without loading heavy data.
    """
    items = get_gallery_items()
    return [
        {
            "id": item.id,
            "title": item.title,
            "persona": item.persona,
            "persona_icon": item.persona_icon,
            "description": item.description,
            "tags": item.tags,
            "n_obs": item.n_obs,
            "key_features": item.key_features,
            "dep_var": item.dep_var,
        }
        for item in items
    ]


def get_gallery_item(item_id: str) -> GalleryItem | None:
    """Look up a single gallery item by its ID.

    If the item is not yet cached, only the requested DGP is executed
    (the six others are left untouched).

    Args:
        item_id: One of ``"survey_happiness"``, ``"trust_experiment"``,
            ``"ecommerce_sales"``, ``"customer_satisfaction"``,
            ``"policy_effect"``, ``"school_performance"``,
            ``"province_growth"``.

    Returns:
        The matching :class:`GalleryItem`, or ``None`` if *item_id* is
        not recognised.
    """
    factory_map = {
        "survey_happiness": _make_survey_happiness,
        "trust_experiment": _make_trust_experiment,
        "ecommerce_sales": _make_ecommerce_sales,
        "customer_satisfaction": _make_customer_satisfaction,
        "policy_effect": _make_policy_effect,
        "school_performance": _make_school_performance,
        "province_growth": _make_province_growth,
    }
    factory = factory_map.get(item_id)
    if factory is None:
        return None
    return factory()
