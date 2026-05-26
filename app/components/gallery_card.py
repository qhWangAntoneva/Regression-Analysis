"""Sample Gallery UI component for the regression analysis app.

Renders a card grid showing pre-computed regression analysis scenarios
organized by user personas. Users click a card to load the scenario's
data and pre-computed results into session_state.
"""

from __future__ import annotations

from collections import OrderedDict

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

GALLERY_AVAILABLE = False
try:
    from src.preprocessing.type_detector import VariableTypeDetector
    from src.utils.gallery import get_gallery_index, get_gallery_item  # type: ignore

    GALLERY_AVAILABLE = True
except ImportError:
    pass

# Persona display ordering (matches the three personas defined in the gallery)
_PERSONA_ORDER = [
    "张薇（社科研究生）",       # 张薇（社科研究生）
    "陈志远（市场研究员）",   # 陈志远（市场研究员）
    "李明远（政策分析师）",   # 李明远（政策分析师）
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_gallery_grid() -> None:
    """Render the sample gallery as a card grid inside a st.expander.

    Call this from the data upload page. The expander label is:
    ``:material/collections_bookmark: 示例数据分析场景库 (Sample Gallery)``

    Inside the expander:
    1. st.caption explaining this is demo data
    2. Group cards by persona with st.markdown headers
    3. Each persona section: row of cards using st.columns
    4. Each card: st.container(border=True) with:
       - Persona icon + scenario title (bold markdown heading)
       - Description (st.caption)
       - Tags as a horizontal line like ```标签1` . `标签2``` `` `
       - Key features as bullet list
       - n_obs as a small badge
       - "加载并查看结果" button (type="primary")
       - Support for showing brief story text in a small expander within the card
    5. When a card's button is clicked, call :func:`_load_gallery_item`
    """
    if st is None:
        return

    if not GALLERY_AVAILABLE:
        return

    with st.expander(
        ":material/collections_bookmark: 示例数据分析场景库 (Sample Gallery)",
        expanded=False,
    ):
        st.caption(
            "以下是预构建的回归分析示例场景，基于真实研究问题设计。\n"
            "点击任意场景卡片即可加载数据并查看完整的回归分析结果。\n"
            "所有数据均为模拟生成，仅用于学习与演示目的。"
        )

        try:
            gallery_items: list[dict] = get_gallery_index()
        except Exception as e:
            st.error(f"加载场景库索引失败: {e}")
            return

        if not gallery_items:
            st.info("暂无可用示例场景。")
            return

        # Group items by persona, preserving persona display order
        grouped: OrderedDict[str, list[dict]] = OrderedDict()
        for persona_key in _PERSONA_ORDER:
            grouped[persona_key] = []
        for item in gallery_items:
            persona = item.get("persona", "其他")
            if persona not in grouped:
                grouped[persona] = []
            grouped[persona].append(item)
        # Drop empty persona groups
        grouped = OrderedDict((k, v) for k, v in grouped.items() if v)

        # Render each persona section
        for persona, items in grouped.items():
            _render_persona_section(persona, items)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_persona_section(persona: str, items: list[dict]) -> None:
    """Render a single persona section heading and its row of cards.

    Args:
        persona: The persona display name (e.g. "张薇（社科研究生）").
        items: List of gallery item metadata dicts belonging to this persona.
    """
    if st is None:
        return

    st.markdown(f"### {persona}")
    st.divider()

    n_cards = len(items)
    # Cap at 3 columns so cards don't get too narrow
    n_cols = min(n_cards, 3)
    cols = st.columns(n_cols)

    for i, item in enumerate(items):
        col_idx = i % n_cols
        with cols[col_idx]:
            _render_card(item)


def _render_card(item: dict) -> None:
    """Render a single gallery card inside a bordered container.

    Args:
        item: Gallery metadata dict from ``get_gallery_index()``.
            Expected keys: id, title, persona_icon, description,
            tags, key_features, n_obs, dep_var, and optionally story.
    """
    if st is None:
        return

    item_id: str = item.get("id", "")
    title: str = item.get("title", "未命名场景")
    persona_icon: str = item.get("persona_icon", ":material/analytics:")
    description: str = item.get("description", "")
    tags: list[str] = item.get("tags", [])
    key_features: list[str] = item.get("key_features", [])
    n_obs = item.get("n_obs", "N/A")
    story_text: str = item.get("story", "")

    with st.container(border=True):
        # Title row with persona icon
        st.markdown(f"### {persona_icon} {title}")

        # Description
        if description:
            st.caption(description)

        # Tags bar
        if tags:
            tag_str = " · ".join(f"`{t}`" for t in tags)  # middle-dot separator
            st.caption(tag_str)

        # Key features bullet list
        if key_features:
            for feat in key_features:
                st.markdown(f"- {feat}")

        # Sample size badge
        st.caption(f"样本量: **{n_obs}**")

        # Optional story expander inside the card
        if story_text:
            with st.expander(":material/auto_stories: 场景故事", expanded=False):
                st.markdown(story_text)

        # Primary action button
        if st.button(
            "加载并查看结果",
            key=f"gallery_btn_{item_id}",
            type="primary",
            use_container_width=True,
        ):
            _load_gallery_item(item_id)


def _load_gallery_item(item_id: str) -> None:
    """Load a gallery item into session_state and navigate to results.

    Steps:
    1. Call ``get_gallery_item(item_id)`` from ``src.utils.gallery``.
    2. If the item is ``None``, show ``st.error`` and return.
    3. Run ``VariableTypeDetector`` on ``item.data`` to get the variables list.
    4. Set all required ``st.session_state`` keys so the rest of the app
       treats this as a completed regression analysis.
    5. Show ``st.toast`` success message.
    6. Attempt ``st.switch_page`` to the results page; fall back to a
       ``st.page_link`` if navigation is not supported in the current context.

    Args:
        item_id: The gallery item identifier (e.g. ``"survey_happiness"``).
    """
    if st is None:
        return

    if not GALLERY_AVAILABLE:
        st.error("场景库模块未加载，无法加载示例场景。")
        return

    with st.spinner("正在加载示例场景数据…"):
        # ---- 1. Fetch the full gallery item ----
        try:
            gallery_item = get_gallery_item(item_id)
        except Exception as e:
            st.error(f"加载场景数据失败: {e}")
            return

        if gallery_item is None:
            st.error(f"未找到示例场景: ``{item_id}``")
            return

        # ---- 2. Extract core objects ----
        data = gallery_item.data
        title: str = gallery_item.title
        model_result = gallery_item.model_result
        model_spec = gallery_item.model_spec

        # ---- 3. Detect variable types ----
        try:
            detector = VariableTypeDetector()
            variables = detector.detect(data)
        except Exception as e:
            st.error(f"变量类型检测失败: {e}")
            return

        # ---- 4. Build data summary (compatible with get_data_summary format) ----
        n_rows, n_cols = data.shape
        memory_bytes = int(data.memory_usage(deep=True).sum())
        memory_mb = memory_bytes / (1024 * 1024)
        memory_formatted = (
            f"{memory_bytes / 1024:.1f} KB"
            if memory_bytes < 1024 * 1024
            else f"{memory_mb:.2f} MB"
        )
        missing_count = int(data.isna().sum().sum())
        missing_pct = round(missing_count / max(n_rows * n_cols, 1) * 100, 2)
        missing_rates: dict[str, float] = {
            str(c): round(float(data[c].isna().mean()), 4) for c in data.columns
        }
        column_types: dict[str, str] = {
            str(c): str(data[c].dtype) for c in data.columns
        }

        data_summary: dict = {
            # Primary keys per the spec
            "rows": n_rows,
            "columns": n_cols,
            "memory_mb": round(memory_mb, 2),
            "encoding": "N/A (示例数据)",
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            # Compatibility keys matching get_data_summary() format
            "n_rows": n_rows,
            "n_cols": n_cols,
            "memory_bytes": memory_bytes,
            "memory_formatted": memory_formatted,
            "missing_rates": missing_rates,
            "column_types": column_types,
        }

        # ---- 5. Populate session_state ----
        st.session_state.data = data.copy()
        st.session_state.variables = variables
        st.session_state.data_summary = data_summary
        st.session_state.filename = f"示例: {title}"
        st.session_state.encoding = "N/A (示例数据)"
        st.session_state.uploaded_file_obj = None
        st.session_state.model_result = model_result
        st.session_state.model_spec = model_spec
        st.session_state.model_results_list = [model_result]
        st.session_state.model_run_time = True
        st.session_state.model_config = {
            "add_constant": True,
            "ci_level": 0.95,
            "se_type": "nonrobust",
            "missing_handling": "drop",
        }
        st.session_state.gallery_mode = True
        st.session_state.gallery_item_id = item_id
        st.session_state.gallery_item_title = title

        # ---- 6. Notify and navigate ----
        st.toast(f"✅ 已加载示例场景: {title}", icon="✅")

        try:
            st.switch_page("app/pages/04_model_results.py")
        except Exception:
            # Fallback: show a clickable link if switch_page is unavailable
            st.page_link(
                "app/pages/04_model_results.py",
                label=":material/arrow_forward: 点击查看回归结果",
                icon=":material/test_tube:",
            )
