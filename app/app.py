# encoding: utf-8
"""Regression Analysis -- Streamlit main entry point."""

import streamlit as st

from app.config import configure_page

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit command)
# ---------------------------------------------------------------------------
configure_page()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 1.8rem; margin: 0;">📊</h1>
        <h2 style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            Regression Analysis
        </h2>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# Define available pages using st.Page (Streamlit >=1.35)
# In Phase 1 these are placeholders -- real pages will be added in later phases.
pages = {
    "Data": [
        st.Page("app/pages/data_upload.py", title="Upload & Preview", icon=":inbox_tray:", default=True),
    ],
    "Analysis": [
        st.Page("app/pages/model_fitting.py", title="Model Fitting", icon=":gear:"),
        st.Page("app/pages/diagnostics.py", title="Diagnostics", icon=":test_tube:"),
    ],
    "Results": [
        st.Page("app/pages/export.py", title="Export Report", icon=":page_facing_up:"),
    ],
}

pg = st.navigation(pages)
pg.run()

# ---------------------------------------------------------------------------
# Sidebar project info
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Version:** 0.1.0 (POC)

    **GitHub:** [qhWangAntoneva/Regression-Analysis](https://github.com/qhWangAntoneva/Regression-Analysis)
    """
)

# ---------------------------------------------------------------------------
# Placeholder pages for pages that don't exist yet
# ---------------------------------------------------------------------------
# st.Page will 404 if the file doesn't exist.  We create placeholder modules
# here so the app doesn't crash before those pages are implemented.
import importlib
import sys
from pathlib import Path

_pages_dir = Path(__file__).parent / "pages"
_page_modules = ["data_upload", "model_fitting", "diagnostics", "export"]

for _mod in _page_modules:
    _path = _pages_dir / f"{_mod}.py"
    if not _path.exists():
        # Create a minimal placeholder module
        _code = (
            f'# encoding: utf-8\n'
            f'"""Placeholder: {_mod} -- not yet implemented."""\n\n'
            f'import streamlit as st\n\n\n'
            f'def run() -> None:\n'
            f'    st.info(":construction: This page is not yet implemented. It will be built in a later phase.")\n'
            f'    st.write(f"**Module:** `{_mod}.py`")\n\n\n'
            f'if __name__ == "__page__":\n'
            f'    run()\n'
        )
        _path.write_text(_code, encoding="utf-8")
        # Force reimport
        if _mod in sys.modules:
            del sys.modules[_mod]
        spec = importlib.util.spec_from_file_location(_mod, _path)
        if spec and spec.loader:
            _mod_obj = importlib.util.module_from_spec(spec)
            sys.modules[_mod] = _mod_obj
            spec.loader.exec_module(_mod_obj)
