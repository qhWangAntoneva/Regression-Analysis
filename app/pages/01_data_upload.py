# encoding: utf-8
"""
数据上传与预览页面

Streamlit 页面：文件上传、数据解析、预览和摘要。
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Data imports (lazy — wrapped in try/except for first-run safety)
# ---------------------------------------------------------------------------
PARSER_AVAILABLE = False
try:
    from src.data_io.parser import FileParser, get_data_summary, preview_dataframe
    from src.preprocessing.type_detector import VariableTypeDetector
    from app.components.data_table import render_data_preview, render_variable_info

    PARSER_AVAILABLE = True
except ImportError:
    pass


def render() -> None:
    """渲染数据上传页面。"""
    if st is None:
        return

    st.title(":material/upload: 数据上传与预览")

    if not PARSER_AVAILABLE:
        st.warning("数据解析模块未完全加载。请先安装依赖: `pip install pandas openpyxl plotly`")
        return

    # 初始化 session state
    if "data" not in st.session_state:
        st.session_state.data = None
    if "variables" not in st.session_state:
        st.session_state.variables = None
    if "data_summary" not in st.session_state:
        st.session_state.data_summary = None
    if "encoding" not in st.session_state:
        st.session_state.encoding = None
    if "filename" not in st.session_state:
        st.session_state.filename = None

    # 左侧：文件上传
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("上传文件")

        uploaded_file = st.file_uploader(
            "选择 CSV 或 Excel 文件",
            type=["csv", "tsv", "txt", "xls", "xlsx"],
            help="支持 CSV（UTF-8/GBK）、TSV、TXT、Excel (.xls/.xlsx）格式",
        )

        # 预览行数限制
        nrows = st.number_input(
            "预览行数",
            min_value=1,
            max_value=10000,
            value=1000,
            step=100,
            help="设置预览时读取的最大行数（0 表示读取全部）",
        )

        # 加载示例数据按钮
        st.divider()
        if st.button(":material/dataset: 加载示例数据集", use_container_width=True):
            _load_sample_data()
            st.rerun()

    with col_right:
        if uploaded_file is not None:
            _process_uploaded_file(uploaded_file, nrows)
        elif st.session_state.data is not None:
            _show_data_preview()
        else:
            st.info("请上传一个 CSV 或 Excel 文件，或点击「加载示例数据集」开始探索。")


def _process_uploaded_file(uploaded_file, nrows: int) -> None:
    """处理上传的文件。"""
    if st is None:
        return

    parser = FileParser()
    detector = VariableTypeDetector()

    ext = Path(uploaded_file.name).suffix.lower()

    with st.spinner("正在解析文件..."):
        try:
            # 保存到临时文件
            suffix = ext if ext else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                tmp_path = tmp.name

            try:
                # 解析
                if ext in (".xls", ".xlsx"):
                    df = parser.parse(tmp_path, nrows=nrows if nrows > 0 else None)
                    encoding = "N/A (Excel)"
                else:
                    df, encoding = parser.parse_csv(tmp_path, nrows=nrows if nrows > 0 else None)

                # 类型检测
                variables = detector.detect(df)
                summary = get_data_summary(df)

                # 保存到 session state
                st.session_state.data = df
                st.session_state.variables = variables
                st.session_state.data_summary = summary
                st.session_state.encoding = encoding
                st.session_state.filename = uploaded_file.name

            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            # 显示结果
            st.success(f"文件「{uploaded_file.name}」解析成功")
            _show_data_preview()

        except Exception as e:
            st.error(f"文件解析失败: {e}")


def _show_data_preview() -> None:
    """显示数据预览和摘要。"""
    if st is None or st.session_state.data is None:
        return

    df = st.session_state.data
    summary = st.session_state.data_summary
    variables = st.session_state.variables

    # 数据摘要卡片
    st.subheader("数据摘要")

    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("行数", summary["n_rows"])
        with col2:
            st.metric("列数", summary["n_cols"])
        with col3:
            st.metric("内存占用", summary.get("memory_formatted", "N/A"))
        with col4:
            enc = st.session_state.get("encoding", "N/A")
            st.metric("编码", enc)

    # 缺失率概览
    if summary and any(v > 0 for v in summary.get("missing_rates", {}).values()):
        with st.expander("缺失值详情", expanded=False):
            missing_cols = {
                col: rate
                for col, rate in summary["missing_rates"].items()
                if rate > 0
            }
            if missing_cols:
                st.write("以下列存在缺失值：")
                missing_df_data = [
                    {"列名": col, "缺失率": f"{rate * 100:.2f}%"}
                    for col, rate in sorted(missing_cols.items(), key=lambda x: -x[1])
                ]
                st.dataframe(missing_df_data, use_container_width=True)
            else:
                st.write("数据集中无缺失值。")

    # 变量信息表
    if variables:
        st.divider()
        render_variable_info(variables)

    # 数据预览表格
    st.divider()
    st.subheader("数据预览")
    render_data_preview(df)


def _load_sample_data() -> None:
    """生成示例回归数据集并加载到 session state。"""
    if st is None:
        return

    try:
        import numpy as np
        import pandas as pd

        from src.preprocessing.type_detector import VariableTypeDetector

        np.random.seed(42)
        n = 500

        # 生成模拟数据
        age = np.random.normal(45, 15, n).clip(18, 80)
        education = np.random.choice(["高中以下", "高中", "本科", "硕士", "博士"], n, p=[0.15, 0.25, 0.35, 0.18, 0.07])
        income = 3000 + 200 * (age - 18) + 5000 * (education == "本科") + 8000 * (education == "硕士") + 15000 * (education == "博士") + np.random.normal(0, 5000, n)
        experience = (age - 18) * np.random.uniform(0.5, 1.0, n).clip(0, 50)
        satisfaction = 3 + 0.02 * income / 1000 + np.random.normal(0, 0.8, n).clip(1, 5)
        is_urban = np.random.choice([0, 1], n, p=[0.4, 0.6])
        spending = 500 + 0.3 * income + 200 * is_urban + np.random.normal(0, 2000, n).clip(100, None)
        id_col = np.arange(1, n + 1)

        df = pd.DataFrame(
            {
                "id": id_col,
                "age": age.round(1),
                "education": education,
                "income": income.round(0).astype(int),
                "experience": experience.round(1),
                "satisfaction": satisfaction.round(2),
                "is_urban": is_urban,
                "spending": spending.round(0).astype(int),
            }
        )

        detector = VariableTypeDetector()
        variables = detector.detect(df)

        st.session_state.data = df
        st.session_state.variables = variables
        st.session_state.data_summary = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "memory_formatted": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
            "missing_rates": {str(c): 0.0 for c in df.columns},
            "column_types": {str(c): str(df[c].dtype) for c in df.columns},
        }
        st.session_state.encoding = "N/A (模拟数据)"
        st.session_state.filename = "模拟数据集 (示例)"

        st.success("示例数据集已加载！")

    except ImportError as e:
        st.error(f"生成示例数据失败（缺少依赖: {e}）")


# 页面入口
render()
