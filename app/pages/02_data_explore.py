# encoding: utf-8
"""
数据探索页面

Streamlit 页面：描述性统计、相关系数矩阵、变量分布图。
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError:
    st = None  # type: ignore[assignment]

# Lazy imports
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.figure_factory as ff

    PLOTLY_AVAILABLE = True
except ImportError:
    px = None  # type: ignore[assignment]
    ff = None  # type: ignore[assignment]


def render() -> None:
    """渲染数据探索页面。"""
    if st is None:
        return

    st.title(":material/insights: 数据探索")

    # 检查数据是否存在
    if st.session_state.get("data") is None:
        st.warning("请先在「数据上传」页面上传数据集。")
        st.page_link("app/pages/01_data_upload.py", label="前往数据上传", icon="📂")
        return

    df = st.session_state.data
    variables = st.session_state.variables

    if not PLOTLY_AVAILABLE:
        st.warning("plotly 未安装。部分可视化功能受限。请运行: `pip install plotly`")

    # 描述性统计
    st.subheader("描述性统计")

    # 只显示数值列的统计
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc.columns = ["计数", "均值", "标准差", "最小值", "25%", "50%", "75%", "最大值"]
        desc = desc.round(4)

        # 添加列类型信息
        if variables:
            type_map = {v.name: v.inferred_type for v in variables}
            desc.insert(0, "推断类型", desc.index.map(lambda x: type_map.get(str(x), "-")))

        st.dataframe(desc, use_container_width=True)
    else:
        st.info("数据集中没有数值列。")

    # 缺失值统计
    missing_counts = df.isna().sum()
    if missing_counts.sum() > 0:
        with st.expander("缺失值详情", expanded=False):
            missing_df = missing_counts[missing_counts > 0].reset_index()
            missing_df.columns = ["列名", "缺失数量"]
            missing_df["缺失率"] = (missing_df["缺失数量"] / len(df) * 100).round(2).apply(lambda x: f"{x}%")
            st.dataframe(missing_df, use_container_width=True)

    # 相关系数矩阵热力图
    st.divider()
    st.subheader("相关系数矩阵")

    if len(numeric_cols) >= 2 and PLOTLY_AVAILABLE:
        corr_matrix = df[numeric_cols].corr()

        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            showscale=True,
            annotation_text=corr_matrix.round(2).values,
        )

        fig.update_layout(
            title={"text": "皮尔逊相关系数矩阵", "x": 0.5, "xanchor": "center"},
            template="plotly_white",
            width=800,
            height=800,
        )

        st.plotly_chart(fig, use_container_width=True)
    elif len(numeric_cols) < 2:
        st.info("需要至少 2 个数值列才能计算相关系数。")
    else:
        st.info("请安装 plotly 以显示相关系数热力图。")

    # 变量分布图
    st.divider()
    st.subheader("变量分布")

    if PLOTLY_AVAILABLE:
        # 为每个数值列绘制直方图
        numeric_cols_to_plot = numeric_cols[:10]  # 最多显示 10 列

        if not numeric_cols_to_plot:
            st.info("没有数值列可显示分布。")
        else:
            # 允许用户选择特定列
            selected_cols = st.multiselect(
                "选择要查看分布的变量",
                options=numeric_cols,
                default=numeric_cols_to_plot[:4],
                key="dist_cols",
            )

            for col in selected_cols:
                fig = px.histogram(
                    df,
                    x=col,
                    marginal="box",
                    opacity=0.7,
                    labels={col: col},
                )
                fig.update_layout(
                    template="plotly_white",
                    title={"text": f"{col} 分布", "x": 0.5, "xanchor": "center"},
                    bargap=0.05,
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("请安装 plotly 以显示变量分布图。")
        st.code("pip install plotly", language="bash")

    # 分类变量频率
    st.divider()
    st.subheader("分类变量频率")

    if variables:
        cat_vars = [v for v in variables if v.inferred_type in ("categorical", "binary")]
        if cat_vars:
            selected_cat = st.selectbox(
                "选择分类变量查看频率",
                options=[v.name for v in cat_vars],
                key="cat_freq",
            )
            if selected_cat and selected_cat in df.columns:
                freq = df[selected_cat].value_counts().reset_index()
                freq.columns = [selected_cat, "频数"]
                freq["占比"] = (freq["频数"] / len(df) * 100).round(2).apply(lambda x: f"{x}%")

                col1, col2 = st.columns([1, 1])
                with col1:
                    st.dataframe(freq, use_container_width=True)
                with col2:
                    if PLOTLY_AVAILABLE:
                        fig = px.pie(
                            freq,
                            names=selected_cat,
                            values="频数",
                            title=f"{selected_cat} 分布",
                        )
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有分类变量可供查看。")


# 页面入口
render()
