"""
数据探索页面

Streamlit 页面：描述性统计、相关系数矩阵、变量分布图、
缺失值处理、异常值检测、样本数据加载。
"""  # noqa: N999

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


def _render_sample_data_sidebar() -> None:
    """在侧边栏渲染样本数据加载区域。"""
    if st is None:
        return

    st.sidebar.divider()
    st.sidebar.markdown("### :material/dataset: 示例数据")
    st.sidebar.markdown("点击加载预置样本数据集：")

    # 获取样本数据集信息
    from src.utils.sample_data import get_sample_datasets, load_sample_dataset

    datasets = get_sample_datasets()
    dataset_names = list(datasets.keys())

    selected_dataset = st.sidebar.radio(
        "选择数据集",
        options=dataset_names,
        index=0,
        key="sample_dataset_radio",
        format_func=lambda name: (
            f"{name} ({datasets[name]['n_rows']}行, {datasets[name]['n_cols']}列)"
        ),
    )

    # 显示数据集描述
    if selected_dataset:
        info = datasets[selected_dataset]
        st.sidebar.caption(info["description"])

    if st.sidebar.button(
        ":material/download: 加载",
        use_container_width=True,
        type="primary",
        key="load_sample_data_btn",
    ):
        with st.sidebar.spinner("正在生成样本数据..."):
            try:
                from src.preprocessing.type_detector import VariableTypeDetector

                df = load_sample_dataset(selected_dataset)
                detector = VariableTypeDetector()
                variables = detector.detect(df)

                st.session_state.data = df
                st.session_state.variables = variables
                st.session_state.data_summary = {
                    "n_rows": len(df),
                    "n_cols": len(df.columns),
                    "memory_bytes": int(df.memory_usage(deep=True).sum()),
                    "memory_formatted": f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB",
                    "missing_rates": {str(c): float(df[c].isna().mean()) for c in df.columns},
                    "column_types": {str(c): str(df[c].dtype) for c in df.columns},
                }
                st.session_state.filename = f"样本数据: {selected_dataset}"
                st.session_state.encoding = "N/A (模拟数据)"

                st.sidebar.success(f"已加载「{selected_dataset}」！")
                st.toast(f"已加载「{selected_dataset}」共 {len(df)} 行数据", icon="✅")
            except Exception as e:
                st.sidebar.error(f"加载失败: {e}")


def render() -> None:
    """渲染数据探索页面。"""
    if st is None:
        return

    st.title(":material/insights: 数据探索")

    # 渲染侧边栏样本数据加载
    _render_sample_data_sidebar()

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
            missing_df["缺失率"] = (missing_df["缺失数量"] / len(df) * 100).round(2).apply(lambda x: f"{x}%")  # noqa: E501
            st.dataframe(missing_df, use_container_width=True)

    # ----- Phase 3.3: 缺失值处理 -----
    st.divider()
    st.subheader("缺失值处理")

    from src.preprocessing.missing import MissingValueHandler

    handler = MissingValueHandler()
    missing_stats = handler.analyze(df)

    with st.expander(":material/cleaning_services: 缺失值处理", expanded=False):
        # 显示缺失值分析
        col_data = missing_stats["columns"]
        has_missing = any(v["count"] > 0 for v in col_data.values())

        if not has_missing:
            st.success("数据集中无缺失值，无需处理。")
        else:
            # 显示分析结果表
            table_rows = []
            for col_name, info in col_data.items():
                if info["count"] == 0:
                    continue
                pct = info["percentage"]
                if pct > 20:
                    level = "🔴 严重"
                elif pct > 5:
                    level = "🟡 关注"
                else:
                    level = "🟢 低"

                suggestion = "建议删除列或填充" if pct > 20 else ("建议填充" if pct > 5 else "影响较小")  # noqa: E501

                table_rows.append({
                    "变量名": col_name,
                    "数据类型": info["dtype"],
                    "缺失数": info["count"],
                    "缺失率": f"{pct:.2f}%",
                    "等级": level,
                    "建议": suggestion,
                })

            st.markdown("**缺失值分析**")
            st.dataframe(table_rows, use_container_width=True)

            # 处理策略选择
            st.divider()
            strategy_labels = {
                "drop": "删除缺失行 (Drop)",
                "mean": "均值填充 (Mean)",
                "median": "中位数填充 (Median)",
            }
            strategy = st.selectbox(
                "选择处理策略",
                options=list(strategy_labels.keys()),
                format_func=lambda x: strategy_labels.get(x, x),
                key="missing_strategy",
            )

            # 可选列筛选
            cols_with_missing = [c for c, info in col_data.items() if info["count"] > 0]
            target_columns = st.multiselect(
                "选择要处理的列（不选则处理所有含缺失值的列）",
                options=cols_with_missing,
                default=cols_with_missing,
                key="missing_target_cols",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button(
                    ":material/check: 应用缺失值处理",
                    type="primary",
                    use_container_width=True,
                    key="apply_missing_handling",
                ):
                    with st.spinner("正在处理缺失值..."):
                        try:
                            before_rows = len(df)
                            before_missing = int(df.isna().sum().sum())

                            target = target_columns if target_columns else None
                            df_cleaned = handler.handle(df, strategy, columns=target)

                            after_rows = len(df_cleaned)
                            after_missing = int(df_cleaned.isna().sum().sum()) if strategy == "drop" else 0  # noqa: E501

                            # 更新 session_state
                            st.session_state.data = df_cleaned

                            # 重新检测变量类型
                            from src.preprocessing.type_detector import VariableTypeDetector
                            detector = VariableTypeDetector()
                            st.session_state.variables = detector.detect(df_cleaned)

                            # 更新 data_summary
                            if st.session_state.get("data_summary"):
                                summary = st.session_state.data_summary
                                summary["n_rows"] = len(df_cleaned)
                                summary["missing_rates"] = {
                                    str(c): float(df_cleaned[c].isna().mean()) for c in df_cleaned.columns  # noqa: E501
                                }

                            st.success("缺失值处理完成！")
                            st.info(f"处理前: {before_rows} 行, {before_missing} 个缺失值 → 处理后: {after_rows} 行, {after_missing} 个缺失值")  # noqa: E501
                            st.rerun()
                        except Exception as e:
                            st.error(f"缺失值处理失败: {e}")

            with col2:
                if st.button(
                    ":material/restart_alt: 重置数据",
                    use_container_width=True,
                    key="reset_data_after_missing",
                ):
                    # 如果有 uploaded_file_obj，重新解析
                    uploaded = st.session_state.get("uploaded_file_obj")
                    if uploaded is not None:
                        st.info("请重新上传文件或加载数据。")
                        # 清空 data 强制用户重新上传
                        st.session_state.data = None
                        st.session_state.variables = None
                        st.rerun()
                    else:
                        st.info("样本数据无法直接恢复，请重新加载。")

    # ----- Phase 3.3: 异常值检测 -----
    st.divider()
    st.subheader("异常值检测")

    from src.preprocessing.outliers import OutlierDetector

    with st.expander(":material/outlier: 异常值检测", expanded=False):
        if not numeric_cols:
            st.info("数据集中没有数值列可供检测。")
        else:
            st.markdown("选择要检测的数值变量并运行异常值检测。")

            out_cols = st.multiselect(
                "选择数值变量",
                options=numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                key="explore_outlier_cols",
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                out_method = st.radio(
                    "检测方法",
                    options=["iqr", "zscore"],
                    index=0,
                    horizontal=True,
                    key="explore_outlier_method",
                    format_func=lambda x: {"iqr": "IQR", "zscore": "Z-Score"}.get(x, x),
                )

            with col2:
                if out_method == "iqr":
                    out_param = st.number_input(
                        "IQR 倍数", min_value=0.5, max_value=5.0, value=1.5, step=0.1,
                        key="explore_outlier_mult",
                    )
                else:
                    out_param = st.number_input(
                        "Z-Score 阈值", min_value=1.0, max_value=6.0, value=3.0, step=0.5,
                        key="explore_outlier_threshold",
                    )

            # 显示异常值行切换
            show_outlier_rows = st.checkbox(
                "显示异常值行（在数据预览中高亮）",
                value=False,
                key="show_outlier_rows",
                disabled=not out_cols,
            )

            if st.button(
                ":material/search: 运行异常值检测",
                type="primary",
                use_container_width=True,
                key="run_outlier_detection",
            ):
                if not out_cols:
                    st.warning("请至少选择一个数值变量。")
                else:
                    with st.spinner("正在检测异常值..."):
                        try:
                            outlier_detector = OutlierDetector()
                            kwargs: dict = {}
                            if out_method == "iqr":
                                kwargs["multiplier"] = out_param
                            else:
                                kwargs["threshold"] = out_param

                            df_result, out_summary = outlier_detector.flag_outliers(
                                df, out_cols, method=out_method, **kwargs
                            )

                            # 保存到 session_state 供切换使用
                            st.session_state._outlier_df_result = df_result
                            st.session_state._outlier_summary = out_summary
                            st.session_state._outlier_active = True

                            st.success("异常值检测完成！")
                            st.session_state._outlier_cols = out_cols
                        except Exception as e:
                            st.error(f"异常值检测失败: {e}")

            # 显示检测结果
            out_summary_ss = st.session_state.get("_outlier_summary")
            if out_summary_ss:
                result_rows = []
                for col_name, info in out_summary_ss.items():
                    if "error" in info:
                        result_rows.append({
                            "变量名": col_name,
                            "异常值数": "N/A",
                            "异常率": "N/A",
                            "状态": f"⚠️ {info['error']}",
                        })
                    else:
                        pct = info["percentage"]
                        level = "🔴 较多" if pct > 10 else ("🟡 少量" if pct > 2 else "🟢 正常")
                        result_rows.append({
                            "变量名": col_name,
                            "异常值数": info["n_outliers"],
                            "异常率": f"{pct:.2f}%",
                            "状态": level,
                        })

                st.markdown("**检测结果**")
                st.dataframe(result_rows, use_container_width=True)

                # 如果开启显示异常值行，展示带标记的数据
                if show_outlier_rows and st.session_state.get("_outlier_df_result") is not None:
                    df_result = st.session_state._outlier_df_result
                    outlier_cols = st.session_state.get("_outlier_cols", [])
                    flag_cols = [f"{c}_outlier" for c in outlier_cols if f"{c}_outlier" in df_result.columns]  # noqa: E501

                    if flag_cols:
                        # 合并所有异常值标记
                        combined_mask = df_result[flag_cols].any(axis=1)
                        outlier_df = df_result[combined_mask]
                        st.markdown(f"**异常值数据行（共 {len(outlier_df)} 行）**")
                        st.dataframe(outlier_df, use_container_width=True)

    # 相关系数矩阵热力图
    st.divider()
    st.subheader("相关系数矩阵")

    if len(numeric_cols) >= 2 and PLOTLY_AVAILABLE:
        corr_matrix = df[numeric_cols].corr()

        # Use color scheme for accessibility-aware colorscale
        from app.config import get_color_scheme
        cs_corr = get_color_scheme()
        corr_cs = cs_corr.get("corr_colorscale", "RdBu_r")
        corr_template = cs_corr.get("plot_template", "plotly_white")

        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            colorscale=corr_cs,
            zmin=-1,
            zmax=1,
            showscale=True,
            annotation_text=corr_matrix.round(2).values,
        )

        fig.update_layout(
            title={"text": "皮尔逊相关系数矩阵", "x": 0.5, "xanchor": "center"},
            template=corr_template,
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
                cb_dist = st.session_state.get("colorblind_mode", False)
                fig = px.histogram(
                    df,
                    x=col,
                    marginal="box",
                    opacity=0.7,
                    labels={col: col},
                    color_discrete_sequence=["#2c7bb6"] if cb_dist else None,
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
                        cb_pie = st.session_state.get("colorblind_mode", False)
                        fig = px.pie(
                            freq,
                            names=selected_cat,
                            values="频数",
                            title=f"{selected_cat} 分布",
                            color_discrete_sequence=(
                                ["#2c7bb6", "#fdae61", "#abd9e9", "#bababa", "#5e4fa2"]
                                if cb_pie else None
                            ),
                        )
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("没有分类变量可供查看。")


# 页面入口
render()
