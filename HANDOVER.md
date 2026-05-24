# Regression Analysis — 交接文档

> 生成日期: 2026-05-25
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 分支: master
> 当前提交: 0bb037f

---

## 项目状态概览

| 阶段 | 状态 | 测试 |
|------|------|------|
| Phase 1 (POC) | ✅ 完成 | 56 tests |
| Phase 2 (MVP) | ✅ 全部完成 | 147 tests |
| Phase 3 (Beta) | ❌ 未开始 | — |
| Phase 4 (v1.0) | ❌ 未开始 | — |

## 项目结构

```
Regression Analysis/
├── app/
│   ├── app.py                    # Streamlit 主入口 (st.navigation)
│   ├── config.py                 # Streamlit 页面配置
│   ├── components/
│   │   ├── data_table.py         # 数据预览 + 变量信息表 + 类型覆盖UI
│   │   ├── variable_selector.py  # 因变量/自变量选择器
│   │   ├── model_control.py      # 模型控制面板 + 多模型对比控件
│   │   ├── result_card.py        # 系数表/统计量/ANOVA/对比表/警示
│   │   ├── export_dialog.py      # 导出选项面板
│   │   └── onboarding.py         # 首次引导 + 中文错误提示
│   └── pages/
│       ├── 01_data_upload.py     # 文件上传 + 类型覆盖UI + 文件大小限制
│       ├── 02_data_explore.py    # 描述性统计 + 相关矩阵 + 分布图
│       ├── 03_model_spec.py      # 变量选择 + 模型选项 + 数据筛选
│       ├── 04_model_results.py   # 系数表/统计量/多模型对比/诊断图
│       └── 06_export.py          # 数据/结果/图表/综合报告导出
├── src/
│   ├── data_io/
│   │   ├── parser.py             # CSV/Excel 解析, 编码自动检测
│   │   ├── encoding.py           # chardet/UTF-8/GBK/latin-1 回退
│   │   └── exporter.py           # CSV/Excel/图表/结果包导出
│   ├── modeling/
│   │   ├── specification.py      # ModelSpec dataclass, build_formula
│   │   ├── fitter.py             # ModelFitter (fit/fit_multiple)
│   │   ├── diagnostics.py        # VIF, residual_tests, influence_stats
│   │   └── engines/
│   │       └── statsmodels_engine.py  # OLS 适配器 + extract_statsmodels
│   ├── preprocessing/
│   │   └── type_detector.py      # VariableInfo + VariableTypeDetector
│   ├── results/
│   │   ├── table.py              # CoefficientRow + ModelResult + to_dataframe + compare_models
│   │   ├── statistics.py         # descriptive_stats + correlation_matrix
│   │   └── summary_generator.py  # 中文摘要/系数解读/假设检验文本
│   ├── visualization/
│   │   ├── scatter.py            # scatter_with_regression (plotly)
│   │   ├── residual.py           # residual_vs_fitted, qq_plot, scale_location, cooks_distance, diagnostic_dashboard
│   │   └── coefficient.py        # coefficient_plot (多模型), coefficient_plot_single (单模型)
│   └── utils/
│       ├── exceptions.py         # 5级异常层次
│       └── logger.py             # loguru 配置
└── tests/
    ├── conftest.py               # 6个fixture: sample_df, sample_csv_*, etc.
    └── unit/
        ├── test_smoke.py         # 基础导入 + fixture 验证
        ├── test_parser.py        # 文件解析测试
        ├── test_fitter.py        # 模型拟合 + OLS 正确性
        ├── test_results_phase2.py       # to_summary_dict, anova, latex, compare, summary_generator
        ├── test_visualization_phase2.py # scale-location, Cook's, coefficient plots, dashboard
        ├── test_exporter.py             # CSV/Excel/chart/results package export
        └── test_phase2_remaining.py     # 文件大小限制, 类型覆盖, 数据筛选
```

## 架构概览

4层架构:
- **Presentation**: Streamlit pages + components
- **Application**: session_state 管理跨页面通信
- **Business Logic**: data_io/preprocessing/modeling/results/visualization/export
- **Data**: 文件系统 (上传/导出)

数据流: 上传 → 解析 → 类型检测 → session_state → 变量选择 → 建模 → 结果 → 导出

## 关键技术选型

| 领域 | 选型 |
|------|------|
| UI 框架 | Streamlit |
| 统计引擎 | statsmodels (OLS) |
| 图表 | plotly (交互式) + matplotlib (静态导出) |
| 包管理 | uv |
| 测试 | pytest + pytest-cov |
| 编码检测 | chardet / UTF-8/GBK/latin-1 回退 |
| 导出 | openpyxl (Excel), kaleido (PNG/SVG), zipfile (综合报告) |

## Phase 2 已实现功能清单

### TODO 2.1 数据导入与预览
- [x] CSV 解析 (UTF-8/GBK 自动检测)
- [x] Excel (.xlsx/.xls) 解析
- [x] 数据表格预览组件
- [x] 变量类型自动检测 (id/continuous/categorical/binary/text)
- [x] **变量类型手动覆盖 UI** (expander + dropdown)
- [x] 缺失值统计与展示
- [x] **文件大小限制**: >50MB 警告, >200MB 阻止

### TODO 2.2 变量选择与模型配置
- [x] 变量选择器 (因变量下拉 + 自变量多选, 自动排除 ID)
- [x] 模型规格构建 (ModelSpec + build_formula)
- [x] 描述性统计面板
- [x] **数据子集筛选** (数值滑块 + 分类多选)
- [x] 模型控制面板 (常数项/CI水平/标准误类型/缺失值处理)

### TODO 2.3 回归结果展示
- [x] 系数表 (绿色高亮 p<0.05/0.01)
- [x] 模型统计量网格 (R²/Adj-R²/RMSE/AIC/BIC/LogL/F/N)
- [x] ANOVA 表
- [x] 多模型对比表

### TODO 2.4 诊断图表
- [x] 残差 vs 拟合值图
- [x] Q-Q 图
- [x] 尺度-位置图
- [x] Cook's Distance 图
- [x] 2×2 诊断总览面板
- [x] 统计警示 (VIF>10 标红, VIF>5 警告, Cook's D, DW)
- [x] 系数 dot-whisker 图 (单模型 + 多模型)

### TODO 2.5 导出
- [x] CSV 导出 (st.download_button)
- [x] Excel 导出
- [x] 图表 PNG/SVG 导出
- [x] 综合报告 ZIP 打包
- [x] 导出选项面板 (格式/路径/内容选择)

### TODO 2.6 用户体验
- [x] 全中文界面
- [x] 首次使用 3 步引导 (popover)
- [x] 中文错误提示 (无 traceback)
- [x] 内联帮助 (tooltip + 组件 help 参数)
- [x] 加载示例数据集按钮

## 关键 session_state keys

| Key | 类型 | 用途 |
|-----|------|------|
| `data` | pd.DataFrame | 原始数据 |
| `filtered_data` | pd.DataFrame | 筛选后数据 (可选) |
| `variables` | list[VariableInfo] | 变量类型元数据 |
| `data_summary` | dict | 数据摘要 (行/列/内存/缺失) |
| `encoding` | str | 文件编码 |
| `filename` | str | 原始文件名 |
| `uploaded_file_obj` | UploadedFile | 文件大小/元信息 |
| `type_overrides` | dict | {变量名: 覆盖类型} |
| `model_result` | ModelResult | 单模型结果 |
| `model_results_list` | list[ModelResult] | 多模型结果 (用于对比) |
| `model_spec` | ModelSpec | 模型规格 |
| `model_config` | dict | 模型配置 (常数/CI/SE/缺失) |
| `export_charts` | dict | {图表名: Figure} |

## 测试说明

```bash
uv run python -m pytest tests/ -v              # 全部 147 tests
uv run python -m pytest tests/unit/test_XXX.py -v  # 单个文件
```

新功能需在 `tests/unit/` 下创建 `test_*.py` 测试文件，使用 `conftest.py` 中的 fixtures。

## 已知问题 / 注意事项

1. **__pycache__ 不要提交**: `.gitignore` 已配置
2. **.claude/ 不要提交**: `.gitignore` 已配置  
3. **LF/CRLF 警告**: Windows 环境下 Git 会提示换行符转换，不影响功能
4. **kaleido 依赖**: 图表 PNG 导出需要 `kaleido>=1.3.0`，已在 `pyproject.toml` 中添加
5. **openpyxl**: Excel 导出需要，在依赖中
6. **worktree 隔离注意事项**: 使用 Agent worktree 隔离时，必须让 agent 在最后一步 `git add -A && git commit && git push`，否则 worktree 清理后更改丢失。推荐直接修改 master 分支

## Phase 3 (Beta) 待实现

参考 `TODO.md` Phase 3 部分，主要工作包括:

### 3.1 高级建模功能
- 变量转换 UI (对数/标准化/中心化/平方项)
- 交互项创建 UI
- 稳健标准误 (HC0-HC3) — 当前只有 classic/HC1
- 多模型并列比较表增强
- 系数图 (dot-whisker) — 已在 Phase 2 实现

### 3.2 完整导出功能
- LaTeX 表格导出 (Jinja2 + booktabs)
- HTML 报告导出
- SVG 图表导出 — 已在 Phase 2 实现
- 完整分析报告
- 分析复现包

### 3.3 数据增强
- 缺失值处理策略 (删除/均值/中位数)
- 异常值检测与提示
- 变量标签管理

### 3.4 工程化
- 性能优化 (缓存策略)
- 会话状态持久化
- 崩溃恢复
- 用户引导 + 示例数据集

### 3.5 Beta 验证
- Beta 测试 ≥ 10 人
- 无 P0 Bug
- 覆盖率 ≥ 90%

## 启动开发

```bash
cd "C:/Users/lenovos/Regression Analysis"
uv run streamlit run app/app.py        # 启动应用
uv run python -m pytest tests/ -v      # 运行测试
```

## 快速参考

| 文件/模块 | 关键函数/类 |
|-----------|------------|
| `parser.py` | `FileParser.parse()`, `FileParser.parse_csv()`, `FileParser.parse_excel()` |
| `type_detector.py` | `VariableTypeDetector.detect()`, `VariableInfo` |
| `specification.py` | `ModelSpec`, `build_formula()`, `build_design_matrix()` |
| `fitter.py` | `ModelFitter.fit()`, `ModelFitter.fit_multiple()` |
| `statsmodels_engine.py` | `extract_statsmodels()`, `run_ols()` |
| `table.py` | `CoefficientRow`, `ModelResult`, `to_dataframe()`, `compare_models()` |
| `statistics.py` | `descriptive_stats()`, `correlation_matrix()` |
| `diagnostics.py` | `vif()`, `residual_tests()`, `influence_stats()` |
| `summary_generator.py` | `generate_summary_text()`, `generate_coefficient_interpretation()`, `generate_assumption_check_text()` |
| `exporter.py` | `DataExporter.export_csv()`, `.export_excel()`, `.export_chart()`, `.export_results_package()` |
| `result_card.py` | `render_coefficient_table()`, `render_model_statistics()`, `render_anova_table()`, `render_comparison_table()`, `render_statistical_alerts()` |
| `model_control.py` | `render_model_controls()`, `render_model_comparison_controls()` |
| `export_dialog.py` | `render_export_options()`, `render_export_result()` |
| `onboarding.py` | `render_first_run_guide()`, `render_error_message()`, `render_help_tooltip()` |
| `coefficient.py` | `coefficient_plot()`, `coefficient_plot_single()` |
