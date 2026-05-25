# Regression Analysis — 交接文档

> 最后更新: 2026-05-25
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 分支: master
> 当前提交: fdc7640 (fix: critical dtype loss + 4 regressions)
> 上次交接: 7659641 (v1.0.0 Phase 4 完成)
> 部署: https://qhwangantoneva.github.io/regression-analysis/

---

## 项目状态概览

| 阶段 | 状态 | 测试 |
|------|------|------|
| Phase 1 (POC) | 完成 | 56 tests |
| Phase 2 (MVP) | 完成 | 147 tests |
| Phase 3 (Beta) | 完成 | 278 tests |
| Phase 3.5 (Sample Gallery) | 完成 | +31 tests |
| Phase 4 (v1.0) | 完成 | — |
| **Phase 4.5 (Web + 测试补充 + Bug修复)** | **完成** | **397 tests** |
| **合计** | — | **397 tests** |

### Phase 4.5 进度 (2026-05-25 本 session)

| 子阶段 | 状态 | 说明 |
|--------|------|------|
| Web 适配 | 完成 | Pyodide 静态站 (web/ 目录 6 文件) + GitHub Pages 跨仓库部署 |
| 测试补充 | 完成 | +88 tests (sample_data 0%→100%, scatter 0%→78%, diagnostics 75%→96%) |
| Code Review | 完成 | 5-angle 深度审查，15 findings |
| Bug 修复 | 完成 | 5 bugs 修复 (1 critical + 2 high + 2 medium) |

---

## 项目结构

```
Regression Analysis/
├── CLAUDE.md                       # Agent 协作规则 (master 工作流 + 安全红线)
├── HANDOVER.md                     # 项目交接文档
├── TODO.md                         # 任务清单
├── ROADMAP.md                      # 项目路线图
├── CHANGELOG.md                    # v1.0.0 发布说明
├── scripts/
│   ├── benchmark.py                # 性能基准测试 (100K×20变量 OLS 0.15s)
│   └── generate_gallery_json.py    # Gallery 数据生成器 (DGP → JSON)
├── docs/
│   ├── 用户手册.md                  # 用户手册 (3 个完整案例)
│   ├── 开发者指南.md                # 架构概览 + 扩展指南
│   ├── 已知问题.md                  # 已知问题和限制
│   ├── 安全审计报告.md              # 依赖安全审计 (0 漏洞)
│   ├── 反馈指南.md                  # Bug 报告/功能建议模板 + GitHub Issues
│   └── v1.1_规划.md                # 12 项功能按优先级组织
├── .github/
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md           # 中文 Bug 报告模板
│       └── feature_request.md      # 中文功能建议模板
├── web/                            # Pyodide 静态 Web 应用 (新增)
│   ├── index.html                  # 5-tab UI (Data/Model/Results/Diagnostics/Export)
│   ├── css/styles.css              # 响应式设计 + CSS 变量
│   ├── js/
│   │   ├── app.js                  # 前端逻辑 (Pyodide 桥接/上传/Gallery/导出/图表)
│   │   └── gallery_data.js         # 5 个预计算场景 (322KB JSON, 自动生成)
│   ├── py/
│   │   └── bridge.py               # Pyodide 内运行: 文件解析/OLS/诊断/图表/导出
│   └── deploy.sh                   # 跨仓库部署 → qhWangAntoneva.github.io
├── app/
│   ├── app.py                    # Streamlit 主入口 (st.navigation)
│   ├── config.py                 # Streamlit 页面配置
│   ├── components/
│   │   ├── data_table.py         # 数据预览 + 变量信息表 + 类型覆盖UI
│   │   ├── variable_selector.py  # 因变量/自变量选择器
│   │   ├── model_control.py      # 模型控制面板 + 多模型对比控件
│   │   ├── result_card.py        # 系数表/统计量/ANOVA/对比表/警示
│   │   ├── export_dialog.py      # 导出选项面板
│   │   ├── onboarding.py         # 首次引导 + 中文错误提示
│   │   └── gallery_card.py       # Sample Gallery 卡片网格 + 加载逻辑
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
│       ├── logger.py             # loguru 配置
│       ├── persistence.py        # 会话持久化 + 崩溃恢复
│       ├── sample_data.py        # 3 个示例数据集
│       └── gallery.py            # Sample Gallery (5 场景 + 预计算结果 + JSON)
└── tests/
    ├── conftest.py               # 6个fixture: sample_df, sample_csv_*, etc.
    └── unit/
        ├── test_smoke.py         # 基础导入 + fixture 验证
        ├── test_parser.py        # 文件解析测试 (含 Excel)
        ├── test_encoding.py      # 编码检测 (chardet/GBK/UTF-8)
        ├── test_fitter.py        # 模型拟合 + OLS 正确性
        ├── test_diagnostics.py   # VIF/residual_tests/influence_stats/model_summary
        ├── test_results_phase2.py       # to_summary_dict, anova, latex, compare, summary_generator
        ├── test_visualization_phase2.py # scale-location, Cook's, coefficient plots, dashboard
        ├── test_exporter.py             # CSV/Excel/chart/results package export
        ├── test_phase2_remaining.py     # 文件大小限制, 类型覆盖, 数据筛选
        ├── test_gallery.py             # Sample Gallery (31 tests)
        ├── test_sample_data.py         # 3 示例数据集 (26 tests)
        └── test_scatter.py             # scatter_with_regression (13 tests)
```

## Web 应用架构 (新增)

### 两种运行模式

| 模式 | 启动方式 | 适用场景 |
|------|---------|---------|
| Streamlit | `uv run streamlit run app/app.py` | 本地开发/原型 |
| Pyodide Web | `bash web/deploy.sh` → GitHub Pages | 零安装浏览器访问 |

### Web 架构

```
Browser (HTML/JS)                Pyodide (WebAssembly)
------------------               ---------------------
File upload (drag & drop)  -->   bridge.parse_file()      → CSV/Excel 解析
Variable selection form    -->   bridge.run_regression()  → OLS via statsmodels
Tab navigation             -->   bridge.compute_diagnostics() → VIF, Shapiro, DW
Plotly.js                  <--   bridge.generate_*_chart() → Plotly JSON spec
Download buttons           <--   bridge.export_csv/excel() → CSV/Excel bytes
Gallery cards (instant)    -/-   (gallery_data.js 预计算 JSON, 无需 Pyodide)
```

### Web 版限制 (vs Streamlit 版)

| 功能 | Streamlit | Web |
|------|-----------|-----|
| 多模型对比图 | 支持 | 仅单模型 |
| 变量转换 (log/标准化) | 支持 | 未开放 (引擎支持) |
| LaTeX/HTML/Word 报告 | 支持 | CSV/Excel/Text/PNG |
| scatter_with_regression | 支持 | 未集成 |
| 交互项 | 支持 UI | 未开放 (引擎支持) |

### Web 版关键设计决策

1. **无 patsy 依赖** — 手动构造设计矩阵 (`pd.get_dummies()`)，避免 patsy import 问题
2. **分类变量处理** — 自动检测 numeric/categorical，dummies drop_first=True
3. **Gallery 离线可用** — 5 个预计算场景内嵌为 JS 数据，无 Pyodide/网络请求
4. **图表渲染** — Python 生成 Plotly JSON spec → JS plotly.js 渲染
5. **导出** — CSV 直接生成，Excel 需 Pyodide (openpyxl)，Gallery 模式 CSV 降级
6. **dtype 保留** — `parse_file` 返回 `columns` 元数据 → JS 侧透传 → bridge 还原 numeric

---

## 架构概览

4层架构 (Streamlit + Web 共用 Business Logic):
- **Presentation**: Streamlit pages + components / Web HTML+JS
- **Application**: session_state 管理跨页面通信 / Pyodide bridge
- **Business Logic**: data_io/preprocessing/modeling/results/visualization/export
- **Data**: 文件系统 (上传/导出) / 浏览器内存

数据流: 上传 → 解析 → 类型检测 → session_state → 变量选择 → 建模 → 结果 → 导出

## 关键技术选型

| 领域 | 选型 |
|------|------|
| UI 框架 | Streamlit (桌面) + Pyodide/HTML/JS (Web) |
| 统计引擎 | statsmodels (OLS) |
| 图表 | plotly (交互式) + matplotlib (静态导出) |
| 包管理 | uv |
| 测试 | pytest + pytest-cov |
| 编码检测 | chardet / UTF-8/GBK/latin-1 回退 |
| 导出 | openpyxl (Excel), kaleido (PNG/SVG), zipfile (综合报告) |

---

## 已知问题 / 注意事项

详细清单见 `docs/已知问题.md`。

核心提醒：
1. **__pycache__ 和 .claude/ 不要提交**: `.gitignore` 已配置
2. **编码问题**: Windows 中文终端打印 Unicode 可能报 GBK 错误，测试中避免
3. **Web bridge dtype 往返**: `parse_file` 返回的 `columns` 元数据必须随 `data` 一起传给 `run_regression`，否则 dtype 丢失导致所有连续性变量变成 dummy。已通过 `app.js:handleFile` 透传修复。
4. **statsmodels 0.14.6**: `fitted.params/bse/tvalues` 返回 numpy array，不是 pandas Series。bridge.py 已通过 `np.asarray()` 兼容。

---

## 测试说明

```bash
uv run python -m pytest tests/ -v              # 全部 397 tests
uv run python -m pytest tests/unit/test_XXX.py -v  # 单个文件
```

新功能需在 `tests/unit/` 下创建 `test_*.py` 测试文件，使用 `conftest.py` 中的 fixtures。

### 覆盖率 (84%)

| 等级 | 覆盖率 | 模块 |
|------|--------|------|
| 100% | gallery, exceptions, logger, fitter, sample_data |
| 96-98% | diagnostics, table, statsmodels_engine, missing |
| 87-94% | exporter, persistence, type_detector, outliers, latex_renderer |
| 72-78% | scatter, coefficient, residual, statistics, encoding |
| 65-71% | summary_generator, specification, html_report, parser |
| 0% | generate_sample_data.py (夹具生成工具, 可接受) |

---

## 启动开发

```bash
cd "C:/Users/lenovos/Regression Analysis"
uv run streamlit run app/app.py        # 启动 Streamlit 应用
uv run python -m pytest tests/ -v      # 运行测试 (397)
bash web/deploy.sh                      # 部署 Web 版到 GitHub Pages
```

## 快速参考

| 文件/模块 | 关键函数/类 |
|-----------|------------|
| `web/py/bridge.py` | `parse_file()`, `run_regression()`, `compute_diagnostics()`, `generate_diagnostic_charts()`, `generate_coefficient_chart()`, `export_csv()`, `export_excel()` |
| `web/js/app.js` | Pyodide 初始化, Tab 管理, 文件上传 (drag & drop), Gallery 加载, 回归执行, 图表渲染, 导出 |
| `web/index.html` | 5-tab 结构, upload area, model form, results/export panels |
| `web/deploy.sh` | 跨仓库部署 → `qhWangAntoneva.github.io/regression-analysis/` |
| `scripts/generate_gallery_json.py` | DGP 场景 → gallery_data.js |
| `parser.py` | `FileParser.parse()`, `FileParser.parse_csv()`, `FileParser.parse_excel()` |
| `type_detector.py` | `VariableTypeDetector.detect()`, `VariableInfo` |
| `specification.py` | `ModelSpec`, `build_formula()`, `build_design_matrix()` |
| `transforms.py` | `VariableTransformer` (log/standardize/center/square + interaction terms) |
| `fitter.py` | `ModelFitter.fit()`, `ModelFitter.fit_multiple()` |
| `statsmodels_engine.py` | `extract_statsmodels()`, `run_ols()` (supports cov_type) |
| `table.py` | `CoefficientRow`, `ModelResult`, `to_dataframe()`, `compare_models()` |
| `statistics.py` | `descriptive_stats()`, `correlation_matrix()` |
| `diagnostics.py` | `vif()`, `residual_tests()`, `influence_stats()` |
| `summary_generator.py` | `generate_summary_text()`, `generate_coefficient_interpretation()`, `generate_assumption_check_text()` |
| `missing.py` | `MissingValueHandler.analyze()`, `.handle()` (drop/mean/median) |
| `outliers.py` | `OutlierDetector.detect_iqr()`, `.detect_zscore()`, `.flag_outliers()` |
| `persistence.py` | `save_session()`, `load_session()`, `clear_session()` |
| `sample_data.py` | 3 built-in datasets (房价/工资/空气质量) |
| `exporter.py` | `DataExporter.export_csv()`, `.export_excel()`, `.export_chart()`, `.export_results_package()`, `.export_reproducibility_package()` |
| `latex_renderer.py` | `LatexRenderer.render_single()`, `.render_comparison()` |
| `html_report.py` | `HtmlReportGenerator.generate_full_report()` |
| `result_card.py` | `render_coefficient_table()`, `render_model_statistics()`, `render_anova_table()`, `render_comparison_table()`, `render_statistical_alerts()` |
| `model_control.py` | `render_model_controls()`, `render_model_comparison_controls()` (adds SE type selector) |
| `variable_selector.py` | `render_variable_selector()`, `render_transforms_ui()`, `render_interaction_ui()` |
| `export_dialog.py` | `render_export_options()`, `render_export_result()` |
| `data_table.py` | `render_data_preview()`, `render_variable_info_table()`, `render_missing_value_summary()`, `render_outlier_detection_ui()` |
| `onboarding.py` | `render_first_run_guide()`, `render_error_message()`, `render_help_tooltip()` |
| `gallery_card.py` | `render_gallery_grid()`, `_load_gallery_item()` |
| `gallery.py` | `GalleryItem`, `get_gallery_items()`, `get_gallery_index()`, `get_gallery_item()`, `_model_result_to_json()`, `_json_to_model_result()` |
| `coefficient.py` | `coefficient_plot()`, `coefficient_plot_single()` |
| `benchmark.py` | Performance benchmarks (1K-100K rows, 10-20 vars, OLS fit + visualization timing) |

---

## Code Review 已修复 Bug (本 session)

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | CRITICAL | JSON 往返后所有连续变量变 object → 全部被 one-hot 编码 | bridge.py 用 columns 元数据还原 numeric dtype |
| 2 | HIGH | mean/median 缺失策略不处理 dep_var NaN | bridge.py 增加 dep_var 填充逻辑 |
| 3 | HIGH | `rmse/aic/bic.toFixed()` 无 null 守卫 | app.js 加 null guards |
| 4 | MEDIUM | VIF 静默返回 None (同 Finding 1 根因) | bridge.py `_compute_vif` 还原 dtype |
| 5 | MEDIUM | `showError('export-error')` 目标 DOM 不存在 | index.html 添加 `#export-error` 元素 |

### 修复合入提交

```
fdc7640 fix: critical dtype loss bug and 4 other regressions in web bridge
e8a707d fix: deploy.sh URL + __pycache__
81d1b12 fix: bridge.py np.asarray() for statsmodels 0.14.6
```

---

## 下个 Session 建议

1. **Web 版功能补齐**: 多模型对比图、变量转换 UI、交互项 UI、scatter_with_regression 集成
2. **遗留低覆盖率模块**: summary_generator.py (69%), statistics.py (77%)
3. **xls 格式支持**: bridge.py 目前仅用 openpyxl，不支持旧 .xls (需 xlrd)
4. **_norm_ppf 休眠 bug**: Q-Q plot 近似公式错误（仅 scipy 缺失时触发，非紧急）
5. **重新部署**: 修完 bug 后需重新运行 `bash web/deploy.sh` 更新线上版本
