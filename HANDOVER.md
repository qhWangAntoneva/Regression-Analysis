# Regression Analysis — 交接文档

> 最后更新: 2026-05-25
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 分支: master
> 当前提交: 7659641 (v1.0.0 release — Phase 4 完成)
> 标签: v1.0.0
> 上次交接: ebda810 (Phase 3.5 Sample Gallery)

---

## 项目状态概览

| 阶段 | 状态 | 测试 |
|------|------|------|
| Phase 1 (POC) | ✅ 完成 | 56 tests |
| Phase 2 (MVP) | ✅ 全部完成 | 147 tests |
| Phase 3 (Beta) | ✅ 全部完成 | 278 tests |
| Phase 3.5 (Sample Gallery) | ✅ 完成 | +31 tests |
| Phase 4 (v1.0) | ✅ 完成 | — |
| **合计** | — | **309 tests** |

### Phase 4 进度

| 子阶段 | 状态 | 说明 |
|--------|------|------|
| 4.1 最终打磨 | ✅ 完成 | UI 审查、色盲友好、响应式布局、键盘导航 |
| 4.2 文档 | ✅ 完成 | `docs/用户手册.md` (3案例), `docs/开发者指南.md`, `docs/已知问题.md` |
| 4.3 发布准备 | ✅ 完成 | 性能基准测试、依赖安全审计、v1.0.0 tag |
| 4.4 发布后跟踪 | ✅ 完成 | 反馈指南、Issue 模板、v1.1 规划 |

### Phase 4.1 最终打磨详情

- **色盲友好模式**: `app/config.py` 新增 `get_color_scheme()` 函数，侧边栏 toggle 切换普通/色盲安全调色板（viridis 替代红绿显著性标注）
- **响应式布局**: 全部页面 1366×768 最低分辨率适配，`st.columns` 宽度优化
- **键盘导航**: Tab 键逻辑流、accesskey 快捷键（Streamlit 支持范围内）
- **视觉一致性**: 统一间距/字号/卡片样式，expander 风格统一，Sample Gallery 卡片匹配整体设计
- **版本号**: Sidebar `Version: 1.0.0`

### Phase 4.2 文档详情

- `docs/用户手册.md` (23.6 KB): 3 个完整案例 — 房价影响因素分析、员工满意度调查、政策效果评估
- `docs/开发者指南.md` (19.2 KB): 4 层架构图、引擎/可视化/导出扩展指南、session_state 生命周期
- `docs/已知问题.md` (10.3 KB): 涵盖编码问题/LF-CRLF/kaleido/性能限制/9 项功能限制

### Phase 4.3 发布准备详情

- `scripts/benchmark.py`: 100K 行 × 20 变量 OLS 拟合 **0.15s** (目标 ≤3s)
- `docs/安全审计报告.md`: 22 个核心依赖 0 已知漏洞
- `CHANGELOG.md`: 完整 Phase 1-4 发布说明
- 标签: `v1.0.0` annotated tag 已推送

## 项目结构

```
Regression Analysis/
├── CLAUDE.md                       # Agent 协作规则 (master 工作流 + 安全红线)
├── HANDOVER.md                     # 项目交接文档
├── TODO.md                         # 任务清单
├── ROADMAP.md                      # 项目路线图
├── CHANGELOG.md                    # v1.0.0 发布说明
├── scripts/
│   └── benchmark.py                # 性能基准测试 (100K×20变量 OLS 0.15s)
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
        ├── test_parser.py        # 文件解析测试
        ├── test_fitter.py        # 模型拟合 + OLS 正确性
        ├── test_results_phase2.py       # to_summary_dict, anova, latex, compare, summary_generator
        ├── test_visualization_phase2.py # scale-location, Cook's, coefficient plots, dashboard
        ├── test_exporter.py             # CSV/Excel/chart/results package export
        ├── test_phase2_remaining.py     # 文件大小限制, 类型覆盖, 数据筛选
        └── test_gallery.py             # Sample Gallery (31 tests)
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

**Sample Gallery keys** (Phase 3.5):

| Key | 类型 | 用途 |
|-----|------|------|
| `gallery_mode` | bool | 是否处于示例数据模式 |
| `gallery_item_id` | str | 当前加载的场景 ID |
| `gallery_item_title` | str | 当前加载的场景标题 |

## 测试说明

```bash
uv run python -m pytest tests/ -v              # 全部 309 tests
uv run python -m pytest tests/unit/test_XXX.py -v  # 单个文件
```

新功能需在 `tests/unit/` 下创建 `test_*.py` 测试文件，使用 `conftest.py` 中的 fixtures。

## 已知问题 / 注意事项

详细清单见 `docs/已知问题.md`（10.3 KB，涵盖编码问题、Git 换行符、kaleido/openpyxl 依赖、性能限制、9 项功能限制）。

核心提醒：
1. **__pycache__ 和 .claude/ 不要提交**: `.gitignore` 已配置
2. **编码问题**: Windows 中文终端打印 Unicode（如 R²）可能报 GBK 错误，测试中避免
3. **Pyodide 适配**: Sample Gallery 预计算结果通过 `result_json` 字段序列化，未来 Web 端无需 statsmodels OLS

## Worktree 隔离安全审计 (2026-05-25)

### 调查结论

使用 3 个 subagent 对 worktree 隔离机制进行根因排查，覆盖源码级 `_cleanup_worktree()` 逻辑、全局/项目 hooks 配置、git reflog 和历史 commit 模式。

**根因**：`_cleanup_worktree()` 的设计理念是 "agent work lives in commits, not in the working tree"。清理决策树：

```
git log --oneline HEAD --not --remotes
  ├── 有未推送 commit → 保留 worktree
  ├── 无未推送 commit → 删除 worktree + 分支
  └── 检查出错    → 假设无 commit，删除
```

未提交的修改被视为临时文件，清理时直接丢弃。

**致命发现**：过去 3 次 worktree session 中，agent 全部提交但**从未推送 worktree 分支**。工作流是"提交到 worktree → 合并到 master → 推送 master"。`_cleanup_worktree()` 检查的是 **worktree 分支**的推送状态而非 master，因此如果清理曾运行过，3 次 session **100% 会被清除**（即使代码已正确合并到 master）。

**方案评估结果**：

| 方案 | 结论 |
|------|------|
| A (PostToolUse hook 自动提交) | 每次会话 ~50s 纯开销（500+ 次 bash 进程启动），.claude/ 被 gitignore 不可共享 |
| B (纪律依赖) | 历史数据显示 worktree 分支推送率 0%，不可靠 |
| **C (直接 master)** | **已采纳** — 项目 61% 提交已用此模式，仅 1 次 fix commit |

### 当前工作流

**默认**：Agent 直接在 master 工作，不使用 worktree 隔离。

**安全机制**（详见 CLAUDE.md）：
- Agent 开始前 `git commit -am "checkpoint: pre-agent"` → 出问题 `git reset --hard HEAD~1`
- **禁止 `git push --force`**（GitHub 免费版无分支保护）
- 提交前运行 `uv run python -m pytest tests/ -v`
- 并行 agent 场景使用手动 feature 分支

### 清理记录

已清理 3 个僵尸 worktree（unlock → force remove → branch -D），零数据丢失（所有提交已存在于 master）。

## Phase 3 (Beta) 已实现功能清单

### 3.1 高级建模功能
- [x] 变量转换引擎 (对数/标准化/中心化/平方项)
- [x] 变量转换 UI (模型设定页面 expander)
- [x] 交互项创建 UI (选择两个变量 → 自动生成乘积项)
- [x] 稳健标准误选项 (HC0-HC3 + 普通标准误)
- [x] 模型控制面板添加 SE 类型选择器
- [x] 结果卡片展示转换/交互/SE 信息
- [x] 系数图 (dot-whisker) — 已在 Phase 2 实现

### 3.2 完整导出功能
- [x] LaTeX 表格导出 (Jinja2 + booktabs, 单模型 + 多模型对比, APA7 格式预设)
- [x] HTML 报告导出 (自包含, base64 内嵌诊断图, 中文界面)
- [x] SVG 图表导出 — 已在 Phase 2 实现
- [x] 分析复现包导出 (数据子集 + 配置 JSON + reproduce.py 脚本)
- [x] 导出界面集成所有新导出类型

### 3.3 数据增强
- [x] 缺失值处理策略 (删除整行 / 均值填充 / 中位数填充)
- [x] 缺失值统计分析 (按列缺失率 + 颜色标注)
- [x] 异常值检测 (IQR + Z-score 方法)
- [x] 缺失/异常处理 UI (数据探索页面 expander)

### 3.4 工程化
- [x] 会话状态持久化 (JSON 序列化, 关闭浏览器后恢复)
- [x] 崩溃恢复 (启动时检测 .session_cache + 恢复提示)
- [x] 性能优化 (缓存策略: @st.cache_data)
- [x] 3 个示例数据集 (房价/工资/空气质量, 一键加载)
- [x] 示例数据加载 UI (侧边栏)

### 3.5 Beta 验证
- [ ] Beta 测试者 ≥ 10 人
- [ ] 无 P0 Bug（数据丢失/错误结果）
- [ ] P1 Bug 关闭率 ≥ 90%
- [ ] 用户满意度 ≥ 4.0/5
- [x] 测试覆盖率: 278 测试通过 (核心模块)

## Phase 3.5 (Sample Gallery) 已实现功能

用于 Phase 4 Beta 验证招募的 onboarding 材料，5 个基于 3 个用户画像的预计算回归分析场景。

### 场景列表

| ID | Persona | n | R² | 特征 |
|----|---------|---|-----|------|
| `survey_happiness` | 张薇（社科研究生） | 400 | 0.43 | 分类 education，income↔education 共线 |
| `trust_experiment` | 张薇（社科研究生） | 200 | 0.19 | 小样本，party_member 显著 |
| `ecommerce_sales` | 陈志远（市场研究员） | 500 | 0.93 | 高 R²，ad_spend↔promotion 相关 |
| `customer_satisfaction` | 陈志远（市场研究员） | 350 | 0.77 | 2 个 4 水平分类变量 → 6 dummy |
| `policy_effect` | 李明远（政策分析师） | 300 | 0.59 | 交互项 + HC1 稳健标准误 |

### 技术特点

- **Pyodide 适配**: ModelResult 预序列化为 JSON（`result_json` 字段），未来 Web 端无需加载 statsmodels
- **JSON 往返**: `_model_result_to_json()` / `_json_to_model_result()` 无损转换
- **数据上传页集成**: expander 中卡片网格，点击按钮 → 自动注入 session_state → 跳转结果页
- **结果页/设定页**: 自动检测 gallery_mode，显示"示例数据"banner
- **用户运行自己模型后**: 自动清除 gallery_mode

## Phase 4.4 (Post-Release Tracking) 已完成

### 反馈通道

- **反馈指南**: `docs/反馈指南.md` — 包含 Bug 报告模板、功能建议模板、GitHub Issues 链接、响应时间预期、自查清单
- **Issue 模板**:
  - `.github/ISSUE_TEMPLATE/bug_report.md` — 中文 Bug 报告模板（YAML frontmatter + markdown）
  - `.github/ISSUE_TEMPLATE/feature_request.md` — 中文功能建议模板（YAML frontmatter + markdown）
- **GitHub Issues**: https://github.com/qhWangAntoneva/Regression-Analysis/issues

### v1.1 规划

- **规划文档**: `docs/v1.1_规划.md` — 按优先级组织的 12 项功能规划
- **高优先级 (★★★)**: 逻辑回归/Probit、多层次模型(MixedLM)、分析快照 — 目标 v1.1 (8-10周)
- **中优先级 (★★☆)**: 面板数据、贝叶斯回归、边际效应图、工具变量回归 — 目标 v1.2 (16-22周)
- **低优先级 (★☆☆)**: Docker 部署、中介/调节效应、正则化、Bootstrap SE、用户历史 — 目标 v1.3 (24-34周)
- 关键风险: `ModelResult` 泛型重构可能被低估，建议 v1.1 开工前独立完成

### 当前 TODO.md 状态

Phase 4.4 三项已全部标记 [x]：
- [x] 用户反馈通道建立
- [x] 问题追踪系统配置
- [x] v1.1 功能规划

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
| `sample_data.py` | `get_sample_datasets()`, `load_sample_dataset()` |
| `coefficient.py` | `coefficient_plot()`, `coefficient_plot_single()` |
| `benchmark.py` | Performance benchmarks (1K-100K rows, 10-20 vars, OLS fit + visualization timing) |
| `CHANGELOG.md` | Full v1.0.0 release notes (Phase 1-4) |
| `安全审计报告.md` | Dependency security audit (22 deps, 0 vulns) |
| `反馈指南.md` | Bug report + feature request templates, GitHub Issues link |
| `v1.1_规划.md` | 12 features across 3 priority tiers with timeline estimates |
