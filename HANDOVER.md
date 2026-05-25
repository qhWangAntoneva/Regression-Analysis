# Regression Analysis — 交接文档

> 最后更新: 2026-05-26 (Session 2)
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 分支: master
> 当前提交: 0934bf5 (Phase 5.4: v1.1.0 发布)
> 上次交接: 4a17a18 (Phase 5.2 UX 改进完成)
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
| Phase 4.5 (Web + 测试补充 + Bug修复) | 完成 | 549 tests |
| Phase 5.0 (Web 功能补齐 + 覆盖率 + _norm_ppf) | 完成 | 471 tests |
| Phase 5.1 (Logit 回归 — 需求 4 补全) | 完成 | 549 tests |
| **Phase 5.2 (UX 改进)** | **完成** | **560 tests** |
| **Phase 5.3 (Web-Streamlit 对齐)** | **完成** | **560 tests** |
| **Phase 5.4 (v1.1 发布)** | **完成** | **599 tests** |
| **合计** | — | **599 tests** |

### Phase 5.2 进度 (2026-05-25 本 session)

| 子阶段 | 状态 | 说明 |
|--------|------|------|
| 5.2.1 Pyodide 加载进度条 | 完成 | 4 阶段进度指示器 (下载→安装→导入→就绪)，首载提示 |
| 5.2.2 分类变量名人性化 | 完成 | `build_variable_labels()` 解析 patsy 列名 → `变量: 水平` |
| 5.2.3 Streamlit-Web 功能对照表 | 完成 | README.md 23 行对照表 + 双版本互链 |
| Bug 修复 | 完成 | Web 前端消费 variable_labels (BLOCKER) |
| build_variable_labels 修复 | 完成 | 发现并修复 regex NO-OP bug (3-agent 分析 + fixer) |

**Session 工作流**: Worker×3 并行实现 → Reviewer 审核 → 3-agent 分析团队审计 bug → Fixer 修复 → Push

### build_variable_labels 修复详情

三 agent 联合分析发现：
1. **regex 期望 `C()` 前缀但 `build_formula()` 从不加** — 函数在 production 中是 NO-OP
2. **patsy 两种括号格式**: `[T.level]`(主效应) 和 `[level]`(交互内)，原 regex 只匹配前者
3. **`$` 锚点阻止交互列匹配** — `cat[T.b]:x` 全部落入 raw fallback

修复: `split(":")` 拆分 → 逐 part 匹配 `(\w+)\[T?\.?([^\]]+)\]` → `×` 重拼。11 个新测试。

---

## 项目结构

```
Regression Analysis/
├── CLAUDE.md                       # Agent 协作规则 (master 工作流 + 安全红线)
├── HANDOVER.md                     # 项目交接文档
├── TODO.md                         # 任务清单
├── ROADMAP.md                      # 项目路线图
├── CHANGELOG.md                    # v1.0.0 发布说明
├── README.md                       # 项目 README (含 Streamlit vs Web 功能对照表)
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
├── web/                            # Pyodide 静态 Web 应用
│   ├── index.html                  # 5-tab UI + 进度条 UI
│   ├── css/styles.css              # 响应式设计 + CSS 变量 + 进度条样式
│   ├── js/
│   │   ├── app.js                  # 前端逻辑 (含进度条 + variable_labels 消费)
│   │   └── gallery_data.js         # 5 个预计算场景 (322KB JSON, 自动生成)
│   ├── py/
│   │   └── bridge.py               # Pyodide 内运行: 文件解析/OLS/logit/诊断/图表/导出
│   └── deploy.sh                   # 跨仓库部署 → qhWangAntoneva.github.io
├── app/
│   ├── app.py                    # Streamlit 主入口 (st.navigation)
│   ├── config.py                 # Streamlit 页面配置
│   ├── components/
│   │   ├── data_table.py         # 数据预览 + 变量信息表 + 类型覆盖UI
│   │   ├── variable_selector.py  # 因变量/自变量选择器
│   │   ├── model_control.py      # 模型控制面板 + 多模型对比控件
│   │   ├── result_card.py        # 系数表/统计量/ANOVA/对比表/警示 (支持 variable_labels)
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
│   │   ├── specification.py      # ModelSpec, build_formula, build_design_matrix, build_variable_labels
│   │   ├── fitter.py             # ModelFitter (fit/fit_multiple)
│   │   ├── diagnostics.py        # VIF, residual_tests, influence_stats
│   │   └── engines/
│   │       ├── statsmodels_engine.py      # OLS 适配器 (传递 variable_labels)
│   │       └── statsmodels_logit_engine.py # Logit 引擎 (返回 tuple: fitted, labels)
│   ├── preprocessing/
│   │   └── type_detector.py      # VariableInfo + VariableTypeDetector
│   ├── results/
│   │   ├── table.py              # CoefficientRow + ModelResult (含 variable_labels 字段)
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
        ├── test_logit_engine.py  # Logit 引擎测试 (run_logit 返回 tuple)
        ├── test_results_phase2.py       # to_summary_dict, anova, latex, compare, summary_generator
        ├── test_visualization_phase2.py # scale-location, Cook's, coefficient plots, dashboard
        ├── test_exporter.py             # CSV/Excel/chart/results package export
        ├── test_phase2_remaining.py     # 文件大小限制, 类型覆盖, 数据筛选
        ├── test_gallery.py             # Sample Gallery (31 tests)
        ├── test_sample_data.py         # 3 示例数据集 (26 tests)
        ├── test_scatter.py             # scatter_with_regression (13 tests)
        ├── test_variable_labels.py     # build_variable_labels (11 tests)
        └── test_logit_plots.py         # ROC + OR 图 (6 tests)
```

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
| 统计引擎 | statsmodels (OLS + Logit) |
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
3. **Web bridge dtype 往返**: `parse_file` 返回的 `columns` 元数据必须随 `data` 一起传给 `run_regression`，否则 dtype 丢失
4. **statsmodels 0.14.6**: `fitted.params/bse/tvalues` 返回 numpy array，不是 pandas Series

### Phase 5.2 已知 Minor Issues (非阻塞)

1. `build_variable_labels` 的 `spec` 参数未被使用（保留以备后用）
2. Web bridge 不支持 categorical×numeric 交互（`pd.get_dummies` 路径，非 patsy）
3. `_build_variable_labels_for_web()` 与 `build_variable_labels()` 是独立实现——列名格式不同（`var_level` vs `var[T.level]`）
4. `ModelSpec.interaction_terms` 仅支持 2-way pairs，3-way 交互不可表示

---

## 测试说明

```bash
uv run python -m pytest tests/ -v              # 全部 560 tests
uv run python -m pytest tests/unit/test_XXX.py -v  # 单个文件
```

新功能需在 `tests/unit/` 下创建 `test_*.py` 测试文件，使用 `conftest.py` 中的 fixtures。

---

## 启动开发

```bash
cd "C:/Users/lenovos/Regression Analysis"
uv run streamlit run app/app.py        # 启动 Streamlit 应用
uv run python -m pytest tests/ -v      # 运行测试 (560)
bash web/deploy.sh                      # 部署 Web 版到 GitHub Pages
```

---

## Phase 5.2 Code Review 修复 (2026-05-25)

| # | 严重度 | 问题 | 修复 |
|---|--------|------|------|
| 1 | BLOCKER | Web 前端未消费 variable_labels | renderCoefficientTable 接受 variable_labels 参数 |
| 2 | HIGH | build_variable_labels regex 期望 C() 前缀但 build_formula() 不生成 | 重写为 split(":") + 匹配 (\w+)\[T?\.?([^\]]+)\] |
| 3 | LOW | spec 参数未使用 | 保留（docstring 注明） |

### Phase 5.2 提交历史

```
4a17a18 fix: build_variable_labels regex matches actual patsy output (no C() prefix, handle interactions)
c5268b9 fix(web): consume variable_labels in coefficient table rendering
d432015 fix: update test_logit_fit_success for run_logit tuple return
```

Phase 5.2 主体提交 (通过 merge 入 master):
- `1ef7a87` feat(web): add Pyodide loading progress bar (Phase 5.2.1)
- `340f844` docs: add Streamlit vs Web feature comparison table (Phase 5.2.3)
- 5.2.2 源码（category labels）随 5.2.3 分支合并

---

## 下个 Session 建议 (v1.2+)

1. **Probit 回归**: logit 引擎就绪后，仅需新增 GLM family=binomial(probit) 包装
2. **多层次模型**: 混合效应模型 (MixedLM / lme4-backend)
3. **面板数据固定/随机效应** (linearmodels)
4. **Poisson / NegativeBinomial 回归** (计数因变量)
5. **Web bridge categorical 交互**: `_build_variable_labels_for_web()` 路径需要处理 categorical 变量参与交互的情况
6. **项目架构现代化**: 考虑 Docker 部署 + 自动化 CI/CD
