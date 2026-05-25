# Changelog

All notable changes to Regression Analysis will be documented in this file.

---

## [1.1.0] — 2026-05-26

### 新增功能

- **Logit 回归支持**：二元 logistic 回归模型，含 OR 值、McFadden 伪 R²、似然比检验
- **Logit 专用可视化**：ROC 曲线 + AUC、Odds Ratio 森林图
- **Logit 模型导出**：LaTeX/HTML/CSV/Excel 全格式支持，含 OR 列
- **Pyodide Web 版加载进度条**：4 阶段进度指示 + 首载提示
- **分类变量名人性化显示**：`教育水平: 本科` 替代技术格式 `C(education)[T.本科]`
- **Web 版功能对齐**：变量转换、交互项、多模型对比图在 Web 版均可用
- **Streamlit-Web 功能对照表** (README)

### 改进

- `build_variable_labels()` 重写：正确解析 patsy 列名（含交互项格式）
- Web 版系数表使用 variable_labels 渲染
- Gallery 模式 CSV 降级导出友好提示
- `compare_models()` CI 缺失时跳过 whisker 而非画在 0 处
- Web bridge `parse_file` columns 元数据透传完整性检查

### 测试

- 560 tests 全部通过 (从 v1.0 的 309 tests 增长 81%)
- 新增 Logit 引擎单元测试、Logit 图表测试、variable_labels 测试、Logit 端到端集成测试
- 跨引擎基准验证：statsmodels Logit vs GLM vs sklearn 5/5 数据集通过

---

## [1.0.0] — 2026-05-25

### Phase 1: Proof of Concept (POC)

Initial prototype validating the core idea of an interactive OLS regression tool.

**Core Engine**
- OLS regression engine via statsmodels
- Unified result data structures (CoefficientRow, ModelResult)
- R `lm()` benchmark validation (coefficient differences < 1e-10)
- Verified with 3 classic datasets: mtcars, iris, synthetic

**Minimal Prototype**
- CSV file parsing with encoding auto-detection
- Streamlit minimal UI: upload, select variables, view coefficient table
- Scatter plot with OLS regression line and confidence band

**Infrastructure**
- Python project structure (pyproject.toml, .gitignore)
- uv-based package management
- pytest test framework with fixtures
- 56 tests passing

---

### Phase 2: Minimum Viable Product (MVP)

Complete regression workflow from data import through export.

**Data Import (2.1)**
- CSV parsing with UTF-8/GBK/latin-1 automatic encoding detection
- Excel (.xlsx/.xls) parsing via openpyxl
- Data preview table component
- Variable type auto-detection: continuous, categorical, binary, ordinal, id, text
- Manual type override UI (expander + dropdown)
- Missing value statistics with color-coded display
- File size limits: >50MB warning, >200MB block

**Variable Selection (2.2)**
- Dependent variable dropdown selector
- Independent variable multi-select (auto-excludes id columns)
- Control variable grouping
- Model specification dataclass (ModelSpec) with patsy formula generation
- Descriptive statistics panel (mean, std, min, max, missing rate)
- Data subset filtering (numeric slider + categorical multi-select)
- Model control panel: intercept toggle, CI level, SE type, missing handling

**Regression Results (2.3)**
- Coefficient table with significance stars and confidence intervals
- Green highlighting for p<0.05 and p<0.01
- Model statistics grid: R-squared, Adj-R-squared, RMSE, AIC, BIC, Log-Likelihood, F-statistic, N
- ANOVA table (Type I sum of squares)
- Multi-model comparison table

**Diagnostic Charts (2.4)**
- Residuals vs fitted values plot (with LOWESS smoother)
- Normal Q-Q plot
- Scale-location plot (sqrt standardized residuals vs fitted)
- Cook's distance plot with threshold lines
- 2x2 diagnostic dashboard panel
- Statistical alerts: VIF >10 (red), VIF >5 (warning), Cook's D, Durbin-Watson
- Coefficient dot-whisker plot (single model + multi-model comparison)

**Export (2.5)**
- CSV coefficient table export
- Excel results export (openpyxl)
- Chart PNG/SVG export (kaleido, 300 DPI)
- Comprehensive report ZIP package

**User Experience (2.6)**
- Full Chinese (Simplified) interface throughout
- First-run 3-step onboarding guide (popover)
- Chinese error messages (no tracebacks shown to users)
- Inline help system (tooltips + component help parameters)
- Load example dataset button

**Testing**
- 147 tests passing
- 6 shared pytest fixtures (sample_df, sample_csv_*, etc.)

---

### Phase 3: Beta

Advanced features, full export pipeline, data enhancement, and engineering improvements.

**Advanced Modeling (3.1)**
- Variable transformation engine: log, standardize (z-score), center, square
- Variable transformation UI (expander in model specification page)
- Interaction term creation UI (select two variables -> auto-generate product)
- Robust standard error options: HC0, HC1, HC2, HC3
- SE type selector in model control panel
- Result cards display transform/interaction/SE metadata
- Coefficient dot-whisker plot (from Phase 2, extended for multi-model)

**Full Export Pipeline (3.2)**
- LaTeX table export via Jinja2 + booktabs (single model + multi-model comparison)
- APA7 format preset for LaTeX tables
- HTML report export (self-contained, base64-embedded diagnostic charts, Chinese)
- SVG chart export (from Phase 2)
- Reproducibility package export (data subset + config JSON + reproduce.py script)
- Integrated export dialog with format/path/content selection

**Data Enhancement (3.3)**
- Missing value handling strategies: drop row, mean imputation, median imputation
- Per-column missing rate analysis with color annotation
- Outlier detection: IQR method + Z-score method
- Missing/outlier handling UI (expander in data exploration page)

**Engineering (3.4)**
- Session state persistence (JSON serialization, survives browser close)
- Crash recovery (auto-detect .session_cache on startup, recovery prompt)
- Performance optimization (@st.cache_data caching strategy)
- 3 built-in example datasets: housing prices, salary survey, air quality
- One-click example data loading via sidebar button

**Testing**
- 278 tests passing (core modules)

---

### Phase 3.5: Sample Gallery

Onboarding material for Phase 4 Beta validation recruitment; 5 pre-computed
regression scenarios based on 3 user personas.

**Scenarios**

| ID | Persona | n | R-squared | Key Features |
|----|---------|---|-----------|--------------|
| `survey_happiness` | Zhang Wei (social science grad) | 400 | 0.43 | Categorical education, income correlated with education |
| `trust_experiment` | Zhang Wei (social science grad) | 200 | 0.19 | Small sample, party_member significant |
| `ecommerce_sales` | Chen Zhiyuan (market researcher) | 500 | 0.93 | High R-squared, ad_spend correlated with promotion |
| `customer_satisfaction` | Chen Zhiyuan (market researcher) | 350 | 0.77 | 2 variables with 4 levels each -> 6 dummies |
| `policy_effect` | Li Mingyuan (policy analyst) | 300 | 0.59 | Interaction term + HC1 robust SE |

**Technical Features**
- Pyodide-ready: ModelResult pre-serialized as JSON for future web deployment
- Lossless JSON round-trip: `_model_result_to_json()` / `_json_to_model_result()`
- Gallery card grid integrated into data upload page (expander)
- One-click load injects session_state and navigates to results page
- Auto-detection of gallery_mode with "Sample Data" banner
- Auto-clears gallery_mode when user runs their own model

**Testing**
- +31 tests (309 total)

---

### Phase 4: v1.0 Release

**4.1 UI Polish**
- Comprehensive visual consistency review
- Color-blind friendly palette support
- Responsive layout (1366x768+)
- Keyboard navigation improvements
- Visual regression validation across components

**4.2 Documentation**
- User manual with 3 complete case studies (`docs/用户手册.md`)
- Developer guide with architecture overview and extension guide (`docs/开发者指南.md`)
- Known issues list (`docs/已知问题.md`)

**4.3 Release Preparation**
- Performance benchmark suite (`scripts/benchmark.py`)
  - Tests sizes: 1K, 10K, 50K, 100K rows x 10, 20 variables
  - Measures: file parsing, type detection, OLS fit, coefficient extraction, visualization
  - Target: 100K rows x 20 variables OLS fit <= 3.0 seconds
- Dependency security audit (`docs/安全审计报告.md`)
  - All 22 core runtime dependencies reviewed
  - Zero critical, high, medium, or low severity vulnerabilities found
- CHANGELOG.md (this file)

---

## Future Versions

### [1.2.0] — Planned

- **Additional Models**: probit regression, mixed-effects models, panel data FE/RE
- **User Experience**: analysis snapshots (JSON save/load), user history management
- **Deployment**: Docker support
- **Advanced Analysis**: mediation/moderation analysis, instrumental variable regression, bootstrap SE
- **Machine Learning**: ridge regression, lasso, elastic net
- **Reporting**: automated analysis summary generation (LLM-assisted or template-based)
