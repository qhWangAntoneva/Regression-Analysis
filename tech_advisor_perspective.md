# Regression Analysis — 技术架构设计草案

> **作者角色**: 技术顾问 Agent
> **目标**: 为社会科学研究工具提供完整的技术架构设计
> **状态**: 草案 v1.0

---

## 目录

1. [技术栈推荐](#1-技术栈推荐)
2. [系统架构](#2-系统架构)
3. [目录结构](#3-目录结构)
4. [关键技术挑战与方案](#4-关键技术挑战与方案)
5. [附录：可选增强路线](#5-附录可选增强路线)

---

## 1. 技术栈推荐

### 1.1 核心决策：框架选择

| 框架 | 适合度 | 理由 |
|------|--------|------|
| **Streamlit** | **强烈推荐** | Python 全栈、零前端门槛、天然支持 pandas/plotly/statsmodels 生态、快速原型到生产、社交科学校验工作流匹配度最高 |
| Shiny for Python | 可选 | 更复杂的 UI 控制能力，但学习曲线陡峭、社区较小 |
| Django + HTMX | 不推荐 | 过度工程化，单用户/小组工具无需全栈框架 |
| FastAPI + React | 不推荐 | 前后端分离增加复杂性，对社会科学研究者不友好 |

**结论**: 以 **Streamlit** 为应用框架，后端逻辑全部由 Python 承担。

### 1.2 统计引擎

| 库 | 用途 | 优先级 |
|----|------|--------|
| **statsmodels** | OLS, Logit, Probit, Poisson, NegativeBinomial, MixedLM（多层模型）, GLM, GEE | 核心引擎 |
| **linearmodels** | 面板数据模型（固定效应、随机效应、IV） | 重要补充 |
| **scikit-learn** | 岭回归、Lasso、弹性网、数据预处理流水线 | 辅助 |
| **PyMC** | 贝叶斯回归（可选增强） | 低优先级 |
| **scipy.stats** | 描述统计、假设检验工具 | 基础依赖 |

### 1.3 前端可视化

| 库 | 用途 | 优先级 |
|----|------|--------|
| **plotly** | 交互式散点图、残差图、系数图（缩放、悬停数据点） | 首选 |
| **matplotlib** | 高质量静态图表（导出 PDF/PNG 用） | 后端渲染 |
| **seaborn** | 快速探索性图表（分布图、配对图） | 辅助 |
| **altair** | 声明式 Vega-Lite 图表（Streamlit 原生支持好） | 可选 |
| **streamlit-aggrid** | 交互式数据表格（变量选择、数据浏览） | 推荐 |

### 1.4 导出与格式化

| 库 | 用途 |
|----|------|
| **pandas** (DataFrame → HTML/CSV/Excel) | 表格导出 |
| **jinja2 + custom LaTeX template** | LaTeX 回归表导出（类 stargazer 风格） |
| **python-docx** | Word 文档导出（含表格和图表） |
| **weasyprint / fpdf2** | PDF 导出 |
| **matplotlib** | PNG/SVG/PDF 静态图表导出 |
| **kaleido** | plotly 静态图导出（PNG/SVG/PDF） |

### 1.5 数据处理与辅助

| 库 | 用途 |
|----|------|
| **pandas** | 数据处理核心（>=2.0 版 PyArrow 后端支持） |
| **numpy** | 数值计算 |
| **polars** | 大型数据集替代引擎（可选，>500MB 场景） |
| **pyreadstat** | 读取 SAS/SPSS/Stata 格式（社会科学数据交换） |
| **openpyxl / xlsxwriter** | Excel 读写 |
| **pydantic** | 模型规格定义、配置验证 |
| **loguru** | 结构化日志 |

### 1.6 技术栈总览

```
┌───────────────────────────────────────────────────┐
│                  Streamlit UI                      │
├───────────────────────────────────────────────────┤
│  Plotly │ Matplotlib │ st-aggrid │ Altair          │
├───────────────────────────────────────────────────┤
│  statsmodels │ linearmodels │ scikit-learn │ PyMC  │
├───────────────────────────────────────────────────┤
│  pandas │ polars │ pyreadstat │ numpy              │
├───────────────────────────────────────────────────┤
│  jinja2 │ python-docx │ weasyprint │ kaleido       │
└───────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 分层架构

系统分为四层，每层职责清晰、单向依赖：

```
┌──────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                  │
│  ┌────────────────────────────────────────────────┐   │
│  │  app.py               Streamlit 主入口          │   │
│  │  pages/               多页界面的各功能页         │   │
│  │  components/          可复用 UI 组件            │   │
│  └──────────────┬─────────────────────────────────┘   │
└─────────────────┼────────────────────────────────────┘
                  │ 调用
┌─────────────────┼────────────────────────────────────┐
│                 ▼                                    │
│              APPLICATION LAYER                        │
│  ┌────────────────────────────────────────────────┐   │
│  │  Session State Manager   状态管理              │   │
│  │  Workflow Orchestrator   工作流编排            │   │
│  │  Cache Manager           缓存管理              │   │
│  └──────────────┬─────────────────────────────────┘   │
└─────────────────┼────────────────────────────────────┘
                  │ 调用
┌─────────────────┼────────────────────────────────────┐
│                 ▼                                    │
│              BUSINESS LOGIC LAYER                     │
│  ┌────────────────────────────────────────────────┐   │
│  │  data_io/          文件解析、导入导出           │   │
│  │  preprocessing/    数据清洗、变量检测、转换     │   │
│  │  modeling/         模型规格、拟合、诊断         │   │
│  │  comparison/       多模型比较                   │   │
│  │  export/           格式生成（LaTeX/Word/HTML） │   │
│  └──────────────┬─────────────────────────────────┘   │
└─────────────────┼────────────────────────────────────┘
                  │ 通过
┌─────────────────┼────────────────────────────────────┐
│                 ▼                                    │
│              DATA LAYER                               │
│  ┌────────────────────────────────────────────────┐   │
│  │  File System          上传文件、缓存、导出文件  │   │
│  │  Session Cache        中间计算结果缓存          │   │
│  │  Config               用户配置持久化            │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

### 2.2 数据流图

```
用户上传
  │
  ▼
┌──────────────────────┐
│  1. 文件导入           │
│  ┌──────────────────┐  │
│  │ CSV │ Excel │    │  │
│  │ SAS/SPSS/Stata   │  │
│  └────────┬─────────┘  │
│  encoding 自动检测     │
│  大文件抽样预览        │
└──────────┬─────────────┘
           ▼
┌──────────────────────┐
│  2. 数据预览与清洗     │
│  ┌──────────────────┐  │
│  │ 数据表格预览      │  │
│  │ 变量类型标注      │  │
│  │   - 连续          │  │
│  │   - 分类          │  │
│  │   - 有序分类      │  │
│  │   - 二值          │  │
│  │ 缺失值统计        │  │
│  │ 行/列筛选         │  │
│  └────────┬─────────┘  │
└──────────┬─────────────┘
           ▼
┌──────────────────────┐
│  3. 变量选择           │
│  ┌──────────────────┐  │
│  │ 因变量 (Y)        │  │
│  │ 自变量 (X)        │  │
│  │ 分组变量          │  │
│  │ 权重变量          │  │
│  │ 聚类标准误变量    │  │
│  └────────┬─────────┘  │
└──────────┬─────────────┘
           ▼
┌──────────────────────┐
│  4. 模型规格           │
│  ┌──────────────────┐  │
│  │ 模型类型:         │  │
│  │  OLS / Logit /   │  │
│  │  Probit / Poisson │  │
│  │  MixedLM / GLM   │  │
│  │ 面板固定/随机效应  │  │
│  │                  │  │
│  │ 高级选项:         │  │
│  │  稳健标准误       │  │
│  │  聚类标准误       │  │
│  │  交互项           │  │
│  │  多项式项         │  │
│  │  固定效应          │  │
│  └────────┬─────────┘  │
└──────────┬─────────────┘
           ▼
┌──────────────────────┐
│  5. 模型拟合           │
│  ┌──────────────────┐  │
│  │ statsmodels 拟合  │  │
│  │ 收敛检查          │  │
│  │ 诊断统计量计算    │  │
│  └────────┬─────────┘  │
└──────────┬─────────────┘
           ▼
┌──────────────────────────────────┐
│  6. 结果展示                      │
│  ┌────────────┬──────────────┐   │
│  │ 回归系数表   │ 模型诊断      │   │
│  │ ─────────── │ ───────────  │   │
│  │ 系数估计     │ R² / Adj-R²  │   │
│  │ 标准误       │ F-statistic  │   │
│  │ t/z 值      │ Log-Likelihood│   │
│  │ p 值        │ AIC / BIC    │   │
│  │ 置信区间     │ 残差检验     │   │
│  │ 显著性标记   │ 多重共线性   │   │
│  ├────────────┴──────────────┤   │
│  │ 图表:                       │   │
│  │  散点图 + 回归线            │   │
│  │  残差图 (fitted vs resid)   │   │
│  │  Q-Q 图                    │   │
│  │  系数图 (dot-whisker)      │   │
│  │  边际效应图                 │   │
│  └──────────────────────────────┘ │
└──────────────────┬─────────────────┘
                   ▼
┌──────────────────────┐
│  7. 导出               │
│  ┌──────────────────┐  │
│  │ LaTeX 表格文件    │  │
│  │ Word (.docx)      │  │
│  │ HTML 报告         │  │
│  │ PDF 报告          │  │
│  │ PNG/SVG 图表      │  │
│  │ CSV 系数表        │  │
│  └──────────────────┘  │
└─────────────────────────┘
```

### 2.3 关键模块划分

| 模块 | 职责 | 关键类/函数 |
|------|------|-------------|
| `data_io.parser` | 多格式文件解析、编码检测、大文件分块 | `FileParser`, `detect_encoding()`, `sample_data()` |
| `data_io.validator` | 数据完整性校验、变量命名规范 | `validate_column_names()`, `check_missing()` |
| `preprocessing.type_detector` | 变量类型自动检测 | `VariableTypeDetector`, `detect_types()` |
| `preprocessing.transformer` | 变量转换（中心化、标准化、对数化、哑变量） | `VariableTransformer` |
| `modeling.specification` | 模型规格构建 | `ModelSpec`, `build_formula()`, `build_design_matrix()` |
| `modeling.fitter` | 模型拟合调度 | `ModelFitter`, `fit_ols()`, `fit_logit()`, `fit_mixedlm()` |
| `modeling.diagnostics` | 模型诊断 | `residual_tests()`, `multicollinearity()`, `influence()` |
| `results.table` | 回归表格式化 | `RegressionTable`, `to_dataframe()`, `to_latex()` |
| `results.comparison` | 多模型比较 | `ModelComparison`, `side_by_side_table()` |
| `visualization.plots` | 图表生成 | `scatter_plot()`, `residual_plot()`, `coefficient_plot()`, `marginal_effects_plot()` |
| `export.latex` | LaTeX 导出 | `LatexExporter`, `stargazer_style()` |
| `export.word` | Word 导出 | `WordExporter` |
| `export.report` | HTML/PDF 综合报告 | `ReportGenerator` |
| `session.state` | Streamlit 会话状态管理 | `SessionState`, `get()`, `set()`, `clear()` |
| `config.settings` | 全局配置 | `Settings`, `load()`, `save()` |

---

## 3. 目录结构

```
regression-analysis/
│
├── app/                            # Streamlit 应用
│   ├── __init__.py
│   ├── app.py                      # 主入口 (streamlit run)
│   ├── config.py                   # Streamlit 配置（页面标题、主题等）
│   │
│   ├── pages/                      # 多页面
│   │   ├── __init__.py
│   │   ├── 01_data_upload.py       # 数据上传与预览
│   │   ├── 02_data_explore.py      # 数据探索与变量类型标注
│   │   ├── 03_model_spec.py        # 模型规格设定
│   │   ├── 04_model_results.py     # 模型结果展示
│   │   ├── 05_model_compare.py     # 多模型比较
│   │   └── 06_export.py            # 导出设置
│   │
│   ├── components/                 # 可复用 Streamlit UI 组件
│   │   ├── __init__.py
│   │   ├── data_table.py           # 交互式数据表格
│   │   ├── variable_selector.py    # 变量选择控件
│   │   ├── model_control.py        # 模型参数面板
│   │   ├── result_card.py          # 结果展示卡片
│   │   └── export_dialog.py        # 导出对话框
│   │
│   └── assets/                     # 静态资源
│       ├── css/
│       │   └── custom.css
│       └── templates/              # Streamlit 自定义模板
│
├── src/                            # 核心业务逻辑库
│   ├── __init__.py
│   │
│   ├── data_io/                    # 数据导入导出
│   │   ├── __init__.py
│   │   ├── parser.py               # 文件解析（CSV/Excel/SAS/SPSS/Stata）
│   │   ├── encoding.py             # 编码检测
│   │   └── exporter.py             # 数据导出
│   │
│   ├── preprocessing/              # 预处理
│   │   ├── __init__.py
│   │   ├── type_detector.py        # 变量类型自动检测
│   │   ├── cleaner.py              # 清洗（缺失值处理、异常值检测）
│   │   ├── transformer.py          # 变量转换
│   │   └── pipeline.py             # 预处理流水线
│   │
│   ├── modeling/                   # 建模
│   │   ├── __init__.py
│   │   ├── specification.py        # 模型规格
│   │   ├── fitter.py               # 模型拟合（调度不同引擎）
│   │   ├── diagnostics.py          # 模型诊断
│   │   ├── comparison.py           # 多模型比较
│   │   └── engines/                # 各统计引擎适配器
│   │       ├── __init__.py
│   │       ├── statsmodels_engine.py
│   │       ├── linearmodels_engine.py
│   │       ├── sklearn_engine.py
│   │       └── pymc_engine.py      # 可选
│   │
│   ├── results/                    # 结果处理
│   │   ├── __init__.py
│   │   ├── table.py                # 系数表构建
│   │   ├── statistics.py           # 模型拟合统计量
│   │   └── comparison.py           # 对比表
│   │
│   ├── visualization/              # 可视化
│   │   ├── __init__.py
│   │   ├── scatter.py              # 散点图 + 回归线
│   │   ├── residual.py             # 残差诊断图
│   │   ├── coefficient.py          # 系数图 (dot-whisker)
│   │   ├── marginal.py             # 边际效应图
│   │   ├── diagnostics.py          # 诊断图（Q-Q, 异方差等）
│   │   └── themes.py               # 图形主题配置
│   │
│   ├── export/                     # 导出
│   │   ├── __init__.py
│   │   ├── latex.py                # LaTeX 表格生成
│   │   ├── word.py                 # Word 导出
│   │   ├── pdf.py                  # PDF 导出
│   │   ├── html_report.py          # HTML 报告
│   │   └── templates/              # 导出模板
│   │       ├── regression_table.tex.j2
│   │       ├── report.html.j2
│   │       └── comparison_table.tex.j2
│   │
│   └── utils/                      # 工具函数
│       ├── __init__.py
│       ├── logger.py               # 日志配置
│       ├── exceptions.py           # 自定义异常
│       ├── profiling.py            # 性能分析工具
│       └── decorators.py           # 通用装饰器（计时、缓存等）
│
├── tests/                          # 测试
│   ├── __init__.py
│   ├── conftest.py                 # pytest 全局 fixture
│   │
│   ├── unit/                       # 单元测试
│   │   ├── __init__.py
│   │   ├── test_parser.py
│   │   ├── test_type_detector.py
│   │   ├── test_specification.py
│   │   ├── test_fitter.py
│   │   ├── test_table.py
│   │   ├── test_diagnostics.py
│   │   ├── test_comparison.py
│   │   ├── test_latex.py
│   │   └── test_transformer.py
│   │
│   ├── integration/                # 集成测试
│   │   ├── __init__.py
│   │   ├── test_full_workflow.py   # 完整端到端流程
│   │   ├── test_model_engines.py   # 各引擎适配器
│   │   └── test_export_formats.py  # 所有导出格式
│   │
│   └── fixtures/                   # 测试数据
│       ├── sample_ols.csv
│       ├── sample_logit.csv
│       ├── sample_panel.csv
│       ├── sample_missing.csv
│       └── sample_types.xlsx
│
├── docs/                           # 文档
│   ├── user_guide.md               # 用户指南
│   ├── developer_guide.md          # 开发者指南
│   ├── api_reference.md            # API 参考
│   └── examples/                   # 示例
│       ├── example_ols.py
│       └── example_panel.py
│
├── notebooks/                      # Jupyter 探索性分析
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_export_format_test.ipynb
│
├── scripts/                        # 辅助脚本
│   ├── generate_test_data.py       # 生成测试数据
│   ├── benchmark.py                # 性能基准测试
│   └── setup_venv.sh               # 环境设置脚本
│
├── pyproject.toml                  # 项目元数据与依赖
├── README.md                       # 项目说明
├── CLAUDE.md                       # AI 辅助开发指令
├── .pre-commit-config.yaml         # pre-commit 钩子
├── .gitignore
└── tech_advisor_perspective.md     # 本文档
```

---

## 4. 关键技术挑战与方案

### 4.1 大型数据集处理

**挑战**: 用户上传 CSV 可能达到数百 MB 甚至 GB 级别，直接 `pd.read_csv()` 会耗尽内存；Streamlit 的响应式刷新机制处理大文件时会反复重算导致性能崩溃。

**方案**:

1. **分层读取策略**:
   - 预览阶段: 仅读取前 1000 行 (`nrows=1000`) 供类型检测和变量选择
   - 拟合阶段: 按需读取全部数据，配合 `dtype` 参数优化内存
   - 大文件后备: 检测文件大小 >100MB 时提示使用 Polars 后端或分块处理

2. **内存优化**:
   - 启用 `pandas >= 2.0` 的 PyArrow 后端 (`mode.dtype_backend='pyarrow'`)
   - 检测并优化数据类型（`float64` → `float32`, `int64` → `int32`）
   - 及时释放中间结果（`del variable`, `gc.collect()`）

3. **缓存策略**:
   ```python
   @st.cache_data(ttl=3600, max_entries=10)
   def load_data(filepath: str) -> pd.DataFrame:
       return pd.read_csv(filepath, nrows=1000)  # 预览用小
   ```

4. **惰性加载**: 模型拟合时才读取完整数据，不提前全量加载。

### 4.2 变量类型自动检测

**挑战**: 社会科学数据中的变量类型复杂——数值可能编码为字符串（如 "1", "2" 表示分类）、分类变量可能只有几个唯一值、ID 列看起来像数值但实际上不是分析变量。

**方案**:

```python
class VariableTypeDetector:
    """
    启发式变量类型检测规则
    """
    CATEGORICAL_CARDINALITY_RATIO = 0.05   # 唯一值 < 总行数 5% → 分类
    CATEGORICAL_MAX_UNIQUE = 50            # 唯一值 ≤ 50 → 分类
    ORDINAL_THRESHOLD = 12                 # 有序分类唯一值 ≤ 12
    ID_PATTERN = re.compile(r'^(id|code|key|num|no)[_\s#]?', re.I)
    
    def detect(self, series: pd.Series, n_rows: int) -> VariableType:
        1. 跳过 ID 列（列名匹配 ID_PATTERN + 唯一值 == n_rows）
        2. object 类型 → 尝试 pd.to_numeric 转换
        3. 布尔类型 → BINARY
        4. 唯一值 / n_rows < CATEGORICAL_CARDINALITY_RATIO:
           - 唯一值 ≤ 2 → BINARY
           - 唯一值 ≤ CATEGORICAL_MAX_UNIQUE → CATEGORICAL
           - 数值且唯一值 ≤ ORDINAL_THRESHOLD → ORDINAL
        5. 其余 → CONTINUOUS
```

用户始终可以在 UI 中手动覆盖自动检测结果。

### 4.3 模型结果格式化输出

**挑战**: `statsmodels` 原生 `summary()` 输出文本格式难以在 Web UI 中友好展示；不同模型类型（OLS vs Logit vs MixedLM）的统计量结构不同。

**方案**:

构建统一的结果提取层 `RegressionTable`，将不同模型的结果转换为标准结构：

```python
@dataclass
class CoefficientRow:
    name: str
    coef: float
    se: float
    t_or_z: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    significance: str  # '***', '**', '*', ''

@dataclass
class ModelResult:
    model_type: str
    coefficients: list[CoefficientRow]
    n_obs: int
    r_squared: float | None
    adj_r_squared: float | None
    log_likelihood: float | None
    aic: float
    bic: float
    f_statistic: tuple[float, float] | None  # (stat, pvalue)
    dep_var: str
    specification: str                     # 公式字符串
```

对每种模型类型实现适配器 `extract_statsmodels()` / `extract_linearmodels()` 方法，统一输出 `ModelResult`。

然后在 UI 渲染时：
- `st.dataframe()` 或 `st-aggrid` 显示系数表
- 可选的显著性星标着色
- 底部显示模型统计量卡片

### 4.4 多模型比较

**挑战**: 研究者经常需要比较多个模型规格（逐步添加变量、替换核心自变量）的系数稳定性。

**方案**:

1. **侧边系数表** (side-by-side):
   - 列 = 不同模型
   - 行 = 自变量
   - 单元格 = 系数(标准误) + 显著性标记
   - 底部合并模型统计量行（N, R², AIC, BIC）
   - 参考《政治学研究方法》通用标准格式

2. **系数图** (dot-whisker plot):
   - 用 plotly 绘制每个系数的点估计 + 置信区间
   - 不同模型用不同颜色区分
   - 可滚动的长图支持

3. **模型拟合统计量比较表**:
   - R² / Adj-R²
   - AIC / BIC
   - Log-Likelihood
   - F 检验 / LR 检验
   - RMSE

4. **实现**:
   - `ModelComparison` 类接收 `list[ModelResult]`
   - 自动对齐变量名（跨模型变量名不匹配时保留各模型特有变量）
   - 输出 `pd.DataFrame`（宽格式）供 UI 渲染和导出

### 4.5 导出格式（LaTeX, HTML, Word, PDF, PNG）

**挑战**: 每种导出格式需要完全不同的技术方案；回归表需要支持多模型并排、显著性标记、模型统计量脚注等学术出版标准。

**方案**:

| 格式 | 技术方案 | 关键考虑 |
|------|----------|---------|
| **LaTeX** | Jinja2 模板生成 `.tex`；支持 `booktabs`、`threeparttable`、`siunitx` 等宏包 | 需生成可直接编译的完整文件（含 `\documentclass` 或仅 `tabular` 片段）；参考 `stargazer` 输出风格 |
| **HTML** | `pd.DataFrame.to_html()` + 自定义 CSS 样式类 | 嵌入 Bootstrap/自定义 CSS 生成可打印版本的 HTML 报告 |
| **Word** | `python-docx` 逐行构建表格 | 回归表转为 `python-docx` Table 对象；图表嵌入为 `InlineShape`；需处理页宽、字体、表格样式 |
| **PDF** | 方案 A: `weasyprint` 渲染 HTML→PDF（推荐）；方案 B: matplotlib 完整渲染 | `weasyprint` 支持 CSS `@page` 控制分页和页边距，适合学术规格 |
| **PNG/SVG** | `matplotlib` 直接保存 / `plotly.io.write_image()` + `kaleido` | 图表导出；PNG 用 300dpi 满足出版要求 |

**导出策略优先级** (按开发复杂度排序):
1. **HTML + CSV** — 最易实现，即得即用
2. **LaTeX** — 学术发表刚需，Jinja2 模板成熟
3. **PNG** — 图表保存，matplotlib 原生支持
4. **Word** — 社会科学校对/分享常用格式
5. **PDF** — 综合报告，最复杂，最后实现

**示例：LaTeX 导出模板结构**:

```latex
% 单模型输出
\begin{table}[!htbp]
\centering
\caption{回归结果：{{ dep_var }}}
\label{tab:regression}
\begin{threeparttable}
\begin{tabular}{l@{\hspace{1.5em}}c}
\toprule
变量 & 模型 1 \\
\midrule
{% for coef in coefficients %}
{{ coef.name }} & {{ coef.estimate }}{{ coef.stars }} \\
 & ({{ coef.se }}) \\
{% endfor %}
\midrule
N & {{ n_obs }} \\
R² & {{ r_squared }} \\
AIC & {{ aic }} \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item \textit{注}: 括号内为标准误。
\item {{ significance_legend }}
\end{tablenotes}
\end{threeparttable}
\end{table}
```

### 4.6 其他挑战

#### 多重比较与 P 值校正
- 当用户运行多个模型时，提供 Bonferroni / Holm / FDR 校正选项
- 在校正后的模型结果中自动标注 adjusted p-value

#### 缺失值处理策略
- 默认 `listwise deletion`（社会科学标准）
- 可选: 均值填充 / 中位数填充 / 多重插补
- 在结果脚注中明确报告缺失值处理方式

#### 交互项与多项式
- UI 层面提供"添加交互项"的便捷界面（下拉选择两个变量→自动生成乘积项）
- 自动中心化交互项中的连续变量（降低多重共线性）

#### 聚类标准误
- 支持 `cluster` 参数（statsmodels 和 linearmodels 都支持）
- UI 提供"聚类变量"选择器
- 在结果表中明确标注标准误类型（稳健 / 聚类 / 经典）

#### 流式进度反馈
- 大模型拟合（如 MixedLM 或贝叶斯）时提供进度条
- 使用 `st.progress()` + 回调或线程分离

---

## 5. 附录：可选增强路线

### 阶段一（MVP）

- [ ] OLS / Logit / Probit 基本模型
- [ ] CSV + Excel 上传
- [ ] 变量类型自动检测（可手动覆盖）
- [ ] 散点图 + 回归线
- [ ] 系数表 + 模型统计量
- [ ] HTML + CSV 导出
- [ ] 单模型诊断（残差图、Q-Q 图）

### 阶段二（核心完善）

- [ ] Poisson / NegativeBinomial / GLM
- [ ] 面板数据模型（固定效应、随机效应）
- [ ] 多水平模型（MixedLM）
- [ ] LaTeX + Word 导出
- [ ] 多模型并列比较表
- [ ] 系数图 (dot-whisker)
- [ ] SAS/SPSS/Stata 格式支持
- [ ] 稳健标准误 + 聚类标准误

### 阶段三（高级功能）

- [ ] 贝叶斯回归（PyMC）
- [ ] 岭回归 / Lasso / 弹性网
- [ ] 边际效应图
- [ ] PDF 综合报告导出
- [ ] 交互项图形化展示
- [ ] 中介效应 / 调节效应分析
- [ ] 工具变量回归
- [ ] 自助法 (bootstrap) 标准误

### 阶段四（工程优化）

- [ ] Polars 后端支持大文件
- [ ] 并行模型拟合（多个模型规格同时跑）
- [ ] Docker 部署
- [ ] 用户项目 / 历史记录管理
- [ ] 回归结果的 R Markdown / Quarto 集成

---

> **文档版本**: v1.0
> **最后更新**: 2026-05-25
> **状态**: 技术架构草案，待团队评审后细化实施
