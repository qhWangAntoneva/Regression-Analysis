# Regression Analysis — 项目框架

> **生成方式**: 客户 Agent + 技术顾问 Agent + 评审 Agent 三方协同
> **日期**: 2026-05-25
> **状态**: v1.0 框架定稿

---

## 1. 项目愿景

构建一个面向社会科学研究者的回归分析工具，允许用户**上传表格数据集 → 选择变量 → 运行回归模型 → 导出投稿级表格和图表**，全程无需编写代码，实现 "Upload. Click. Publish."

---

## 2. 项目范围

### 2.1 In Scope (本次覆盖)

- 表格数据上传（CSV/Excel）
- 变量选择与基础转换
- OLS 回归（核心引擎）
- 回归结果表格展示（系数、标准误、p 值、R² 等）
- 基础诊断图（残差图、Q-Q 图）
- 描述性统计输出
- 结果导出（CSV/Excel/HTML/PNG）
- 全中文界面

### 2.2 Out of Scope (本次不做)

| 功能 | 排除理由 |
|------|---------|
| 逻辑回归 / Probit | 增加模型引擎复杂度，第一阶段聚焦连续因变量 |
| 面板数据模型 | 需 Hausman 检验等，架构复杂 |
| 时间序列分析 | 完全不同的统计范式 |
| 结构方程模型 | 需路径图渲染，超出工具定位 |
| 机器学习方法 | 定位为推断式回归，非预测 |
| 多用户协作 / 云端存储 | 初期为本地单机工具 |
| 数据库直连 | 增加大量适配工作，MVP 仅支持文件上传 |
| REST API | 产品初期无需被外部调用 |

---

## 3. 技术架构

### 3.1 核心技术栈

| 层级 | 技术选择 | 理由 |
|------|---------|------|
| 应用框架 | **Streamlit** | Python 全栈、零前端门槛、天然对接 statsmodels/plotly 生态 |
| 统计引擎 | **statsmodels** | 原生支持 OLS/Logit/Probit/MixedLM，社会科学标准 |
| 可视化 | **plotly + matplotlib** | plotly 交互 + matplotlib 高质量静态导出双引擎 |
| 数据处理 | **pandas (PyArrow 后端)** | 社区标准，>=2.0 性能大幅提升 |
| 表格导出 | **jinja2 + LaTeX 模板** | 学术出版级表格，类 stargazer 风格 |
| 报告导出 | **python-docx + weasyprint** | Word 和 PDF 双格式覆盖 |

### 3.2 四层架构

```
┌──────────────────────────────────────────────┐
│            PRESENTATION LAYER                 │
│  Streamlit UI (app + pages + components)      │
├──────────────────────────────────────────────┤
│            APPLICATION LAYER                  │
│  会话状态管理 / 工作流编排 / 缓存              │
├──────────────────────────────────────────────┤
│            BUSINESS LOGIC LAYER               │
│  data_io / preprocessing / modeling           │
│  results / visualization / export             │
├──────────────────────────────────────────────┤
│            DATA LAYER                         │
│  文件系统 / 会话缓存 / 配置持久化              │
└──────────────────────────────────────────────┘
```

### 3.3 核心数据流

```
上传 → 解析 → 预览清洗 → 变量选择 → 模型规格 → 拟合 → 结果展示 → 导出
```

---

## 4. 目录结构

```
regression-analysis/
├── app/                          # Streamlit 应用
│   ├── app.py                    # 主入口
│   ├── config.py                 # 页面配置
│   ├── pages/                    # 多页面
│   │   ├── 01_data_upload.py
│   │   ├── 02_data_explore.py
│   │   ├── 03_model_spec.py
│   │   ├── 04_model_results.py
│   │   ├── 05_model_compare.py
│   │   └── 06_export.py
│   ├── components/               # 可复用 UI 组件
│   │   ├── data_table.py
│   │   ├── variable_selector.py
│   │   ├── model_control.py
│   │   ├── result_card.py
│   │   └── export_dialog.py
│   └── assets/
├── src/                          # 核心业务逻辑
│   ├── data_io/                  # 文件解析与导出
│   │   ├── parser.py
│   │   ├── encoding.py
│   │   └── exporter.py
│   ├── preprocessing/            # 预处理
│   │   ├── type_detector.py
│   │   ├── cleaner.py
│   │   └── transformer.py
│   ├── modeling/                 # 建模
│   │   ├── specification.py
│   │   ├── fitter.py
│   │   ├── diagnostics.py
│   │   ├── comparison.py
│   │   └── engines/
│   ├── results/                  # 结果处理
│   │   ├── table.py
│   │   └── statistics.py
│   ├── visualization/            # 可视化
│   │   ├── scatter.py
│   │   ├── residual.py
│   │   ├── coefficient.py
│   │   └── diagnostics.py
│   ├── export/                   # 多种格式导出
│   │   ├── latex.py
│   │   ├── word.py
│   │   ├── pdf.py
│   │   ├── html_report.py
│   │   └── templates/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── docs/
├── notebooks/
├── scripts/
├── pyproject.toml
└── .gitignore
```

---

## 5. 目标用户画像

| 画像 | 背景 | 技术能力 | 核心痛点 |
|------|------|---------|---------|
| **张薇** — 社科研究生 | 硕士论文写作，使用 CGSS 数据 | Stata 入门级，无 Python/R | Stata 授权贵；LaTeX 表格不会做；中英文术语对不上 |
| **陈志远** — 市场研究员 | 互联网公司用户研究，5 年经验 | 熟练 Excel，pandas 基础 | 无 SPSS 授权；汇报格式不统一；需向非技术业务方展示 |
| **李明远** — 政策分析师 | 省级研究机构，经济学硕士 | 熟悉 Stata 和计量 | 数据格式混乱；需向领导汇报"一看就懂"的图表；操作需留档备查 |

---

## 6. 功能优先级 (MoSCoW)

### Must Have

- CSV/Excel 数据集上传
- 变量选择界面（因变量 + 自变量）
- OLS 回归引擎
- 格式化结果表（系数、标准误、p 值、R²）
- 描述性统计
- 基础数据预览
- CSV/Excel 结果导出
- 全中文界面

### Should Have

- 变量转换（对数、标准化、平方项）
- 残差图、Q-Q 图
- VIF 多重共线性诊断
- 多模型并列对比
- 交互项自动生成
- 控制变量选择

### Could Have

- LaTeX 表格导出
- 稳健标准误选项
- 加权最小二乘法
- 分析日志显示
- 变量标签管理

### Won't Have (本次)

- 逻辑回归 / Probit
- 面板数据 / 时间序列
- 结构方程模型 / 机器学习
- 多用户协作 / 云端
- 数据库直连 / REST API

---

## 7. 非功能需求

### 性能
- 数据规模上限: 10,000 行 × 200 列
- OLS 模型 (20 变量): ≤3 秒
- 首屏加载: ≤5 秒
- 峰值内存: ≤500MB

### 可用性
- 全中文界面，术语中英对照
- 错误信息为中文自然语言，不暴露堆栈
- 首次使用 3-5 步引导
- 键盘导航支持 (WCAG 2.1 Level A)

### 安全与隐私
- 100% 本地处理，数据不上传
- 无跟踪、无远程依赖
- 导出文件无水印或追踪元数据

### 可维护性
- 引擎与 UI 严格解耦
- 核心模块测试覆盖率 ≥ 90%
- 新增模型类型不修改现有代码超过 3 个文件

---

## 8. 关键风险

| ID | 风险 | 可能性 | 影响 | 缓解措施 |
|----|------|--------|------|---------|
| R01 | 统计输出不准确（边界条件） | 中 | 高 | 50+ 边界测试用例 + 输入校验层 |
| R02 | 大文件导致前端崩溃 | 中 | 高 | 文件大小硬限制 + 分页处理 |
| R03 | 数据隐私泄露 | 高 | 高 | 纯本地处理架构 + 声明不上传 |
| R04 | 用户误用统计方法 | 高 | 中 | 变量类型自动检测 + 警告系统 |
| R08 | 项目范围蔓延 | 高 | 中 | v1.0 仅线性回归，每新模型需 80% 覆盖率审查 |

---

## 9. 竞品差异化定位

| 维度 | 我们的定位 | 竞品现状 |
|------|-----------|---------|
| **输出质量** | 投稿级表格（LaTeX APA7 就绪） | SPSS/jamovi 需要手动调整格式 |
| **智能解读** | 自动生成"结果"章节草稿 | 无工具提供此功能 |
| **中文体验** | 编码检测 / 中文 UI / 中文论文格式 | 英文工具在中文环境有显著摩擦 |
| **一键复现** | 参数配置 JSON 持久化 | 介于 SPSS（不可复现）和 R（完整编程）之间 |

**品牌口号**: "上传数据，点击运行，直接投稿。"
