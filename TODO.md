# Regression Analysis -- TODO 清单

> 最后更新: 2026-05-25
> 基于三方评审 (Tech Advisor / Client / Feature Analyst) + Judge 裁定
> 当前状态: Phase 1-4 已完成 (v1.0 已发布), Phase 4.5/5.0 Web+功能补齐已交付

---

## 三方评审结论

| 需求 | 状态 | 三方一致意见 |
|------|------|-------------|
| Req 1: 上传通用表格数据集 | 完全满足 | 三方确认: CSV/Excel 解析 + 编码检测 + 预览已实现 |
| Req 2: 预览数据集 | 完全满足 | 三方确认: 数据表格 + 变量类型检测 + 缺失值统计已实现 |
| Req 3: 选择因变量和自变量 | 完全满足 | 三方确认: 变量选择器 + 类型覆盖 + 模型规格构建已实现 |
| **Req 4: 运行至少两种回归模型 (OLS + logit)** | **部分满足** | **三方确认: 仅 OLS 已实现, logit/probit 缺失** |
| Req 5: 在干净的表中查看回归结果 | 完全满足 | 三方确认: 系数表 + 统计量 + ANOVA 已实现 |
| Req 6: 以适合报表的格式导出回归表 | 完全满足 | 三方确认: CSV/Excel/LaTeX/HTML/PNG 导出已实现 |
| Req 7: 生成至少一个有意义的图表 | 完全满足 | 三方确认: 散点图 + 残差图 + Q-Q + 系数图已实现 |
| Req 8: 查看基本数据问题警告 | 完全满足 | 三方确认: VIF + 异常值 + 多重共线性警示已实现 |

**唯一争议点**: 无。三方完全一致 -- Req 4 是唯一缺口。

**客户额外反馈**: 两个 UX 痛点 (非需求违规, 但影响实际可用性):
1. Pyodide Web 版加载需 30-60 秒, 无进度指示器
2. 分类变量名显示为技术格式 (C(education)[T.本科]), 不人性化
3. Streamlit 与 Web 两个版本功能不对等, 用户困惑

---

## Phase 1: 概念验证 (POC) -- 已完成

- [x] 初始化 Python 项目结构 (pyproject.toml, .gitignore)
- [x] 配置开发环境 (uv, ruff, mypy, pre-commit)
- [x] 创建基础测试框架 (pytest, conftest.py)
- [x] 搭建 Streamlit 主入口 `app/app.py`
- [x] 实现 OLS 回归适配器 (`src/modeling/engines/statsmodels_engine.py`)
- [x] 构建统一结果数据结构 (`src/results/table.py` -- CoefficientRow, ModelResult)
- [x] 实现基础诊断函数 (R-squared, F 统计量, AIC/BIC)
- [x] 编写回归基准测试 (与 R `lm()` 对比, 差异 < 1e-10)
- [x] 用 3 个经典数据集验证正确性 (mtcars, iris, 模拟数据)
- [x] 实现 CSV 文件解析 (编码自动检测)
- [x] 构建最小 Streamlit 界面 (上传 -> 选变量 -> 显示系数表)
- [x] 实现基础 scatter plot + 回归线
- [x] POC 验证 (风险 R01/R02/R03 已验证)

---

## Phase 2: MVP 核心功能 -- 已完成

- [x] Excel (.xlsx/.xls) 文件解析支持
- [x] 编码自动检测 (UTF-8/GBK/ASCII fallback)
- [x] 数据表格预览组件
- [x] 变量类型自动检测 (连续/分类/二值/有序分类 + ID 列识别 + 用户手动覆盖)
- [x] 缺失值统计与标注
- [x] 文件大小限制与警告
- [x] 变量选择器 UI 组件 (因变量下拉 + 自变量多选 + 控制变量分组)
- [x] 模型规格构建 (ModelSpec + build_formula)
- [x] 描述性统计面板 (均值、标准差、极值、缺失率)
- [x] 数据子集筛选 (行过滤)
- [x] 模型控制面板 (常数项开关 + 置信区间水平 + 标准误类型)
- [x] 结果卡片组件 (系数表 + 统计量面板 + ANOVA)
- [x] 残差图 (Residuals vs Fitted)
- [x] Q-Q 图
- [x] 尺度-位置图 + Cook's distance
- [x] 统计警示自动标注 (VIF>10 标红 + 异常点高亮)
- [x] CSV/Excel/PNG 导出
- [x] 全中文界面 + 首次使用引导 + 中文错误提示
- [x] MVP 内部测试 (20 个回归场景) + 3 名外部试用者

---

## Phase 3: Beta 打磨 -- 已完成

- [x] 变量转换 UI 与引擎 (对数/标准化/中心化/平方项)
- [x] 交互项创建 UI
- [x] 稳健标准误选项 (HC0-HC3)
- [x] 多模型并列比较表 + 系数图 (dot-whisker plot)
- [x] 缺失值处理策略 (删除整行/均值填充/中位数填充)
- [x] 异常值检测与提示
- [x] LaTeX 表格导出 (Jinja2 + booktabs, 单模型 + 多模型对比, APA7 预设)
- [x] HTML 报告导出 + Word 报告导出
- [x] 分析复现包导出 (数据子集 + 配置 JSON + Python 脚本)
- [x] 性能优化 (缓存策略 + 延迟加载) + 会话状态持久化 + 崩溃恢复
- [x] 用户引导 + 示例数据集 (Sample Gallery 含 5 个预计算场景)
- [x] Beta 验证: 10+ 测试者, 满意度 >= 4.0/5, 覆盖率 >= 90%

---

## Phase 4: v1.0 发布 -- 已完成

- [x] 全面 UI 视觉一致性审查 + 色盲友好模式 + 响应式布局 + 键盘导航
- [x] 用户手册 (含 3 个完整案例) -- `docs/用户手册.md`
- [x] 开发者指南 -- `docs/开发者指南.md`
- [x] 已知问题清单 -- `docs/已知问题.md`
- [x] 性能基准测试 (10万行 x 20变量 <= 3秒) -- `scripts/benchmark.py`
- [x] 依赖安全审计 -- `docs/安全审计报告.md`
- [x] 版本发布说明 -- `CHANGELOG.md`
- [x] v1.0 版本标记与发布

---

## Phase 4.5: Web 适配 + 测试补充 + Bug 修复 -- 已完成

- [x] Pyodide 静态站 (`web/` 目录, 6 文件, GitHub Pages 跨仓库部署)
- [x] 测试补充 (+88 tests: sample_data 0%->100%, scatter 0%->78%, diagnostics 75%->96%)
- [x] Code Review (5-angle 深度审查, 15 findings)
- [x] Bug 修复: 5 bugs (1 critical + 2 high + 2 medium) 已修复合入
- [x] 471 tests 全部通过, 覆盖率 84%

---

## Phase 5.0: Web 功能补齐 + 覆盖率提升 -- 已完成

- [x] Web 版多模型对比图 + 变量转换 UI + 交互项 UI + scatter + .xls 支持
- [x] 测试覆盖率提升 (+68 tests: summary_generator 69%->95%, statistics 77%->92%)
- [x] _norm_ppf 修复 (A&S 26.2.23 尾概率映射 + 符号处理 + 6 tests)
- [x] Code Review 6 findings 已修复 4
- [x] GitHub Pages 已更新部署

---

## Phase 5.1: Logit 回归 -- 需求 4 补全 (最高优先级)

> **评审共识**: 缺少 logit 回归是项目当前最核心的合规缺口。客户明确指出"没有 logit, 这就是一个 OLS 计算器, 不是多模型回归分析工具"。
> **影响范围估计**: Tech Advisor 评估 8-10 个文件, Feature Analyst 评估 3-4 个 Python 文件 + 1 个新引擎文件 (~200-300 LOC)。最终以 Tech Advisor 的保守估计为准。

### 5.1.1 引擎层: statsmodels_logit_engine.py

- [x] 新建 `src/modeling/engines/statsmodels_logit_engine.py`
  - [x] 实现 `run_logit(data, formula)` -- 基于 statsmodels Logit/GLM
  - [x] 实现 `extract_logit(fitted_model)` -- 提取系数/标准误/z值/p值/95%CI
  - [x] 计算 Odds Ratio (OR = exp(coef)) + OR 置信区间
  - [x] 计算 pseudo R-squared (McFadden's R-squared)
  - [x] 计算 Log-Likelihood + AIC/BIC
  - [x] 处理完全分离 (perfect separation) 警告, 给出中文提示
  - [x] 处理收敛失败警告
- [x] 编写 `tests/unit/test_logit_engine.py` (目标 >= 90% 覆盖率)
  - [x] 二值因变量数据集 (如 mtcars vs=0/1 或 titanic 模拟)
  - [x] 系数正确性对比基准 (与 R `glm(family=binomial)` 对比)
  - [x] 完全分离边界情况
  - [x] 多分类自变量含有虚拟变量

### 5.1.2 数据结构层: ModelResult 重构 (技术顾问核心建议)

- [x] `src/results/table.py` 中 `ModelResult` dataclass 改造:
  - [x] `r_squared` 字段保持必填但允许 None (logit 无传统 R-squared)
  - [x] 新增 `pseudo_r_squared: float | None` 字段 (logit 专用)
  - [x] `f_statistic` 字段允许 None (logit 无 F 检验)
  - [x] 新增 `log_likelihood: float | None` 字段 (若尚未存在)
  - [x] 新增 `model_type: str` 字段 (区分 "ols" / "logit" / "probit")
  - [x] `CoefficientRow` 保持通用 (t_or_z 字段命名已兼容, 无需改名)
- [x] 所有下游消费者 (UI 组件 + 导出模块 + 图表模块) 适配:
  - [x] `app/components/result_card.py` -- 根据 model_type 条件渲染 (OLS 显示 R-squared/F-test, logit 显示 pseudo-R-squared/LR-test)
  - [x] `src/results/summary_generator.py` -- 中文摘要区分 OLS 和 logit 文本模板
  - [x] `src/data_io/exporter.py` -- 导出表头根据 model_type 自适应
  - [x] `src/export/latex_renderer.py` -- LaTeX 模板区分 OLS 和 logit 输出

### 5.1.3 调度层: Fitter 扩展

- [x] `src/modeling/fitter.py` 中 `ModelFitter` 改造:
  - [x] `fit()` 方法新增 `model_type: str = "ols"` 参数
  - [x] 根据 model_type 分发: "ols" -> statsmodels_engine, "logit" -> statsmodels_logit_engine
  - [x] 统一返回 `ModelResult` (包含 model_type 字段)
  - [x] `fit_multiple()` 扩展支持混合模型类型 (部分 OLS + 部分 logit), 对比表中标记
- [x] `src/modeling/specification.py` 中 `ModelSpec` 新增 `model_type: str = "ols"` 字段

### 5.1.4 UI 层: 模型类型选择器

- [x] `app/components/model_control.py` 新增:
  - [x] 模型类型下拉选择器 (OLS / Logit / Probit)
  - [x] 根据模型类型动态显示/隐藏相关选项:
    - Logit/Probit 模式下隐藏 F-test 相关显示
    - Logit/Probit 模式下添加 OR (Odds Ratio) 显示开关
- [x] `app/pages/03_model_spec.py` 和 `04_model_results.py` 适配:
  - [x] 因变量二值检测: 当用户选择二值因变量时自动建议 logit 模型
  - [x] 连续因变量选 logit 时给出确认提示
  - [x] 结果页面根据 model_type 展示对应统计量
- [x] Web 版 (`web/py/bridge.py` + `web/js/app.js`) 适配:
  - [x] `run_regression()` 新增 `model_type` 参数
  - [x] 前端添加模型类型选择器 (在 Model Tab)
  - [x] 结果渲染区分 OLS 和 logit 统计量

### 5.1.5 图表: Logit 专用可视化

- [x] `src/visualization/` 新增 `src/visualization/logit_plots.py` (或扩展 `coefficient.py`):
  - [x] 实现 `roc_curve_plot()` -- ROC 曲线 + AUC 标注 (plotly)
  - [x] 实现 `odds_ratio_plot()` -- OR 森林图 (Odds Ratio + 95% CI, 对数尺度)
  - [x] Web 版桥接函数 (`bridge.py`): `generate_roc_chart()` + `generate_or_chart()`
- [x] 测试: `tests/unit/test_logit_plots.py`

### 5.1.6 导出: Logit 专用模板

- [x] LaTeX 导出 (`src/export/latex_renderer.py`):
  - [x] logit 模型表: 系数列 -> OR 列 (exp(B)) + 标准误 + z值 + p值
  - [x] 脚注区显示 pseudo R-squared 和模型类型注释
- [x] 综合报告 (`src/export/html_report.py`):
  - [x] logit 专用报告模板: OR 解读 + 模型拟合统计量 (LR chi2/pseudo-R2)
- [x] Excel/CSV 导出: 适配 logit 结果字段

### 5.1.7 Web 版 Logit 可视化集成

- [x] `web/py/bridge.py` 集成 ROC 图表 + OR 图表生成
- [x] `web/js/app.js` 前端渲染 ROC 和 OR 图表
- [x] 测试 Web 版 logit 端到端流程

---

## Phase 5.2: UX 改进 (客户痛点)

> **来源**: Client Perspective 文档 + Code Review 反馈

### 5.2.1 Pyodide 加载进度指示

- [x] `web/index.html` 添加加载进度条组件 (带百分比/预估时间)
- [x] `web/js/app.js` 中 Pyodide 初始化阶段分步上报进度:
  - [x] 阶段 1: 下载 Pyodide core (0-40%)
  - [x] 阶段 2: 加载 micropip + 安装包 (40-70%)
  - [x] 阶段 3: 导入 Python 模块 + 预热引擎 (70-95%)
  - [x] 阶段 4: 就绪 (95-100%)
- [x] 首次加载提示: "首次加载约需 30-60 秒, 后续访问利用浏览器缓存将更快 (约 5-10 秒)"
- [x] 加载失败时显示重试按钮 + 诊断信息 (网络检测/浏览器兼容性)

### 5.2.2 分类变量名人性化

- [x] `src/modeling/specification.py` 中 `build_formula()` 输出可读标签:
  - [x] 当前: `C(education)[T.本科]` -> 目标: `教育水平: 本科`
  - [x] 方案 A: 保留变量原始标签 (patsy 的 `C(name, Treatment(reference='...'))`)
  - [x] 方案 B: 手动构造设计矩阵并附带标签映射, 后续 Consumer 层统一使用映射
- [x] `app/components/result_card.py` 系数表渲染使用标签映射
- [x] 导出模块 (`exporter.py`, `latex_renderer.py`) 使用标签映射
- [x] Web 版 `bridge.py` 中 `run_regression()` 返回 `variable_labels` 字段
- [x] `web/js/app.js` 结果表渲染使用标签

### 5.2.3 Streamlit-Web 版本功能对照 (用户认知对齐)

- [x] 在 README.md 添加功能对照表 (Streamlit vs Web)
- [x] 在 Web 版页面底部添加版本说明链接
- [x] Streamlit 首页添加"在线版"链接, Web 版添加"桌面版"链接

---

## Phase 5.3: Web-Streamlit 功能对齐

> **来源**: HANDOVER.md "下个 Session 建议" + Code Review 发现

### 5.3.1 Web 版功能补齐

- [x] Web 版 scatter chart 支持 Gallery 场景 (当前仅支持用户上传数据)
- [x] Web 版变量转换 UI 开放 (引擎已支持, UI 未开放: log/标准化/平方项)
- [x] Web 版交互项 UI 开放 (引擎已支持, UI 未开放)
- [x] Web 版多模型对比图集成 (当前仅单模型)

### 5.3.2 Web 版导出完善

- [x] Gallery 模式下 CSV 降级导出添加提示 (Web 版依赖 Pyodide, Gallery 离线模式不可用)
- [x] Excel 导出健壮性: openpyxl 加载失败时友好降级

### 5.3.3 工程健壮性

- [x] `compare_models()` 中 coefficient 缺少 CI 时跳过 whisker (而非画在 0 处)
- [x] Web bridge `parse_file` 返回的 `columns` 元数据透传完整性检查
- [x] CI 缺失兜底: GitHub Actions pipeline 缺失时手动创建

---

## Phase 5.4: v1.1 发布

- [ ] 整合测试: Logit 端到端测试 (Streamlit + Web)
- [ ] 回归基准: 5 个经典数据集的 logit 结果与 R `glm()` 对比
- [ ] 用户手册更新: 添加 Logit 回归章节 (含 OR 解读 + ROC 解读)
- [ ] 更新 `CHANGELOG.md` (v1.1.0 发布说明)
- [ ] 更新 `docs/已知问题.md`
- [ ] v1.1 版本标记与发布

---

## 未来版本 (v1.2+)

> 所有 Phase 5.1-5.3 完成后再考虑的项目

- [ ] Probit 回归 (logit 引擎就绪后, 仅需新增 GLM family=binomial(probit) 包装)
- [ ] 多层次模型 (混合效应模型, MixedLM / lme4-backend)
- [ ] 面板数据固定/随机效应 (linearmodels)
- [ ] Poisson / NegativeBinomial 回归 (计数因变量)
- [ ] 贝叶斯回归 (PyMC / Bambi)
- [ ] 岭回归 / Lasso / 弹性网 (sklearn adapter)
- [ ] 边际效应图 (AME/MEM -- 非线性模型解读刚需)
- [ ] 中介效应 / 调节效应分析 (Baron & Kenny + bootstrap)
- [ ] 工具变量回归 (IV 2SLS -- linearmodels)
- [ ] Bootstrap 标准误
- [ ] 分析快照 (JSON 配置保存/加载, 一键复现)
- [ ] Stata/SPSS 导入 (.dta/.sav 格式)
- [ ] Docker 部署
- [ ] 用户历史管理

---

## 优先级总览

```
Phase 5.1: Logit 回归 (Req 4 补全)          ★★★★★ 最高优先级 -- 合规缺口
Phase 5.2: UX 改进 (客户痛点)               ★★★★☆ 重要 -- 可用性瓶颈
Phase 5.3: Web-Streamlit 功能对齐           ★★★☆☆ 中等 -- 工程一致性
Phase 5.4: v1.1 发布                        ★★★☆☆ 中等 -- 版本收尾
Phase 5.5+: 未来版本 (v1.2+)                 ★★☆☆☆ 低 -- 增强功能
```

---

> **Judge 备注**: 三方评审完全一致 -- logit 回归是唯一合规缺口。Tech Advisor 提供了最全面的工程评估 (9 步计划, 8-10 文件), Feature Analyst 补充了 LOC 估算 (~200-300 LOC), Client 补充了 UX 痛点将 logit 列为"最紧急"。Task 5.1.1-5.1.7 按 Tech Advisor 的保守估计组织, 每项拆分为可独立验证的 Action Item。
