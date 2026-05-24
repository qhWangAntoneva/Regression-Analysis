# Regression Analysis — TODO 清单

> 基于三方 Agent 协同产出，按交付阶段组织

---

## Phase 1: 概念验证 (POC) — 第 1-3 周

### 1.1 项目初始化
- [ ] 初始化 Python 项目结构（pyproject.toml, .gitignore）
- [ ] 配置开发环境（uv, ruff, mypy, pre-commit）
- [ ] 创建基础测试框架（pytest, conftest.py）
- [ ] 搭建 Streamlit 主入口 `app.py`

### 1.2 统计引擎原型
- [ ] 实现 OLS 回归适配器（`src/modeling/engines/statsmodels_engine.py`）
- [ ] 构建统一结果数据结构（`src/results/table.py` — CoefficientRow, ModelResult）
- [ ] 实现基础诊断函数（R², F 统计量, AIC/BIC）
- [ ] 编写回归基准测试（与 R `lm()` 对比，差异 < 1e-10）
- [ ] 用 3 个经典数据集验证正确性（mtcars, iris, 模拟数据）

### 1.3 最小端到端原型
- [ ] 实现 CSV 文件解析（`src/data_io/parser.py` — preview 模式）
- [ ] 构建最小 Streamlit 界面（上传 → 选变量 → 显示系数表）
- [ ] 实现基础 scatter plot + 回归线（`src/visualization/scatter.py`）

### 1.4 POC 验证
- [ ] 验证所有"高可能性 + 高影响"风险（R01, R02, R03）
- [ ] 准备演示脚本（10 分钟内展示给非技术观众）
- [ ] 决策评审：继续 MVP / 否决项目

---

## Phase 2: MVP 核心功能 — 第 4-7 周

### 2.1 数据导入与预览
- [ ] Excel (.xlsx/.xls) 文件解析支持
- [ ] 编码自动检测（UTF-8/GBK/ASCII fallback）（`src/data_io/encoding.py`）
- [ ] 数据表格预览组件（`app/components/data_table.py`）
- [ ] 变量类型自动检测（`src/preprocessing/type_detector.py`）
  - [ ] ID 列识别
  - [ ] 连续/分类/二值/有序分类检测
  - [ ] 用户手动覆盖 UI
- [ ] 缺失值统计与标注
- [ ] 文件大小限制与警告

### 2.2 变量选择与模型配置
- [ ] 变量选择器 UI 组件（`app/components/variable_selector.py`）
  - [ ] 因变量下拉选择
  - [ ] 自变量多选
  - [ ] 控制变量分组
- [ ] 模型规格构建（`src/modeling/specification.py`）
- [ ] 描述性统计面板（均值、标准差、极值、缺失率）
- [ ] 数据子集筛选（行过滤）
- [ ] 模型控制面板组件（`app/components/model_control.py`）
  - [ ] 常数项开关
  - [ ] 置信区间水平设置

### 2.3 回归结果展示
- [ ] 结果卡片组件（`app/components/result_card.py`）
- [ ] 系数表渲染（系数、标准误、t 值、p 值、95% CI、显著性星标）
- [ ] 模型拟合统计量面板（R², Adj-R², F 统计量, AIC/BIC, RMSE）
- [ ] 方差分析表

### 2.4 诊断图表
- [ ] 残差图（Residuals vs Fitted）（`src/visualization/residual.py`）
- [ ] Q-Q 图（`src/visualization/diagnostics.py`）
- [ ] 尺度-位置图
- [ ] 统计警示自动标注（VIF>10 标红、异常点高亮）

### 2.5 基础导出
- [ ] CSV 系数表导出
- [ ] Excel 结果导出（`src/data_io/exporter.py`）
- [ ] 图表保存为 PNG（300 DPI）

### 2.6 用户体验
- [ ] 全中文界面
- [ ] 首次使用引导（3-5 步弹窗）
- [ ] 中文错误提示（非堆栈跟踪）
- [ ] 内联帮助系统

### 2.7 MVP 验证
- [ ] 内部测试 20 个回归场景（无不正确输出）
- [ ] 3 名外部试用者完成端到端任务
- [ ] 核心模块测试覆盖率 ≥ 85%
- [ ] 用户满意度 ≥ 3.5/5

---

## Phase 3: Beta 打磨 — 第 8-11 周

### 3.1 高级建模功能
- [ ] 变量转换 UI 与引擎
  - [ ] 对数变换
  - [ ] 标准化 (z-score)
  - [ ] 中心化
  - [ ] 平方项生成
- [ ] 交互项创建 UI（选择两个变量 → 自动生成乘积项）
- [ ] 稳健标准误选项（HC0-HC3）
- [ ] 多模型并列比较表（`src/results/comparison.py`）
- [ ] 系数图（dot-whisker plot）（`src/visualization/coefficient.py`）

### 3.2 完整导出功能
- [ ] LaTeX 表格导出（Jinja2 + booktabs 模板）
  - [ ] 单模型表
  - [ ] 多模型对比表
  - [ ] APA7 格式预设
- [ ] HTML 报告导出（`src/export/html_report.py`）
- [ ] SVG 图表导出
- [ ] 完整分析报告（含模型描述 + 描述统计 + 回归表 + 诊断图）
- [ ] 分析复现包导出（数据子集 + 配置 JSON + Python 脚本）

### 3.3 数据增强
- [ ] 缺失值处理策略（删除整行 / 均值填充 / 中位数填充）
- [ ] 异常值检测与提示
- [ ] 变量标签管理

### 3.4 工程化
- [ ] 性能优化（缓存策略、延迟加载）
- [ ] 会话状态持久化（关闭浏览器后恢复）
- [ ] 崩溃恢复（自动保存最近结果）
- [ ] 用户引导 + 示例数据集

### 3.5 Beta 验证
- [ ] Beta 测试者 ≥ 10 人
- [ ] 无 P0 Bug（数据丢失/错误结果）
- [ ] P1 Bug 关闭率 ≥ 90%
- [ ] 用户满意度 ≥ 4.0/5
- [ ] 测试覆盖率 ≥ 90%（核心模块）

---

## Phase 4: v1.0 发布 — 第 12-14 周

### 4.1 最终打磨
- [x] 全面 UI 视觉一致性审查
- [x] 色盲友好模式
- [x] 响应式布局适配（1366×768+）
- [x] 键盘导航完善

### 4.2 文档
- [x] 用户手册（含 3 个完整案例）— `docs/用户手册.md`
- [x] 开发者指南 — `docs/开发者指南.md`
- [x] 已知问题清单 — `docs/已知问题.md`

### 4.3 发布准备
- [ ] 最终性能基准测试（10万行 × 20 变量 ≤ 3 秒）
- [ ] 依赖安全审计
- [ ] 打包为可执行文件（可选）
- [ ] v1.0 版本标记与发布

### 4.4 发布后跟踪
- [x] 用户反馈通道建立
- [x] 问题追踪系统配置
- [x] v1.1 功能规划

---

## 未来版本 (v1.1+)

- [ ] 逻辑回归 / Probit 支持
- [ ] 多层次模型（混合效应模型, MixedLM）
- [ ] 面板数据固定/随机效应
- [ ] Poisson / NegativeBinomial 回归
- [ ] 贝叶斯回归 (PyMC / Bambi)
- [ ] 岭回归 / Lasso / 弹性网
- [ ] 边际效应图
- [ ] 中介效应 / 调节效应分析
- [ ] 工具变量回归
- [ ] Bootstrap 标准误
- [ ] 分析快照（JSON 配置保存/加载）
- [ ] Docker 部署
- [ ] 用户历史管理
