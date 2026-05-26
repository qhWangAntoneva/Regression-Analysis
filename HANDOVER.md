# Regression Analysis — 交接文档

> 最后更新: 2026-05-26 (Session 4 — v1.2 交付)
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 部署: https://qhwangantoneva.github.io/regression-analysis/

---

## 项目状态

| 阶段 | 状态 | 测试数 |
|------|------|--------|
| Phase 1-4 (POC → v1.0) | 完成 | 278+ |
| Phase 4.5 (Web + 测试补充) | 完成 | 549 |
| Phase 5.0-5.4 (Logit/UX/对齐/v1.1) | 完成 | 599 |
| Phase 6.0 (v1.2 — 5 种新模型) | 完成 | 849 |
| **当前** | **v1.2** | **849 tests** |

### v1.2 新增统计模型

| 模型 | 引擎文件 | 说明 |
|------|----------|------|
| Probit | `src/modeling/engines/statsmodels_probit_engine.py` | 二分类 MLE，z 统计量 |
| Poisson | `src/modeling/engines/statsmodels_count_engine.py` | 计数数据 GLM，IRR=exp(coef) |
| Negative Binomial | 同上 | 过度离散，含 dispersion 参数 |
| MixedLM (多层次) | `src/modeling/engines/statsmodels_mixedlm_engine.py` | 嵌套数据 REML，随机效应 |
| Panel FE/RE | `src/modeling/engines/statsmodels_panel_engine.py` | 面板固定/随机效应，linearmodels |

### v1.2 工程交付

- **CI/CD**: GitHub Actions — ruff lint + mypy type check + pytest + coverage + GitHub Pages 部署 + Docker 推送
- **Docker**: 多阶段构建 (uv + Python 3.12-slim) + docker-compose.yml
- **Web bridge**: 分类变量交互支持 (cat×num, cat×cat)，与 patsy 输出对齐
- **重构**: ModelResult 添加 `is_mle_model`/`is_binary_choice`/`is_count_model` 语义属性，消除硬编码 model_type 检查

---

## 关键目录

```
Regression Analysis/
├── CLAUDE.md / HANDOVER.md / README.md
├── app/                    # Streamlit 应用 (app.py + pages/)
├── web/                    # Pyodide 静态 Web (index.html + js/ + py/bridge.py)
├── src/                    # 核心库
│   ├── data_io/            # 数据 I/O
│   ├── modeling/           # 模型引擎 (7 种: ols/logit/probit/poisson/negbin/mixedlm/panel)
│   │   └── engines/        # statsmodels_engine / logit / probit / count / mixedlm / panel
│   ├── preprocessing/      # 数据预处理
│   ├── results/            # ModelResult + table + summary
│   └── visualization/      # 绑图
├── tests/                  # pytest (849 tests)
├── docs/                   # 用户手册/开发者指南/已知问题
├── scripts/                # benchmark / generate_gallery_json
├── Dockerfile              # 多阶段构建
├── docker-compose.yml      # Streamlit 服务
└── .github/workflows/      # CI (lint/typecheck/test) + deploy (pages/docker)
```

---

## 架构

4 层 (Streamlit + Web 共享 Business Logic):
- **Presentation**: Streamlit pages / Web HTML+JS
- **Application**: session_state / Pyodide bridge
- **Business Logic**: data_io → preprocessing → modeling (7 engines) → results → visualization
- **Data**: 文件系统 (上传/导出) / 浏览器内存

模型调度: `ModelFitter.fit()` 按 `spec.model_type` 分派到对应引擎。

---

## 技术选型

| 领域 | 选型 |
|------|------|
| UI | Streamlit + Pyodide/HTML/JS |
| 统计引擎 | statsmodels (OLS/Logit/Probit/MixedLM/GLM) + linearmodels (Panel) |
| 图表 | plotly (交互) + matplotlib (静态) |
| 包管理 | uv |
| 测试 | pytest + pytest-cov |
| CI/CD | GitHub Actions (ruff + mypy + pytest + deploy) |
| 容器化 | Docker multi-stage + docker-compose |

---

## CI 状态

| 检查 | 状态 |
|------|------|
| ruff lint | 0 errors |
| mypy app/ | 0 errors |
| pytest | 849 passed, 3 skipped |

CI 配置: `.github/workflows/ci.yml`。Deploy 需要 `GH_PAT` secret (对 GitHub Pages 仓库的写入权限)。

---

## 已知问题

详见 `docs/已知问题.md`。核心:
1. Windows 终端 GBK 编码限制
2. Web bridge categorical×numeric 交互走 pd.get_dummies 非 patsy（v1.2 已修复交互列生成，列名格式与 patsy 对齐）
3. ModelSpec.interaction_terms 仅支持 2-way pairs

---

## 下个 Session 建议

### 复查清单 (priority)

- [ ] CI 全绿验证: `uv run ruff check && uv run mypy app/ --ignore-missing-imports && uv run python -m pytest tests/`
- [ ] 5 种新模型的 Streamlit UI 端到端测试（Probit/Poisson/NegBin/MixedLM/Panel FE/Panel RE 各跑一次完整流程）
- [ ] Web 版 (Pyodide) 新模型功能测试
- [ ] Docker 构建验证: `docker build -t regression-analysis . && docker compose up`
- [ ] noqa 审计：`grep -r "# noqa:" --include="*.py" src/ app/` — 复核每个 noqa 是否合理（Session 4 已清理欺诈性 noqa，剩余 ~130 条需抽样复查）

### 剩余 v1.2+ TODO (来自 TODO.md)

- [ ] 岭回归 / Lasso / 弹性网 (sklearn adapter)
- [ ] 边际效应图 (AME/MEM)
- [ ] 中介效应 / 调节效应分析 (Baron & Kenny + bootstrap)
- [ ] 工具变量回归 (IV 2SLS)
- [ ] Bootstrap 标准误
- [ ] 贝叶斯回归 (PyMC / Bambi)
- [ ] 分析快照 (JSON 配置保存/加载)
- [ ] Stata/SPSS 导入 (.dta/.sav)

---

## 测试与开发

```bash
uv run python -m pytest tests/ -v              # 849 tests
uv run ruff check                               # lint
uv run mypy app/ --ignore-missing-imports       # type check
uv run streamlit run app/app.py                 # 启动 Streamlit
bash web/deploy.sh                               # 部署 Web 版
docker build -t regression-analysis .           # Docker 构建
docker compose up                                # Docker 启动
```
