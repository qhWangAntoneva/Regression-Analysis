# Regression Analysis — 交接文档

> 最后更新: 2026-05-27 (Session 9 — deploy workflow 修复，if:secrets bug + docker 权限)
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
| Phase 6.1 (Hausman/exposure/robust SE/noqa) | 完成 | 852 |
| **当前** | **v1.2** | **852 tests** |

### v1.2 新增统计模型

| 模型 | 引擎文件 | 说明 |
|------|----------|------|
| Probit | `src/modeling/engines/statsmodels_probit_engine.py` | 二分类 MLE，z 统计量 |
| Poisson | `src/modeling/engines/statsmodels_count_engine.py` | 计数数据 GLM，IRR=exp(coef)；支持 exposure 变量（rate 模型） |
| Negative Binomial | 同上 | 过度离散，含 dispersion 参数；支持 exposure 变量（rate 模型） |
| MixedLM (多层次) | `src/modeling/engines/statsmodels_mixedlm_engine.py` | 嵌套数据 REML，随机效应 |
| Panel FE/RE | `src/modeling/engines/statsmodels_panel_engine.py` | 面板固定/随机效应，linearmodels；含 Hausman 检验 |

### v1.2 工程交付

- **CI/CD**: GitHub Actions — ruff lint + mypy type check + pytest + coverage + GitHub Pages 部署 + Docker 推送
- **Docker**: 多阶段构建 (uv + Python 3.12-slim) + docker-compose.yml
- **Web bridge**: 分类变量交互支持 (cat×num, cat×cat)，与 patsy 输出对齐
- **重构**: ModelResult 添加 `is_mle_model`/`is_binary_choice`/`is_count_model` 语义属性，消除硬编码 model_type 检查
- **Hausman 检验**: Panel FE vs RE 选择的关键诊断，集成在 04_model_results.py 结果页
- **暴露变量 (Exposure)**: Poisson/NegBin rate 模型支持，ModelSpec + model_control UI + count engine 联动
- **MLE Robust SE (HC0-HC3)**: Logit/Probit/Poisson/NegBin 模型可选稳健标准误
- **死代码清理**: 移除 13 条欺诈性 noqa（imports/unused 变量），跨 6 个源文件

---

## 关键目录

```
Regression Analysis/
├── CLAUDE.md / HANDOVER.md / README.md
├── app/                    # Streamlit 应用 (app.py + pages/)
├── web/                    # Pyodide 静态 Web (index.html + js/ + py/bridge.py)
├── src/                    # 核心库
│   ├── data_io/            # 数据 I/O
│   ├── modeling/           # 模型引擎 (7 种) + Hausman 检验
│   │   └── engines/        # statsmodels_engine / logit / probit / count / mixedlm / panel
│   ├── preprocessing/      # 数据预处理
│   ├── results/            # ModelResult + table + summary
│   └── visualization/      # 绑图
├── tests/                  # pytest (852 tests)
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
- **Business Logic**: data_io → preprocessing → modeling (7 engines + Hausman) → results → visualization
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
| pytest | 849 passed (852 collected), 3 skipped |

CI 配置: `.github/workflows/ci.yml`。Deploy 使用 SSH deploy key (`DEPLOY_KEY` secret，仅限 `qhWangAntoneva.github.io` 仓库写权限)。注意: job 级 `if: secrets.*` 会导致 GitHub Actions 解析失败 ("workflow file issue")，已移除该守卫条件。

---

## 已知问题

详见 `docs/已知问题.md`。核心:
1. Windows 终端 GBK 编码限制
2. Web bridge categorical×numeric 交互走 pd.get_dummies 非 patsy（v1.2 已修复交互列生成，列名格式与 patsy 对齐）
3. ModelSpec.interaction_terms 仅支持 2-way pairs

---

## Session 回顾

### Session 5 → Session 6 (已完成)

- [x] **Web 版 MixedLM/Panel UI 修复** (CRITICAL): `web/index.html` 已添加 group_var / entity_var / time_var 选择器 + 条件显示逻辑
- [x] **Hausman 检验** (MAJOR): Panel FE vs RE 选择的关键诊断，集成在 `src/modeling/hausman.py` + `04_model_results.py`，含 Hausman 统计量 + p-value 输出
- [x] **清理 13 条死代码 noqa**: 7 处 removals 跨 6 个源文件，0 CRITICAL
- [x] **Exposure 变量支持** (MINOR): Poisson/NegBin rate 模型，ModelSpec + model_control UI + count engine 联动，含 UI 下拉选择器 + 引擎 offset 处理
- [x] **MLE 模型 Robust SE 选项** (MINOR): HC0-HC3 对 Logit/Probit/Poisson/NegBin，`model_control.py` 动态显示 SE 选择

### Session 6 复查清单 (已完成 — 2026-05-27 验证)

- [x] CI 全绿验证: ruff PASS + mypy PASS + pytest 849 passed (852 collected) + 3 skipped
- [x] **Hausman 检验代码审查**: `src/modeling/hausman.py` 公式 (Wooldridge 2010) 已审查，04_model_results.py 条件显示逻辑正确，Web 版兼容
- [x] **Exposure + Robust SE 集成测试**: Poisson/NegBin rate 模型端到端已验证，MLE 模型各 HC 变体输出正常
- [ ] **Docker 构建验证**: `docker build -t regression-analysis . && docker compose up`
  - Docker 环境仍未就绪，待有环境时验证
- [x] **noqa 残留审计**: 107 处 noqa 分布在源文件和测试中，无可疑 QUESTIONABLE 残留；均为合法单行抑制
- [x] **Web 版 (Pyodide) 新特性验证**: Hausman / Exposure / Robust SE 在 Pyodide 中可用 (commit c7fde7f + 19b9536)

### Session 7 (已完成 — 2026-05-27)

- [x] **Deploy Pages 修复 (CRITICAL)**: 停用 GH_PAT (secret 缺失导致所有 deploy 运行失败)，改为 SSH deploy key (`DEPLOY_KEY` secret)。3 条 gh CLI 命令完成设置，`deploy.yml` 改为 `deploy_key` 参数，保留 `if` 守卫
- [x] **文档同步**: HANDOVER/TODO/CHANGELOG/已知问题/memory 全面更新至 v1.2 状态

### Session 8 (已完成 — 2026-05-27)

- [x] **Gallery 更新**: 新增 MixedLM（学校学业成绩）和 Panel（省级经济增长）2 个展示场景，共 7 个场景
- [x] **Code Review Bug 修复**: `_json_to_model_result` 死代码修复（`return` 导致 MixedLM/Panel 字段恢复无效）+ `f_pooled` 元组序列化修复
- [x] **发布 v1.2 tag**: `git tag v1.2.0 && git push --tags`
- [x] **v1.2 发布说明**: 基于 CHANGELOG 生成 GitHub Release → https://github.com/qhWangAntoneva/Regression-Analysis/releases/tag/v1.2.0
- [ ] **Docker 构建验证**: `docker build -t regression-analysis . && docker compose up`（需 Docker 环境）

### Session 9 — Deploy Workflow 修复 (已完成 — 2026-05-27)

- [x] **GitHub Actions "workflow file issue" 根因定位**: job 级 `if: secrets.DEPLOY_KEY != ''` 导致 YAML 解析失败。`${{ }}` 包裹与否均触发，`if: true` 无此问题。GitHub 文档称 secrets 可在 `if` 中使用，但实际行为不一致
- [x] **移除有问题的 `if` 守卫**: deploy-pages 不再使用 `if: secrets.*`，改为无条件运行（workflow 仅 push-to-master，secret 必然存在）
- [x] **修复 Docker 权限**: docker job 的 `permissions: packages: write` 覆盖了默认含 `contents: read` 的权限集，导致 `actions/checkout@v4` 失败。添加 `contents: read` 后恢复
- [x] **清理 debug commit 历史**: `git reset --soft` 将 6 个 debug commit 压缩为 1 个干净 commit
- [x] **Git remote SSH 迁移**: 从 HTTPS (需 PAT) 切换至 SSH（`id_ed25519` 密钥已注册 GitHub 账户）
- [x] **验证通过**: CI ✅ + Deploy ✅ (deploy-pages + docker 双 job 成功)

### v1.3+ 候选 (来自 TODO.md)

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
uv run python -m pytest tests/ -v              # 852 tests
uv run ruff check                               # lint
uv run mypy app/ --ignore-missing-imports       # type check
uv run streamlit run app/app.py                 # 启动 Streamlit
bash web/deploy.sh                               # 部署 Web 版
docker build -t regression-analysis .           # Docker 构建
docker compose up                                # Docker 启动
```
