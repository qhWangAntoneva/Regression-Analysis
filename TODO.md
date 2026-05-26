# Regression Analysis — TODO

> 最后更新: 2026-05-27 | v1.2 已发布 | 852 tests (849 passed, 3 skipped)

## 已完成阶段

- [x] Phase 1-4: POC → MVP → Beta → v1.0 (278+ tests)
- [x] Phase 4.5: Web 适配 + 测试补充 + Bug 修复 (549 tests)
- [x] Phase 5.0-5.4: Logit/UX/对齐/v1.1 (599 tests)
- [x] Phase 6.0: v1.2 — 5 种新模型 (Probit, Poisson, NegBin, MixedLM, Panel FE/RE) (849 tests)
- [x] Phase 6.1: Hausman 检验 + Exposure 变量 + MLE Robust SE + 死代码清理 (852 tests)

## v1.2 已交付功能（原 v1.2+ 候选 → 已完成）

- [x] Probit 回归 (GLM family=binomial(probit) 包装)
- [x] 多层次模型 (MixedLM / lme4-backend)
- [x] 面板数据固定/随机效应 (linearmodels)
- [x] Poisson / NegativeBinomial 回归 (计数因变量)
- [x] Web bridge categorical 交互支持 (cat×num, cat×cat)
- [x] Docker 部署 + CI/CD (GitHub Actions + Docker multi-stage)
- [x] Hausman 检验 (Panel FE vs RE 诊断)
- [x] Exposure 变量支持 (Poisson/NegBin rate 模型)
- [x] MLE Robust 标准误 (HC0-HC3 对 Logit/Probit/Poisson/NegBin)

## v1.3+ 候选（按优先级排列）

| 优先级 | 功能 | 说明 | 预计工作量 |
|--------|------|------|-----------|
| ★★★ | 边际效应图 (AME/MEM) | 非线性模型解读辅助 | 1-2 周 |
| ★★★ | 分析快照 (JSON 配置保存/加载) | 可复现性核心功能 | 2 周 |
| ★★☆ | 岭回归 / Lasso / 弹性网 | sklearn adapter | 2-3 周 |
| ★★☆ | 工具变量回归 (IV 2SLS) | 因果推断需求 | 2-3 周 |
| ★★☆ | Bootstrap 标准误 | 稳健推断补充 | 1-2 周 |
| ★★☆ | 中介效应 / 调节效应分析 | Baron & Kenny + bootstrap | 2-3 周 |
| ★☆☆ | 贝叶斯回归 | PyMC / Bambi | 3-4 周 |
| ★☆☆ | Stata/SPSS 导入 | .dta/.sav 格式支持 | 1 周 |

## 工程待办

- [ ] 发布 v1.2 GitHub Release (含版本 tag v1.2.0)
- [ ] Docker 构建验证（需 Docker 环境）
- [ ] 样本 Gallery 补充（MixedLM、面板数据展示场景）
- [ ] noqa 总量优化（当前 ~107 条，逐步减少）
- [ ] Web 版 (Pyodide) gallery 场景对齐 Streamlit 版
