# Regression Analysis — 交接文档

> 最后更新: 2026-05-26 (Session 3)
> GitHub: https://github.com/qhWangAntoneva/Regression-Analysis
> 部署: https://qhwangantoneva.github.io/regression-analysis/

---

## 项目状态

| 阶段 | 状态 | 测试数 |
|------|------|--------|
| Phase 1-4 (POC → v1.0) | 完成 | 278+ |
| Phase 4.5 (Web + 测试补充) | 完成 | 549 |
| Phase 5.0-5.1 (Logit + _norm_ppf) | 完成 | 549 |
| Phase 5.2 (UX 改进) | 完成 | 560 |
| Phase 5.3 (Web-Streamlit 对齐) | 完成 | 560 |
| Phase 5.4 (v1.1 发布) | 完成 | 599 |
| **当前** | **v1.1** | **599 tests** |

## 关键目录

```
Regression Analysis/
├── CLAUDE.md / HANDOVER.md / README.md
├── app/                    # Streamlit 应用 (app.py + pages/)
├── web/                    # Pyodide 静态 Web (index.html + js/ + py/bridge.py)
├── src/                    # 核心库: data_io/ modeling/ preprocessing/ results/ visualization/
├── tests/                  # pytest (conftest.py + unit/ + integration/)
├── docs/                   # 用户手册/开发者指南/已知问题/安全审计/v1.1规划
└── scripts/                # benchmark.py, generate_gallery_json.py
```

## 架构

4 层 (Streamlit + Web 共享 Business Logic):
- **Presentation**: Streamlit pages / Web HTML+JS
- **Application**: session_state / Pyodide bridge
- **Business Logic**: data_io → preprocessing → modeling → results → visualization
- **Data**: 文件系统 (上传/导出) / 浏览器内存

数据流: 上传 → 解析 → 类型检测 → 变量选择 → 建模 → 结果 → 导出

## 技术选型

| 领域 | 选型 |
|------|------|
| UI | Streamlit + Pyodide/HTML/JS |
| 统计引擎 | statsmodels (OLS + Logit) |
| 图表 | plotly (交互) + matplotlib (静态) |
| 包管理 | uv |
| 测试 | pytest + pytest-cov |
| 导出 | openpyxl, kaleido, zipfile |

## 已知问题

详见 `docs/已知问题.md`。核心: (1) Windows 终端 GBK 编码限制 (2) Web bridge categorical×numeric 交互走 pd.get_dummies 非 patsy (3) ModelSpec.interaction_terms 仅支持 2-way pairs

## 测试与开发

```bash
uv run python -m pytest tests/ -v              # 599 tests
uv run python -m pytest tests/unit/test_XXX.py -v
uv run streamlit run app/app.py                # 启动 Streamlit
bash web/deploy.sh                              # 部署 Web 版
```

## 下个 Session 建议 (v1.2+)

1. Probit / 多层次模型 / 面板数据 / Poisson / NegativeBinomial 回归
2. Web bridge categorical 交互支持
3. Docker 部署 + CI/CD
