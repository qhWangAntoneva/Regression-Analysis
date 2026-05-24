# Regression Analysis — CLAUDE.md

> 生效于所有 Agent 会话。规则优先级：HANDOVER.md > CLAUDE.md > 单次 prompt。

## Agent 协作模式

### 默认模式：直接在 master 工作

Agent 直接修改 master 分支，不启用 worktree 隔离。

### Pre-agent checkpoint

Agent 开始修改前，自动执行：
```
git add -A && git commit -m "checkpoint: pre-agent $(date -Iseconds)"
```

如果 agent 输出有问题：
```
git reset --hard HEAD~1    # 完全回滚
git reset --soft HEAD~1    # 回滚 commit 但保留修改
```

### 并行 agent（expert mode）

仅在需要同时运行多个 agent 时使用手动 feature 分支：
```
git checkout -b feature/xxx
# agent 工作...
git checkout master && git merge feature/xxx
git branch -d feature/xxx
```

## 安全红线

- **禁止 `git push --force`** — GitHub 无分支保护，force push 会永久破坏远程历史
- **提交前必须跑测试**：`uv run python -m pytest tests/ -v`
- **不要启用 worktree 隔离** — 2026-05-25 调查证实 _cleanup_worktree() 会删除未推送 worktree 分支的提交。过去 3 次 session 中 agent 全部提交但从未推送 worktree 分支，如果清理曾运行过将 100% 丢失

## 开发命令

```bash
uv run streamlit run app/app.py        # 启动 Streamlit 应用
uv run python -m pytest tests/ -v      # 运行全部 309 tests
uv run python -m pytest tests/unit/test_XXX.py -v  # 单个测试文件
```

## 关键技术约束

- Python：`uv run python`，不用裸 `python`
- 编码：所有 `open()` 必须显式 `encoding='utf-8'`
- Windows 路径：跨 Git Bash / Python 传递文件用完整 Windows 路径
- 图表 PNG 导出依赖 kaleido >= 1.3.0
