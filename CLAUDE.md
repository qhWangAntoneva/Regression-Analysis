# Regression Analysis — CLAUDE.md

> 规则优先级：HANDOVER.md > CLAUDE.md > 单次 prompt

## Worktree 隔离 — 绝对禁令

**禁止在此项目使用 EnterWorktree 或 Agent `isolation: "worktree"`。**

原因：worktree 清理逻辑要求分支已推送才能保留。此项目的 agent 工作流是"提交到 worktree → 合并到 master → 推送 master"，worktree 分支从未推送 → 清理时 100% 丢失。这不是 bug，是系统设计要求。

详见 HANDOVER.md。

## Agent 协作模式

**默认直接在 master 工作。** 不创建分支，不启用 worktree。

改代码前先 checkpoint：
```
git add -A && git commit -m "checkpoint: pre-agent $(date -Iseconds)"
```
回滚：`git reset --hard HEAD~1`（完全）/ `git reset --soft HEAD~1`（保留文件）

并行 agent（仅用户明确要求时）：手动 `git checkout -b feature/xxx` → merge → delete。禁止 worktree。

## 安全红线

- 禁止 `git push --force`
- 禁止修改 `.gitignore` 中 `.claude/` 排除规则
- 提交前跑测试 + commit 后立即 push

## 工作结束检查

```
[ ] git add -A && git commit -m "..."
[ ] git push
[ ] git status  # clean
[ ] uv run python -m pytest tests/ -v  # 599 tests
```

## 开发命令

```bash
uv run streamlit run app/app.py        # 启动
uv run python -m pytest tests/ -v      # 测试
```

## 项目特有约束

- 图表 PNG 导出依赖 kaleido >= 1.3.0
- Python 通用规则（uv、编码、路径）见全局 CLAUDE.md
