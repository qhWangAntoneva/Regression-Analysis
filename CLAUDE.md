# Regression Analysis — CLAUDE.md

> 生效于所有 Agent 会话。规则优先级：HANDOVER.md > CLAUDE.md > 单次 prompt。

## Worktree 隔离 — 绝对禁令

**禁止在此项目使用 EnterWorktree 或 Agent `isolation: "worktree"`。**

原因：2026-05-25 经 3-agent 并行审计确认，`_cleanup_worktree()` 源码检查 worktree 分支推送状态——未推送则删除整个 worktree 目录。过去 3 次 session 中 agent 全部提交但**从未推送 worktree 分支**（流程是"提交到 worktree → 合并到 master → 推送 master"），worktree 分支推送率 0%。如果清理曾运行过，3 次 session 100% 丢失。

**这不是 bug，是设计**：源码注释明确 "agent work lives in commits, not in the working tree"。只有已推送的 commit 能存活。未提交的修改 = 直接丢弃。

**没有技术护栏**：此项目无 settings.json hook 阻止 EnterWorktree 调用，禁令是纯文本指令。违反此规则 → 工作必定丢失。

详见 HANDOVER.md "Worktree 隔离安全审计" 章节。

## Agent 协作模式

### 默认模式：直接在 master 工作

Agent 直接修改 master 分支。不创建 feature 分支，不启用 worktree 隔离。

### Pre-agent checkpoint（启动时强制执行）

**在对代码做任何修改之前**，必须先执行：

```
git add -A && git commit -m "checkpoint: pre-agent $(date -Iseconds)"
```

这是唯一的回滚安全网。如果没有 checkpoint commit，坏输出无法干净撤销。

回滚方法：
```
git reset --hard HEAD~1    # 完全回滚到 agent 修改前
git reset --soft HEAD~1    # 回滚 commit 但保留文件（可挑拣保留）
```

### 并行 agent（仅在必须时使用）

仅当用户明确要求同时运行多个 agent 时，才使用手动 feature 分支：

```
git checkout -b feature/xxx
# agent 工作...
git checkout master && git merge feature/xxx
git branch -d feature/xxx
```

**禁止**用 worktree 隔离实现并行。

## 安全红线

- **禁止 `git push --force`** — GitHub 仓库为私有免费版，无分支保护，force push 不可逆
- **禁止 EnterWorktree / `isolation: "worktree"`** — 见顶部禁令
- **提交前必须跑测试**：`uv run python -m pytest tests/ -v`（当前 309 tests）
- **修改后必须 push**：commit 后立即 `git push`，确保远程有备份
- **禁止修改 `.gitignore` 中的 `.claude/` 排除规则**

## 工作结束检查清单

```
[ ] git add -A && git commit -m "描述性提交信息"
[ ] git push
[ ] git status  # 确认 working tree clean
[ ] uv run python -m pytest tests/ -v  # 全部通过
```

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
