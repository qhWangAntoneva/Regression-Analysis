"""
会话状态持久化模块

提供 save_session / load_session / clear_session 功能，
用于在 Streamlit 重启后恢复上次的会话状态。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# 默认缓存路径（相对于项目根目录）
DEFAULT_SAVE_PATH = ".session_cache"


def _get_project_root() -> Path:
    """获取项目根目录（包含 .git 或 pyproject.toml 的目录）。"""
    cwd = Path.cwd()
    # 向上查找包含 pyproject.toml 或 .git 的目录
    for parent in [cwd] + list(cwd.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return cwd


def save_session(state: dict[str, Any], save_path: str = DEFAULT_SAVE_PATH) -> None:
    """将 session_state 的相关数据保存到 JSON 文件。

    仅保存可序列化的键值对（排除大型 DataFrame、模型对象等）。

    Args:
        state: session_state 字典（如 st.session_state 的键子集）。
        save_path: 保存路径（相对于项目根目录），默认为 '.session_cache'。
    """
    project_root = _get_project_root()
    filepath = project_root / save_path

    # 仅保留可 JSON 序列化的值
    serializable: dict[str, Any] = {}
    for key, value in state.items():
        try:
            # 尝试序列化以验证可 JSON 化
            json.dumps(value)
            serializable[key] = value
        except (TypeError, ValueError, OverflowError):
            # 跳过无法序列化的对象（DataFrame、模型结果等）
            serializable[key] = _safe_serialize(value)

    data = {
        "_version": "1.0",
        "state": serializable,
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _safe_serialize(value: Any) -> Any:
    """安全尝试将不可序列化的值转为基本类型。"""
    try:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return str(value)
        return str(value)
    except Exception:
        return str(value)


def load_session(save_path: str = DEFAULT_SAVE_PATH) -> dict[str, Any]:
    """从 JSON 文件加载保存的会话状态。

    Args:
        save_path: 保存路径（相对于项目根目录），默认为 '.session_cache'。

    Returns:
        保存的状态字典，若文件不存在则返回空字典。
    """
    project_root = _get_project_root()
    filepath = project_root / save_path

    if not filepath.exists():
        return {}

    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("state", {})
    except (json.JSONDecodeError, OSError):
        return {}


def clear_session(save_path: str = DEFAULT_SAVE_PATH) -> None:
    """删除保存的会话状态文件。

    Args:
        save_path: 保存路径（相对于项目根目录），默认为 '.session_cache'。
    """
    project_root = _get_project_root()
    filepath = project_root / save_path

    if filepath.exists():
        filepath.unlink()


def session_cache_exists(save_path: str = DEFAULT_SAVE_PATH) -> bool:
    """检查会话缓存文件是否存在。

    Args:
        save_path: 保存路径（相对于项目根目录），默认为 '.session_cache'。

    Returns:
        缓存文件是否存在。
    """
    project_root = _get_project_root()
    return (project_root / save_path).exists()
