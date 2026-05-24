# encoding: utf-8
"""Test the session persistence utility: save / load / clear."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.utils.persistence import (
    _get_project_root,
    clear_session,
    load_session,
    save_session,
    session_cache_exists,
)


@pytest.fixture
def temp_cache_path():
    """Create a temporary path for the cache file and clean up."""
    tmp_dir = Path(tempfile.mkdtemp())
    cache_file = tmp_dir / "test_cache.json"
    yield str(cache_file)
    if cache_file.exists():
        cache_file.unlink()
    tmp_dir.rmdir()


class TestSessionPersistence:
    """Tests for save_session / load_session / clear_session."""

    def test_save_session_creates_file(self, temp_cache_path):
        state = {"key1": "value1", "key2": 42}
        save_session(state, temp_cache_path)
        assert Path(temp_cache_path).exists()

    def test_save_session_file_content(self, temp_cache_path):
        state = {"key1": "value1", "key2": 42}
        save_session(state, temp_cache_path)

        with open(temp_cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["_version"] == "1.0"
        assert data["state"]["key1"] == "value1"
        assert data["state"]["key2"] == 42

    def test_load_session_returns_saved_data(self, temp_cache_path):
        state = {"key1": "hello", "key2": 123}
        save_session(state, temp_cache_path)

        loaded = load_session(temp_cache_path)
        assert loaded["key1"] == "hello"
        assert loaded["key2"] == 123

    def test_load_session_no_file(self, temp_cache_path):
        # File doesn't exist
        loaded = load_session(temp_cache_path + "_nonexistent")
        assert loaded == {}

    def test_clear_session_removes_file(self, temp_cache_path):
        state = {"key": "value"}
        save_session(state, temp_cache_path)
        assert Path(temp_cache_path).exists()

        clear_session(temp_cache_path)
        assert not Path(temp_cache_path).exists()

    def test_clear_session_no_file_does_not_error(self, temp_cache_path):
        # Clearing a non-existent file should not error
        clear_session(temp_cache_path + "_nonexistent")  # Should not raise

    def test_session_cache_exists_true(self, temp_cache_path):
        state = {"key": "value"}
        save_session(state, temp_cache_path)
        assert session_cache_exists(temp_cache_path) is True

    def test_session_cache_exists_false(self, temp_cache_path):
        assert session_cache_exists(temp_cache_path + "_nonexistent") is False

    def test_save_session_with_multiple_types(self, temp_cache_path):
        """Test serialization of various Python types."""
        state = {
            "string": "hello",
            "integer": 42,
            "float": 3.14,
            "bool_true": True,
            "bool_false": False,
            "none_val": None,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "tuple_list": [1, 2],  # tuple works as list index in JSON
        }
        save_session(state, temp_cache_path)
        loaded = load_session(temp_cache_path)

        assert loaded["string"] == "hello"
        assert loaded["integer"] == 42
        assert loaded["float"] == 3.14
        assert loaded["bool_true"] is True
        assert loaded["list"] == [1, 2, 3]
        assert loaded["dict"] == {"nested": "value"}

    def test_save_and_load_empty_state(self, temp_cache_path):
        save_session({}, temp_cache_path)
        loaded = load_session(temp_cache_path)
        assert loaded == {}

    def test_save_session_ignores_non_serializable(self, temp_cache_path):
        """Non-serializable objects should be converted to string representation."""
        class CustomObject:
            def __init__(self):
                self.value = 42

            def __str__(self):
                return f"CustomObject({self.value})"

        state = {"normal_key": "value", "custom": CustomObject()}
        # Should not raise
        save_session(state, temp_cache_path)
        loaded = load_session(temp_cache_path)
        assert loaded["normal_key"] == "value"
        assert "custom" in loaded

    def test_get_project_root_finds_pyproject(self):
        """_get_project_root should find the project root containing pyproject.toml."""
        root = _get_project_root()
        assert (root / "pyproject.toml").exists()

    def test_round_trip_preserves_types(self, temp_cache_path):
        """Test that common types survive a round-trip."""
        original = {
            "filename": "test.csv",
            "encoding": "UTF-8",
            "model_run_time": True,
            "version": 1,
        }
        save_session(original, temp_cache_path)
        loaded = load_session(temp_cache_path)

        assert loaded["filename"] == "test.csv"
        assert loaded["encoding"] == "UTF-8"
        assert loaded["model_run_time"] is True
        assert loaded["version"] == 1
