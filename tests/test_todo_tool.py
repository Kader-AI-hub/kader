import json
from pathlib import Path

import pytest

from kader.tools.todo import TodoTool


@pytest.fixture
def todo_tool(tmp_path):
    """Fixture to create TodoTool with a temporary home directory."""
    # We need to mock Path.home() or change where get_todo_path points to.
    # Since _get_todo_path uses Path.home(), we should monkeypatch it or
    # make the base path configurable.
    # For this test, let's subclass TodoTool to override _get_todo_path for testing

    class TestTodoTool(TodoTool):
        def _get_todo_path(self, session_id: str, todo_id: str) -> Path:
            base_dir = tmp_path / ".kader" / "memory" / "sessions"
            todo_dir = base_dir / session_id / "todos"
            todo_dir.mkdir(parents=True, exist_ok=True)
            return todo_dir / f"{todo_id}.json"

    tool = TestTodoTool()
    tool.set_session_id("test-session")
    return tool


def test_create_todo(todo_tool):
    """Test creating a todo list."""
    todo_id = "list-1"
    items = [
        {"task": "Task 1", "status": "not-started"},
        {"task": "Task 2", "status": "in-progress"},
    ]

    result = todo_tool.execute(action="create", todo_id=todo_id, items=items)

    assert "Successfully created" in result

    # Verify file content
    result_read = todo_tool.execute(action="read", todo_id=todo_id)
    data = json.loads(result_read)
    assert len(data) == 2
    assert data[0]["task"] == "Task 1"
    assert data[0]["status"] == "not-started"


def test_read_todo_not_found(todo_tool):
    """Test reading a non-existent todo list."""
    result = todo_tool.execute(action="read", todo_id="non-existent")
    assert "Error: Todo list 'non-existent' not found" in result


def test_update_todo(todo_tool):
    """Test updating a todo list."""
    todo_id = "list-2"
    items = [{"task": "Init", "status": "not-started"}]
    todo_tool.execute(action="create", todo_id=todo_id, items=items)

    new_items = [
        {"task": "Init", "status": "completed"},
        {"task": "New Task", "status": "not-started"},
    ]

    result = todo_tool.execute(action="update", todo_id=todo_id, items=new_items)
    assert "Successfully updated" in result

    result_read = todo_tool.execute(action="read", todo_id=todo_id)
    data = json.loads(result_read)
    assert len(data) == 2
    assert data[0]["status"] == "completed"


def test_delete_todo(todo_tool):
    """Test deleting a todo list."""
    todo_id = "list-3"
    todo_tool.execute(action="create", todo_id=todo_id, items=[])

    result = todo_tool.execute(action="delete", todo_id=todo_id)
    assert "Successfully deleted" in result

    result_read = todo_tool.execute(action="read", todo_id=todo_id)
    assert "not found" in result_read


def test_session_id_override(todo_tool):
    """Test overriding session ID."""
    # This should act on a different session
    result = todo_tool.execute(
        action="create",
        todo_id="other-session-list",
        items=[],
        session_id="other-session",
    )
    assert "Successfully created" in result

    # Read back with override
    result_read = todo_tool.execute(
        action="read", todo_id="other-session-list", session_id="other-session"
    )
    assert "[]" in result_read  # empty list JSON
