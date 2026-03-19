import json
from pathlib import Path

import pytest

from kader.tools.todo import TodoTool
from kader.utils.todo_metadata import TodoMetadataHandler


@pytest.fixture
def metadata_handler(tmp_path):
    """Fixture to create TodoMetadataHandler with a temporary directory."""

    class TestMetadataHandler(TodoMetadataHandler):
        def _get_metadata_path(self, session_id: str):
            todo_dir = (
                tmp_path / ".kader" / "memory" / "sessions" / session_id / "todos"
            )
            todo_dir.mkdir(parents=True, exist_ok=True)
            return todo_dir / f"metadata-{session_id}.json"

    return TestMetadataHandler()


@pytest.mark.asyncio
async def test_compute_todo_stats_all_completed(metadata_handler):
    """Test stats computation when all tasks are completed."""
    items = [
        {"task": "Task 1", "status": "completed"},
        {"task": "Task 2", "status": "completed"},
    ]
    stats = metadata_handler._compute_todo_stats(items)
    assert stats["completed"] is True
    assert stats["remaining-tasks"] == 0
    assert stats["completed-tasks"] == 2


@pytest.mark.asyncio
async def test_compute_todo_stats_none_completed(metadata_handler):
    """Test stats computation when no tasks are completed."""
    items = [
        {"task": "Task 1", "status": "not-started"},
        {"task": "Task 2", "status": "in-progress"},
    ]
    stats = metadata_handler._compute_todo_stats(items)
    assert stats["completed"] is False
    assert stats["remaining-tasks"] == 2
    assert stats["completed-tasks"] == 0


@pytest.mark.asyncio
async def test_compute_todo_stats_partial_completed(metadata_handler):
    """Test stats computation when some tasks are completed."""
    items = [
        {"task": "Task 1", "status": "completed"},
        {"task": "Task 2", "status": "not-started"},
        {"task": "Task 3", "status": "in-progress"},
    ]
    stats = metadata_handler._compute_todo_stats(items)
    assert stats["completed"] is False
    assert stats["remaining-tasks"] == 2
    assert stats["completed-tasks"] == 1


@pytest.mark.asyncio
async def test_compute_todo_stats_empty_list(metadata_handler):
    """Test stats computation when the task list is empty."""
    items = []
    stats = metadata_handler._compute_todo_stats(items)
    assert stats["completed"] is False
    assert stats["remaining-tasks"] == 0
    assert stats["completed-tasks"] == 0


@pytest.mark.asyncio
async def test_handle_create(metadata_handler):
    """Test metadata update on todo creation."""
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-1",
        [
            {"task": "Task 1", "status": "not-started"},
            {"task": "Task 2", "status": "completed"},
        ],
        "create",
    )

    metadata = await metadata_handler._load_metadata("session-1")
    assert metadata["session-id"] == "session-1"
    assert "todo-1" in metadata["plans"]
    assert metadata["plans"]["todo-1"]["completed"] is False
    assert metadata["plans"]["todo-1"]["remaining-tasks"] == 1
    assert metadata["plans"]["todo-1"]["completed-tasks"] == 1


@pytest.mark.asyncio
async def test_handle_create_with_empty_items(metadata_handler):
    """Test metadata update on todo creation with no items."""
    await metadata_handler.handle_todo_change("session-1", "todo-empty", [], "create")

    metadata = await metadata_handler._load_metadata("session-1")
    assert metadata["plans"]["todo-empty"]["completed"] is False
    assert metadata["plans"]["todo-empty"]["remaining-tasks"] == 0
    assert metadata["plans"]["todo-empty"]["completed-tasks"] == 0


@pytest.mark.asyncio
async def test_handle_update(metadata_handler):
    """Test metadata update on todo update."""
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-1",
        [
            {"task": "Task 1", "status": "not-started"},
            {"task": "Task 2", "status": "not-started"},
        ],
        "create",
    )

    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-1",
        [
            {"task": "Task 1", "status": "completed"},
            {"task": "Task 2", "status": "completed"},
        ],
        "update",
    )

    metadata = await metadata_handler._load_metadata("session-1")
    assert metadata["plans"]["todo-1"]["completed"] is True
    assert metadata["plans"]["todo-1"]["remaining-tasks"] == 0
    assert metadata["plans"]["todo-1"]["completed-tasks"] == 2


@pytest.mark.asyncio
async def test_handle_delete(metadata_handler):
    """Test metadata update on todo deletion."""
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-1",
        [{"task": "Task 1", "status": "not-started"}],
        "create",
    )

    await metadata_handler.handle_todo_change("session-1", "todo-1", None, "delete")

    metadata = await metadata_handler._load_metadata("session-1")
    assert "todo-1" not in metadata["plans"]


@pytest.mark.asyncio
async def test_multiple_todos_same_session(metadata_handler):
    """Test metadata tracking multiple todos in the same session."""
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-1",
        [{"task": "Task 1", "status": "completed"}],
        "create",
    )
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-2",
        [
            {"task": "Task 1", "status": "completed"},
            {"task": "Task 2", "status": "not-started"},
        ],
        "create",
    )

    metadata = await metadata_handler._load_metadata("session-1")
    assert len(metadata["plans"]) == 2
    assert metadata["plans"]["todo-1"]["completed"] is True
    assert metadata["plans"]["todo-2"]["completed"] is False
    assert metadata["plans"]["todo-2"]["remaining-tasks"] == 1


@pytest.mark.asyncio
async def test_update_nonexistent_todo(metadata_handler):
    """Test that updating a non-existent todo does not crash."""
    await metadata_handler.handle_todo_change(
        "session-1",
        "todo-does-not-exist",
        [{"task": "Task 1", "status": "completed"}],
        "update",
    )

    metadata = await metadata_handler._load_metadata("session-1")
    assert "todo-does-not-exist" not in metadata["plans"]


@pytest.fixture
def todo_tool_with_metadata(tmp_path):
    """Fixture to create TodoTool with a temporary home directory."""

    class TestTodoTool(TodoTool):
        def _get_todo_path(self, session_id: str, todo_id: str) -> Path:
            base_dir = tmp_path / ".kader" / "memory" / "sessions"
            todo_dir = base_dir / session_id / "todos"
            todo_dir.mkdir(parents=True, exist_ok=True)
            return todo_dir / f"{todo_id}.json"

    class TestMetadataHandler(TodoMetadataHandler):
        def _get_metadata_path(self, session_id: str) -> Path:
            base_dir = tmp_path / ".kader" / "memory" / "sessions"
            todo_dir = base_dir / session_id / "todos"
            todo_dir.mkdir(parents=True, exist_ok=True)
            return todo_dir / f"metadata-{session_id}.json"

    tool = TestTodoTool()
    tool._metadata_handler = TestMetadataHandler()
    tool.set_session_id("test-session")
    return tool


@pytest.mark.asyncio
async def test_metadata_created_on_todo_create(todo_tool_with_metadata):
    """Test that metadata is created when a todo is created."""
    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-1",
        [
            {"task": "Task 1", "status": "not-started"},
            {"task": "Task 2", "status": "completed"},
        ],
        "create",
    )

    metadata_path = todo_tool_with_metadata._metadata_handler._get_metadata_path(
        "test-session"
    )
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["session-id"] == "test-session"
    assert "plan-1" in metadata["plans"]
    assert metadata["plans"]["plan-1"]["completed"] is False
    assert metadata["plans"]["plan-1"]["remaining-tasks"] == 1
    assert metadata["plans"]["plan-1"]["completed-tasks"] == 1


@pytest.mark.asyncio
async def test_metadata_updated_on_todo_update(todo_tool_with_metadata):
    """Test that metadata is updated when a todo is updated."""
    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-2",
        [
            {"task": "Task 1", "status": "not-started"},
            {"task": "Task 2", "status": "not-started"},
        ],
        "create",
    )

    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-2",
        [
            {"task": "Task 1", "status": "completed"},
            {"task": "Task 2", "status": "completed"},
        ],
        "update",
    )

    metadata_path = todo_tool_with_metadata._metadata_handler._get_metadata_path(
        "test-session"
    )
    metadata = json.loads(metadata_path.read_text())
    assert metadata["plans"]["plan-2"]["completed"] is True
    assert metadata["plans"]["plan-2"]["remaining-tasks"] == 0
    assert metadata["plans"]["plan-2"]["completed-tasks"] == 2


@pytest.mark.asyncio
async def test_metadata_deleted_on_todo_delete(todo_tool_with_metadata):
    """Test that metadata entry is removed when a todo is deleted."""
    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-3",
        [{"task": "Task 1", "status": "completed"}],
        "create",
    )

    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session", "plan-3", None, "delete"
    )

    metadata_path = todo_tool_with_metadata._metadata_handler._get_metadata_path(
        "test-session"
    )
    metadata = json.loads(metadata_path.read_text())
    assert "plan-3" not in metadata["plans"]


@pytest.mark.asyncio
async def test_metadata_read_action_does_not_trigger_update(todo_tool_with_metadata):
    """Test that reading a todo does not update metadata."""
    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-read",
        [{"task": "Task 1", "status": "completed"}],
        "create",
    )

    metadata_path = todo_tool_with_metadata._metadata_handler._get_metadata_path(
        "test-session"
    )
    metadata = json.loads(metadata_path.read_text())
    assert "plan-read" in metadata["plans"]
    assert metadata["plans"]["plan-read"]["completed"] is True


@pytest.mark.asyncio
async def test_metadata_multiple_todos_same_session(todo_tool_with_metadata):
    """Test metadata tracking multiple todos in the same session."""
    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-a",
        [{"task": "Task 1", "status": "completed"}],
        "create",
    )

    await todo_tool_with_metadata._metadata_handler.handle_todo_change(
        "test-session",
        "plan-b",
        [
            {"task": "Task 1", "status": "completed"},
            {"task": "Task 2", "status": "not-started"},
        ],
        "create",
    )

    metadata_path = todo_tool_with_metadata._metadata_handler._get_metadata_path(
        "test-session"
    )
    metadata = json.loads(metadata_path.read_text())
    assert len(metadata["plans"]) == 2
    assert metadata["plans"]["plan-a"]["completed"] is True
    assert metadata["plans"]["plan-b"]["completed"] is False
    assert metadata["plans"]["plan-b"]["remaining-tasks"] == 1
