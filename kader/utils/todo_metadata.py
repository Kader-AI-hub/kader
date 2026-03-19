"""
Todo metadata handler for tracking plan completion stats.

Manages a metadata JSON file per session that stores completion statistics
for all todo lists in that session.
"""

from pathlib import Path
from typing import Any, Literal

from kader.memory.types import aload_json, asave_json, get_default_memory_dir


class TodoMetadataHandler:
    """
    Manages metadata for todo lists in a session.

    Tracks completion statistics for each todo list and stores them
    in a metadata JSON file alongside the todo files.

    The metadata file is stored at:
    ~/.kader/memory/sessions/<session_id>/todos/metadata-<session_id>.json

    Format:
    {
        "session-id": "<session_id>",
        "plans": {
            "<todo_id>": {
                "completed": <bool>,
                "remaining-tasks": <int>,
                "completed-tasks": <int>
            }
        }
    }
    """

    def __init__(self) -> None:
        """Initialize the TodoMetadataHandler."""
        pass

    def _get_metadata_path(self, session_id: str) -> Path:
        """Get the file path for the session metadata file."""
        base_dir = get_default_memory_dir() / "sessions"
        todo_dir = base_dir / session_id / "todos"
        todo_dir.mkdir(parents=True, exist_ok=True)
        return todo_dir / f"metadata-{session_id}.json"

    def _compute_todo_stats(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute statistics for a todo list.

        Args:
            items: List of todo items with 'task' and 'status' fields

        Returns:
            Dict with 'completed', 'remaining-tasks', and 'completed-tasks' values
        """
        completed_count = sum(1 for item in items if item.get("status") == "completed")
        total_count = len(items)
        remaining_count = total_count - completed_count

        return {
            "completed": completed_count == total_count and total_count > 0,
            "remaining-tasks": remaining_count,
            "completed-tasks": completed_count,
        }

    async def _load_metadata(self, session_id: str) -> dict[str, Any]:
        """
        Load existing metadata or create new structure.

        Args:
            session_id: Session identifier

        Returns:
            Metadata dictionary with session-id and plans
        """
        path = self._get_metadata_path(session_id)
        if path.exists():
            data = await aload_json(path)
            if "session-id" in data and "plans" in data:
                return data
        return {"session-id": session_id, "plans": {}}

    async def _save_metadata(self, session_id: str, data: dict[str, Any]) -> None:
        """
        Save metadata to disk.

        Args:
            session_id: Session identifier
            data: Complete metadata dictionary to save
        """
        path = self._get_metadata_path(session_id)
        await asave_json(path, data)

    async def _handle_create(
        self,
        session_id: str,
        todo_id: str,
        items: list[dict[str, Any]] | None,
    ) -> None:
        """
        Handle metadata update for todo creation.

        Args:
            session_id: Session identifier
            todo_id: Todo list identifier
            items: List of todo items
        """
        items = items or []
        metadata = await self._load_metadata(session_id)
        metadata["plans"][todo_id] = self._compute_todo_stats(items)
        await self._save_metadata(session_id, metadata)

    async def _handle_update(
        self,
        session_id: str,
        todo_id: str,
        items: list[dict[str, Any]] | None,
    ) -> None:
        """
        Handle metadata update for todo update.

        Args:
            session_id: Session identifier
            todo_id: Todo list identifier
            items: Updated list of todo items
        """
        if items is None:
            return
        metadata = await self._load_metadata(session_id)
        if todo_id in metadata["plans"]:
            metadata["plans"][todo_id] = self._compute_todo_stats(items)
            await self._save_metadata(session_id, metadata)

    async def _handle_delete(self, session_id: str, todo_id: str) -> None:
        """
        Handle metadata update for todo deletion.

        Args:
            session_id: Session identifier
            todo_id: Todo list identifier
        """
        metadata = await self._load_metadata(session_id)
        if todo_id in metadata["plans"]:
            del metadata["plans"][todo_id]
            await self._save_metadata(session_id, metadata)

    async def handle_todo_change(
        self,
        session_id: str,
        todo_id: str,
        items: list[dict[str, Any]] | None,
        action: Literal["create", "update", "delete"],
    ) -> None:
        """
        Public entry point for handling todo metadata changes.

        Dispatches to the appropriate handler based on action type.

        Args:
            session_id: Session identifier
            todo_id: Todo list identifier
            items: List of todo items (None for delete)
            action: One of 'create', 'update', or 'delete'
        """
        if action == "create":
            await self._handle_create(session_id, todo_id, items)
        elif action == "update":
            await self._handle_update(session_id, todo_id, items)
        elif action == "delete":
            await self._handle_delete(session_id, todo_id)
