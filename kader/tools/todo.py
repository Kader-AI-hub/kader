import asyncio
import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from kader.tools.base import BaseTool, ParameterSchema, ToolCategory
from kader.utils.todo_metadata import TodoMetadataHandler


class TodoStatus(str, Enum):
    """Status of a todo item."""

    NOT_STARTED = "not-started"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"


class TodoItem(BaseModel):
    """A single item in a todo list."""

    task: str
    status: TodoStatus = TodoStatus.NOT_STARTED


class TodoToolInput(BaseModel):
    """Input model for TodoTool."""

    action: Literal["create", "read", "update", "delete"]
    todo_id: str = Field(
        ..., min_length=1, description="Unique identifier for the todo list"
    )
    items: Optional[list[TodoItem]] = Field(
        None, description="List of items for create/update"
    )
    session_id: Optional[str] = Field(None, description="Session ID override")


class TodoTool(BaseTool[str]):
    """
    Tool for managing todo lists associated with an agent's memory session.

    Allows creating, reading, updating, and deleting todo lists.
    Data is stored as JSON files in ~/.kader/memory/sessions/<session_id>/todos/.
    """

    def __init__(self) -> None:
        """Initialize the TodoTool."""
        super().__init__(
            name="todo_tool",
            description=(
                "Manage todo lists for planning. "
                "Supports creating, reading, updating, and deleting todo lists. "
                "Each list is identified by a todo_id and contains items with status "
                "(not-started, in-progress, completed). "
                "IMPORTANT: When updating, you can ONLY change the status of existing items. "
                "You cannot add, remove, or modify task descriptions. "
                "If you need to change tasks, delete and recreate the list."
            ),
            category=ToolCategory.UTILITY,
            parameters=[
                ParameterSchema(
                    name="action",
                    type="string",
                    description="Action to perform: create, read, update, delete",
                    enum=["create", "read", "update", "delete"],
                ),
                ParameterSchema(
                    name="todo_id",
                    type="string",
                    description="ID of the todo list",
                ),
                ParameterSchema(
                    name="items",
                    type="array",
                    description="List of todo items (for create/update). Each item has 'task' and optional 'status'.",
                    items_type="object",
                    required=False,
                    properties=[
                        ParameterSchema(
                            name="task", type="string", description="Task description"
                        ),
                        ParameterSchema(
                            name="status",
                            type="string",
                            description="Status",
                            enum=["not-started", "in-progress", "completed"],
                            required=False,
                        ),
                    ],
                ),
                ParameterSchema(
                    name="session_id",
                    type="string",
                    description="Optional session ID to override the current agent session",
                    required=False,
                ),
            ],
        )
        self._metadata_handler = TodoMetadataHandler()
        self._current_todo_id: str | None = None

    @property
    def current_todo_id(self) -> str | None:
        """Get the most recently created todo ID."""
        return self._current_todo_id

    def _schedule_metadata(
        self,
        session_id: str,
        todo_id: str,
        items: list[dict[str, Any]] | None,
        action: Literal["create", "update", "delete"],
    ) -> None:
        """Schedule metadata update as a non-blocking background task."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._metadata_handler.handle_todo_change(
                    session_id, todo_id, items, action
                )
            )
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(
                        self._metadata_handler.handle_todo_change(
                            session_id, todo_id, items, action
                        )
                    )
                else:
                    loop.run_in_executor(
                        None,
                        self._run_sync_metadata,
                        session_id,
                        todo_id,
                        items,
                        action,
                    )
            except RuntimeError:
                pass

    def _run_sync_metadata(
        self,
        session_id: str,
        todo_id: str,
        items: list[dict[str, Any]] | None,
        action: Literal["create", "update", "delete"],
    ) -> None:
        """Run metadata update synchronously (fallback for non-async contexts)."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                self._metadata_handler.handle_todo_change(
                    session_id, todo_id, items, action
                )
            )
        finally:
            loop.close()

    def execute(self, **kwargs: Any) -> str:
        """
        Execute the todo tool.

        Args:
            **kwargs: Arguments matching TodoToolInput

        Returns:
            JSON string results or success message
        """
        try:
            # Validate input using Pydantic
            # We handle potential dict items in 'items' list automatically via Pydantic parsing
            input_data = TodoToolInput(**kwargs)
        except ValidationError as e:
            return f"Input Validation Error: {e}"

        # Resolve session ID
        session_id = input_data.session_id or self._session_id
        if not session_id:
            return "Error: No session ID available. ensure the agent is running with a session or provide 'session_id'."

        try:
            if input_data.action == "create":
                result = self._create_todo(
                    session_id, input_data.todo_id, input_data.items
                )
                if not result.startswith("Error:"):
                    items_dicts = (
                        [item.model_dump() for item in input_data.items]
                        if input_data.items
                        else []
                    )
                    self._schedule_metadata(
                        session_id, input_data.todo_id, items_dicts, "create"
                    )
                return result
            elif input_data.action == "read":
                return self._read_todo(session_id, input_data.todo_id)
            elif input_data.action == "update":
                result = self._update_todo(
                    session_id, input_data.todo_id, input_data.items
                )
                if not result.startswith("Error:"):
                    items_dicts = (
                        [item.model_dump() for item in input_data.items]
                        if input_data.items
                        else []
                    )
                    self._schedule_metadata(
                        session_id, input_data.todo_id, items_dicts, "update"
                    )
                return result
            elif input_data.action == "delete":
                result = self._delete_todo(session_id, input_data.todo_id)
                if not result.startswith("Error:"):
                    self._schedule_metadata(
                        session_id, input_data.todo_id, None, "delete"
                    )
                return result
            else:
                return f"Error: Unknown action '{input_data.action}'"
        except Exception as e:
            return f"Error executing todo action '{input_data.action}': {str(e)}"

    async def aexecute(self, **kwargs: Any) -> str:
        """Asynchronous execution (delegates to synchronous for now)."""
        return self.execute(**kwargs)

    def get_interruption_message(self, action: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute todo_{action}"

    def _get_todo_path(self, session_id: str, todo_id: str) -> Path:
        """Get the file path for a todo list."""
        # Hardcoded base path as per requirements: ~/.kader/memory/sessions/
        base_dir = Path.home() / ".kader" / "memory" / "sessions"
        todo_dir = base_dir / session_id / "todos"
        todo_dir.mkdir(parents=True, exist_ok=True)
        return todo_dir / f"{todo_id}.json"

    def _create_todo(
        self, session_id: str, todo_id: str, items: list[TodoItem] | None
    ) -> str:
        """Create a new todo list."""
        path = self._get_todo_path(session_id, todo_id)
        if path.exists():
            return f"Error: Todo list '{todo_id}' already exists."

        items_list = items or []
        data = [item.model_dump() for item in items_list]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        self._current_todo_id = todo_id

        return json.dumps(data, indent=2)

    def _read_todo(self, session_id: str, todo_id: str) -> str:
        """Read a todo list."""
        path = self._get_todo_path(session_id, todo_id)
        if not path.exists():
            return f"Error: Todo list '{todo_id}' not found."

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return json.dumps(data, indent=2)
            except json.JSONDecodeError:
                return "Error: Failed to decode todo list JSON."

    def _update_todo(
        self, session_id: str, todo_id: str, items: list[TodoItem] | None
    ) -> str:
        """Update an existing todo list (status changes only).
        This method enforces integrity by only allowing status updates.
        The task descriptions must match the existing todo list exactly.
        """
        path = self._get_todo_path(session_id, todo_id)
        if not path.exists():
            return f"Error: Todo list '{todo_id}' not found. Use 'create' to make a new list."

        if items is None:
            return "Error: 'items' must be provided for update action."

        # Read existing todo list
        with open(path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                return "Error: Failed to decode existing todo list JSON."

        # Validate integrity: check that task descriptions match
        existing_tasks = [item.get("task", "") for item in existing_data]
        new_tasks = [item.task for item in items]

        # Check if the number of items matches
        if len(existing_tasks) != len(new_tasks):
            return self._format_integrity_error(todo_id, existing_data)

        # Check if all task descriptions match (order matters)
        if existing_tasks != new_tasks:
            return self._format_integrity_error(todo_id, existing_data)

        # Validation passed - update with new statuses
        data = [item.model_dump() for item in items]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return json.dumps(data, indent=2)

    def _format_integrity_error(self, todo_id: str, existing_data: list[dict]) -> str:
        """Format an integrity error message with the current todo list content."""
        items_description = "\n".join(
            f"  {i + 1}. [{item.get('status', 'not-started')}] {item.get('task', '')}"
            for i, item in enumerate(existing_data)
        )
        return (
            f"Error: Update rejected - todo list integrity violation.\n"
            f"The provided items do not match the existing todo list '{todo_id}'.\n"
            f"You can only update the STATUS of existing items, not add, remove, or modify task descriptions.\n\n"
            f"Current todo list '{todo_id}' content:\n{items_description}\n\n"
            f"Please update using the exact task descriptions from the list above, "
            f"only changing the 'status' field as needed."
        )

    def _delete_todo(self, session_id: str, todo_id: str) -> str:
        """Delete a todo list."""
        path = self._get_todo_path(session_id, todo_id)
        if not path.exists():
            return f"Error: Todo list '{todo_id}' not found."

        path.unlink()
        return f"Successfully deleted todo list '{todo_id}'."
