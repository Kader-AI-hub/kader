import json
from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import ScrollableContainer
from textual.widgets import Static

from kader.tools.todo import TodoStatus


class TodoListItem(Static):
    """A single item in the todo list."""

    def __init__(self, task: str, status: TodoStatus) -> None:
        super().__init__()
        self.todo_task = task
        self.status = status

    def compose(self) -> ComposeResult:
        icon = {
            TodoStatus.NOT_STARTED: "( )",
            TodoStatus.IN_PROGRESS: "(*)",
            TodoStatus.COMPLETED: "(+)",
        }.get(self.status, "(?)")

        # Determine style class based on status
        classes = f"todo-item {self.status.value}"

        yield Static(f"{icon} {self.todo_task}", classes=classes, markup=False)


class TodoList(ScrollableContainer):
    """Widget to display the current TODO list."""

    DEFAULT_CSS = """
    TodoList {
        height: 1fr;
        border-top: solid #a855f7;
        background: #161622;
        padding: 0 1;
        overflow: auto auto;
    }

    .todo-list-title {
        background: rgba(168, 85, 247, 0.2);
        color: #cdd6f4;
        text-style: bold;
        padding: 1;
        text-align: center;
        dock: top;
    }

    .todo-item {
        padding: 0 1;
        margin-bottom: 1;
        color: #6c7086;
    }

    .todo-item.not-started {
        color: #6c7086;
    }

    .todo-item.in-progress {
        color: #facc15;
        text-style: bold;
        background: rgba(250, 204, 21, 0.05);
        border: solid rgba(250, 204, 21, 0.2);
        padding: 1;
    }

    .todo-item.completed {
        color: #10b981;
        text-style: strike;
    }

    .todo-message {
        color: #6c7086;
        text-style: italic;
        padding: 1;
    }

    .todo-error {
        color: #f87171;
        padding: 1;
    }

    .todo-file-name {
        color: #a855f7;
        text-style: bold;
        padding: 0 1;
        margin-bottom: 1;
    }
    """

    def __init__(self, id: Optional[str] = None) -> None:
        super().__init__(id=id)
        self._session_id: Optional[str] = None

    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID and refresh."""
        self._session_id = session_id
        self.refresh_todos()

    def refresh_todos(self) -> None:
        """Reload todos from file in background."""
        # Show loading state if empty
        if not self.children:
            self.mount(Static("Loading...", classes="todo-message"))

        self.run_worker(self._load_todos_bg, thread=True)

    def _load_todos_bg(self) -> None:
        """Background worker to load todo data."""
        if not self._session_id:
            self.app.call_from_thread(self._update_todos_ui, None, "No active session.")
            return

        base_dir = Path.home() / ".kader" / "memory" / "sessions"
        todo_dir = base_dir / self._session_id / "todos"

        if not todo_dir.exists():
            self.app.call_from_thread(self._update_todos_ui, None, "No todos found.")
            return

        files = list(todo_dir.glob("*.json"))
        if not files:
            self.app.call_from_thread(self._update_todos_ui, None, "No todos found.")
            return

        # Sort by modification time (descending)
        latest_file = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.app.call_from_thread(
                self._update_todos_ui, data, None, latest_file.stem
            )
        except Exception as e:
            self.app.call_from_thread(
                self._update_todos_ui, None, f"Error loading todos: {str(e)}"
            )

    def _update_todos_ui(
        self,
        data: Optional[list],
        error_message: Optional[str] = None,
        list_name: Optional[str] = None,
    ) -> None:
        """Update UI on main thread."""
        self.remove_children()

        if error_message:
            self.mount(
                Static(
                    error_message,
                    classes="todo-message"
                    if "Error" not in error_message
                    else "todo-error",
                )
            )
            return

        if not data:
            self.mount(Static("Empty todo list.", classes="todo-message"))
            return

        if list_name:
            self.mount(Static(list_name, classes="todo-file-name"))

        for item in data:
            task = item.get("task", "Unknown task")
            status_str = item.get("status", "not-started")
            try:
                status = TodoStatus(status_str)
            except ValueError:
                status = TodoStatus.NOT_STARTED

            self.mount(TodoListItem(task, status))
