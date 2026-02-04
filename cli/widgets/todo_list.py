from pathlib import Path
import json
from typing import Optional

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static, Markdown

from kader.tools.todo import TodoStatus

class TodoListItem(Static):
    """A single item in the todo list."""

    def __init__(self, task: str, status: TodoStatus) -> None:
        super().__init__()
        self.todo_task = task
        self.status = status
        
    def compose(self) -> ComposeResult:
        icon = {
            TodoStatus.NOT_STARTED: "[ ]",
            TodoStatus.IN_PROGRESS: "[+]",
            TodoStatus.COMPLETED: "[x]",
        }.get(self.status, "[?]")
        
        # Determine style class based on status
        classes = f"todo-item {self.status.value}"
        
        yield Static(f"{icon} {self.todo_task}", classes=classes, markup=False)

class TodoList(VerticalScroll):
    """Widget to display the current TODO list."""

    DEFAULT_CSS = """
    TodoList {
        height: 1fr;
        border-top: solid $primary;
        background: $surface;
        padding: 0 1;
    }
    
    .todo-list-title {
        background: $primary 20%;
        color: $text;
        text-style: bold;
        padding: 1;
        text-align: center;
        dock: top;
    }

    .todo-item {
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
    }

    .todo-item.not-started {
        color: $text;
    }

    .todo-item.in-progress {
        color: $warning;
        text-style: bold;
    }

    .todo-item.completed {
        color: $success;
        text-style: strike;
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
        """Reload todos from file."""
        # Clear existing items
        self.remove_children()
        
        if not self._session_id:
            self.mount(Static("No active session.", classes="todo-message"))
            return

        # Path to todos
        # ~/.kader/memory/sessions/<session_id>/todos/
        base_dir = Path.home() / ".kader" / "memory" / "sessions"
        todo_dir = base_dir / self._session_id / "todos"
        
        if not todo_dir.exists():
            self.mount(Static("No todos found.", classes="todo-message"))
            return

        # Find most recent json file
        files = list(todo_dir.glob("*.json"))
        if not files:
            self.mount(Static("No todos found.", classes="todo-message"))
            return
            
        # Sort by modification time (descending)
        latest_file = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if not data:
                self.mount(Static("Empty todo list.", classes="todo-message"))
                return
                
            self.mount(Static(f"List: {latest_file.stem}", classes="todo-file-name"))
            
            for item in data:
                task = item.get("task", "Unknown task")
                # Fix: Handle potential string status or Enum
                status_str = item.get("status", "not-started")
                try:
                    status = TodoStatus(status_str)
                except ValueError:
                    status = TodoStatus.NOT_STARTED
                    
                self.mount(TodoListItem(task, status))
                
        except Exception as e:
            self.mount(Static(f"Error loading todos: {str(e)}", classes="todo-error"))
