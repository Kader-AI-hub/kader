"""Conversation display widget for Kader CLI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Markdown, Static


class Message(Static):
    """A single message in the conversation."""

    def __init__(self, content: str, role: str = "user") -> None:
        super().__init__()
        self.content = content
        self.role = role
        self.add_class(f"message-{role}")

    def compose(self) -> ComposeResult:
        prefix = "(**) **You:**" if self.role == "user" else "(^^) **Kader:**"
        yield Markdown(f"{prefix}\n\n{self.content}")


class ConversationView(VerticalScroll):
    """Scrollable conversation history with markdown rendering."""

    DEFAULT_CSS = """
    ConversationView {
        padding: 1 2;
    }

    ConversationView Message {
        margin-bottom: 1;
        padding: 1;
    }

    ConversationView .message-user {
        background: $surface;
        border-left: thick $primary;
    }

    ConversationView .message-assistant {
        background: $surface-darken-1;
        border-left: thick $success;
    }
    """

    def add_message(self, content: str, role: str = "user") -> None:
        """Add a message to the conversation."""
        message = Message(content, role)
        self.mount(message)
        self.scroll_end(animate=True)

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        for child in self.query(Message):
            child.remove()
