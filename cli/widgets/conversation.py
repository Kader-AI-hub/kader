"""Conversation display widget for Kader CLI."""

from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal
from textual.widgets import Markdown, Static


class Message(Static):
    """A single message in the conversation."""

    def __init__(
        self,
        content: str,
        role: str = "user",
        model_name: str | None = None,
        usage_cost: float | None = None,
    ) -> None:
        super().__init__()
        self.content = content
        self.role = role
        self.model_name = model_name
        self.usage_cost = usage_cost
        self.add_class(f"message-{role}")

    def compose(self) -> ComposeResult:
        prefix = "(**) **You:**" if self.role == "user" else "(^^) **Kader:**"
        yield Markdown(f"{prefix}\n\n{self.content}")

        if self.role == "assistant" and (self.model_name or self.usage_cost is not None):
            with Horizontal(classes="message-footer"):
                model_label = f"[*] {self.model_name}" if self.model_name else ""
                yield Static(model_label, classes="footer-left")

                usage_label = (
                    f"($) {self.usage_cost:.6f}" if self.usage_cost is not None else ""
                )
                yield Static(usage_label, classes="footer-right")


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

    .message-footer {
        height: auto;
        margin-top: 0;
        padding: 0 1;
        border-top: none;
    }

    .footer-left {
        color: $secondary;
        text-style: italic;
        width: 1fr;
    }

    .footer-right {
        color: $success;
        text-style: bold;
        text-align: right;
        width: auto;
    }
    """

    def add_message(
        self,
        content: str,
        role: str = "user",
        model_name: str | None = None,
        usage_cost: float | None = None,
    ) -> None:
        """Add a message to the conversation."""
        message = Message(content, role, model_name, usage_cost)
        self.mount(message)
        self.scroll_end(animate=True)

    def clear_messages(self) -> None:
        """Clear all messages from the conversation."""
        for child in self.query(Message):
            child.remove()
