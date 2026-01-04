"""Inline selection widget for tool confirmation."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message as TextualMessage
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class InlineSelector(Widget, can_focus=True):
    """
    Inline selector widget for Yes/No confirmation.
    
    Uses arrow keys to navigate, Enter to confirm.
    """
    
    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("left", "move_up", "Left", show=False),
        Binding("right", "move_down", "Right", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("y", "confirm_yes", "Yes", show=False),
        Binding("n", "confirm_no", "No", show=False),
    ]
    
    DEFAULT_CSS = """
    InlineSelector {
        width: 100%;
        height: auto;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }
    
    InlineSelector:focus {
        border: double $primary;
    }
    
    InlineSelector .selector-container {
        width: 100%;
        height: auto;
        align: center middle;
    }
    
    InlineSelector .option {
        padding: 0 3;
        margin: 0 2;
        min-width: 12;
        text-align: center;
    }
    
    InlineSelector .option.selected {
        background: $primary;
        color: $text;
        text-style: bold reverse;
    }
    
    InlineSelector .option.not-selected {
        background: $surface-darken-1;
        color: $text-muted;
    }
    
    InlineSelector .prompt-text {
        margin-bottom: 1;
        text-align: center;
        width: 100%;
        color: $text-muted;
    }
    
    InlineSelector .message-text {
        margin-bottom: 1;
        text-align: center;
        width: 100%;
        color: $warning;
        text-style: bold;
    }
    """
    
    selected_index: reactive[int] = reactive(0)
    
    def __init__(
        self, 
        message: str,
        options: list[str] = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.message = message
        self.options = options or ["✅ Yes", "❌ No"]
    
    def compose(self) -> ComposeResult:
        yield Static(f"🔧 {self.message}", classes="message-text")
        yield Static("↑↓ to select • Enter to confirm • Y/N for quick select", classes="prompt-text")
        with Horizontal(classes="selector-container"):
            for i, option in enumerate(self.options):
                cls = "option selected" if i == self.selected_index else "option not-selected"
                yield Static(option, classes=cls, id=f"option-{i}")
    
    def on_mount(self) -> None:
        """Focus self when mounted."""
        self.focus()
    
    def watch_selected_index(self, old_index: int, new_index: int) -> None:
        """Update visual selection when index changes."""
        try:
            old_option = self.query_one(f"#option-{old_index}", Static)
            old_option.remove_class("selected")
            old_option.add_class("not-selected")
            
            new_option = self.query_one(f"#option-{new_index}", Static)
            new_option.remove_class("not-selected")
            new_option.add_class("selected")
        except Exception:
            pass
    
    def action_move_up(self) -> None:
        """Move selection up/left."""
        self.selected_index = (self.selected_index - 1) % len(self.options)
    
    def action_move_down(self) -> None:
        """Move selection down/right."""
        self.selected_index = (self.selected_index + 1) % len(self.options)
    
    def action_confirm(self) -> None:
        """Confirm current selection."""
        self.post_message(self.Confirmed(self.selected_index == 0))
    
    def action_confirm_yes(self) -> None:
        """Quick confirm Yes."""
        self.selected_index = 0
        self.post_message(self.Confirmed(True))
    
    def action_confirm_no(self) -> None:
        """Quick confirm No."""
        self.selected_index = 1
        self.post_message(self.Confirmed(False))
    
    class Confirmed(TextualMessage):
        """Message sent when user confirms selection."""
        def __init__(self, confirmed: bool) -> None:
            super().__init__()
            self.confirmed = confirmed
