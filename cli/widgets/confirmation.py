"""Inline selection widget for tool confirmation and model selection."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
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

    def __init__(self, message: str, options: list[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.message = message
        self.options = options or ["(+) Yes", "(-) No"]

    def compose(self) -> ComposeResult:
        from textual.containers import Horizontal

        yield Static(f">_ {self.message}", classes="message-text")
        yield Static(
            "↑↓ to select • Enter to confirm • Y/N for quick select",
            classes="prompt-text",
        )
        with Horizontal(classes="selector-container"):
            for i, option in enumerate(self.options):
                cls = (
                    "option selected"
                    if i == self.selected_index
                    else "option not-selected"
                )
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


class ModelSelector(Widget, can_focus=True):
    """
    Model selector widget for choosing LLM models.

    Uses arrow keys to navigate, Enter to confirm, Escape to cancel.
    """

    BINDINGS = [
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("enter", "confirm", "Confirm", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    DEFAULT_CSS = """
    ModelSelector {
        width: 100%;
        height: auto;
        padding: 1;
        border: solid $primary;
        background: $surface;
    }

    ModelSelector:focus {
        border: double $primary;
    }

    ModelSelector .title-text {
        margin-bottom: 1;
        text-align: center;
        width: 100%;
        color: $warning;
        text-style: bold;
    }

    ModelSelector .prompt-text {
        margin-bottom: 1;
        text-align: center;
        width: 100%;
        color: $text-muted;
    }

    ModelSelector .model-list {
        width: 100%;
        height: auto;
        max-height: 15;
    }

    ModelSelector .model-option {
        padding: 0 2;
        width: 100%;
    }

    ModelSelector .model-option.selected {
        background: $primary;
        color: $text;
        text-style: bold;
    }

    ModelSelector .model-option.not-selected {
        background: $surface;
        color: $text-muted;
    }

    ModelSelector .model-option.current {
        color: $success;
    }
    """

    selected_index: reactive[int] = reactive(0)

    def __init__(self, models: list[str], current_model: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.models = models
        self.current_model = current_model
        # Start with current model selected if it exists
        if current_model in models:
            self.selected_index = models.index(current_model)

    def compose(self) -> ComposeResult:
        yield Static("(^^) Select Model", classes="title-text")
        yield Static(
            "↑↓ to navigate • Enter to select • Esc to cancel", classes="prompt-text"
        )
        with Vertical(classes="model-list"):
            for i, model in enumerate(self.models):
                is_current = model == self.current_model
                is_selected = i == self.selected_index
                classes = "model-option"
                if is_selected:
                    classes += " selected"
                else:
                    classes += " not-selected"
                if is_current:
                    classes += " current"
                label = f"  >> {model}" if is_selected else f"    {model}"
                if is_current:
                    label += " (current)"
                yield Static(label, classes=classes, id=f"model-{i}")

    def on_mount(self) -> None:
        """Focus self when mounted."""
        self.focus()

    def watch_selected_index(self, old_index: int, new_index: int) -> None:
        """Update visual selection when index changes."""
        try:
            # Update old option
            old_option = self.query_one(f"#model-{old_index}", Static)
            old_option.remove_class("selected")
            old_option.add_class("not-selected")
            old_model = self.models[old_index]
            old_label = f"    {old_model}"
            if old_model == self.current_model:
                old_label += " (current)"
            old_option.update(old_label)

            # Update new option
            new_option = self.query_one(f"#model-{new_index}", Static)
            new_option.remove_class("not-selected")
            new_option.add_class("selected")
            new_model = self.models[new_index]
            new_label = f"  >> {new_model}"
            if new_model == self.current_model:
                new_label += " (current)"
            new_option.update(new_label)
        except Exception:
            pass

    def action_move_up(self) -> None:
        """Move selection up."""
        self.selected_index = (self.selected_index - 1) % len(self.models)

    def action_move_down(self) -> None:
        """Move selection down."""
        self.selected_index = (self.selected_index + 1) % len(self.models)

    def action_confirm(self) -> None:
        """Confirm current selection."""
        selected_model = self.models[self.selected_index]
        self.post_message(self.ModelSelected(selected_model))

    def action_cancel(self) -> None:
        """Cancel selection."""
        self.post_message(self.ModelCancelled())

    class ModelSelected(TextualMessage):
        """Message sent when user selects a model."""

        def __init__(self, model: str) -> None:
            super().__init__()
            self.model = model

    class ModelCancelled(TextualMessage):
        """Message sent when user cancels selection."""

        pass
