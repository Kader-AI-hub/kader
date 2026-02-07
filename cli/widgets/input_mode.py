"""Mode-aware input widget for Kader CLI."""

from enum import Enum

from textual.app import ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Input, Static


class InputMode(Enum):
    """Input modes for the CLI."""

    PROMPT = "prompt"
    COMMAND = "command"
    SYSTEM = "system"


class ModeAwareInput(Container):
    """
    Input widget with mode detection and visual feedback.

    Modes:
    - PROMPT (cyan): Default mode for regular chat
    - COMMAND (purple): When input starts with /
    - SYSTEM (orange): When input starts with !
    """

    DEFAULT_CSS = """
    ModeAwareInput {
        height: auto;
        min-height: 5;
        max-height: 8;
        background: #0d0d14;
        padding: 0 1;
    }

    ModeAwareInput .mode-header {
        height: 1;
        layout: horizontal;
        padding: 0 1;
    }

    ModeAwareInput .mode-hint {
        width: 1fr;
        color: #6c7086;
        text-style: italic;
    }

    ModeAwareInput .mode-label {
        width: auto;
        text-style: bold;
    }

    /* Mode-specific label colors */
    ModeAwareInput .mode-label.prompt {
        color: #22d3ee;
        background: rgba(34, 211, 238, 0.1);
    }

    ModeAwareInput .mode-label.command {
        color: #a855f7;
        background: rgba(168, 85, 247, 0.1);
    }

    ModeAwareInput .mode-label.system {
        color: #fb923c;
        background: rgba(251, 146, 60, 0.1);
    }

    ModeAwareInput .input-wrapper {
        height: auto;
        min-height: 3;
        margin-bottom: 1;
    }

    ModeAwareInput .input-prefix {
        width: 3;
        height: 3;
        content-align: center middle;
        text-style: bold;
    }

    ModeAwareInput .input-prefix.prompt {
        color: #22d3ee;
    }

    ModeAwareInput .input-prefix.command {
        color: #a855f7;
    }

    ModeAwareInput .input-prefix.system {
        color: #fb923c;
    }

    ModeAwareInput .input-badge {
        dock: right;
        width: auto;
        padding: 0 1;
        text-style: bold;
    }

    ModeAwareInput .input-badge.prompt {
        color: #000000;
        background: #22d3ee;
    }

    ModeAwareInput .input-badge.command {
        color: #ffffff;
        background: #a855f7;
    }

    ModeAwareInput .input-badge.system {
        color: #000000;
        background: #fb923c;
    }

    ModeAwareInput #mode-input {
        background: transparent;
        height: 3;
        width: 1fr;
    }

    ModeAwareInput #mode-input.prompt {
        border: round #22d3ee;
    }

    ModeAwareInput #mode-input.command {
        border: round #a855f7;
    }

    ModeAwareInput #mode-input.system {
        border: round #fb923c;
    }

    ModeAwareInput #mode-input:focus.prompt {
        border: round #22d3ee;
    }

    ModeAwareInput #mode-input:focus.command {
        border: round #a855f7;
    }

    ModeAwareInput #mode-input:focus.system {
        border: round #fb923c;
    }
    """

    HINTS = {
        InputMode.PROMPT: 'Try: "How do I optimize this SQL query?"',
        InputMode.COMMAND: "Kader Command Mode",
        InputMode.SYSTEM: "Direct shell execution enabled",
    }

    LABELS = {
        InputMode.PROMPT: "PROMPT MODE",
        InputMode.COMMAND: "COMMAND MODE",
        InputMode.SYSTEM: "SYSTEM MODE",
    }

    BADGES = {
        InputMode.PROMPT: "PROMPT",
        InputMode.COMMAND: "COMMAND",
        InputMode.SYSTEM: "SYSTEM",
    }

    PREFIXES = {
        InputMode.PROMPT: ">",
        InputMode.COMMAND: "/",
        InputMode.SYSTEM: "[>",
    }

    PLACEHOLDERS = {
        InputMode.PROMPT: "Ask Kader anything... (use / for commands, ! for system)",
        InputMode.COMMAND: "Enter command...",
        InputMode.SYSTEM: "Enter system command...",
    }

    current_mode: reactive[InputMode] = reactive(InputMode.PROMPT)

    def __init__(self, id: str = None) -> None:
        super().__init__(id=id)
        self._input: Input | None = None

    def compose(self) -> ComposeResult:
        with Container(classes="mode-header"):
            yield Static(
                self.HINTS[InputMode.PROMPT], id="mode-hint", classes="mode-hint"
            )
            yield Static(
                self.LABELS[InputMode.PROMPT],
                id="mode-label",
                classes="mode-label prompt",
            )

        with Container(classes="input-wrapper"):
            yield Static(
                self.BADGES[InputMode.PROMPT],
                id="input-badge",
                classes="input-badge prompt",
            )
            self._input = Input(
                placeholder=self.PLACEHOLDERS[InputMode.PROMPT],
                id="mode-input",
                classes="prompt",
            )
            yield self._input

    def on_mount(self) -> None:
        """Focus the input when mounted."""
        if self._input:
            self._input.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Detect mode based on input content."""
        value = event.value
        if value.startswith("/"):
            new_mode = InputMode.COMMAND
        elif value.startswith("!"):
            new_mode = InputMode.SYSTEM
        else:
            new_mode = InputMode.PROMPT

        if new_mode != self.current_mode:
            self.current_mode = new_mode

    def watch_current_mode(self, old_mode: InputMode, new_mode: InputMode) -> None:
        """Update visuals when mode changes."""
        # Update hint
        try:
            hint = self.query_one("#mode-hint", Static)
            hint.update(self.HINTS[new_mode])
        except Exception:
            pass

        # Update label
        try:
            label = self.query_one("#mode-label", Static)
            label.remove_class(old_mode.value)
            label.add_class(new_mode.value)
            label.update(self.LABELS[new_mode])
        except Exception:
            pass

        # Update badge
        try:
            badge = self.query_one("#input-badge", Static)
            badge.remove_class(old_mode.value)
            badge.add_class(new_mode.value)
            badge.update(self.BADGES[new_mode])
        except Exception:
            pass

        # Update input styling
        if self._input:
            self._input.remove_class(old_mode.value)
            self._input.add_class(new_mode.value)
            self._input.placeholder = self.PLACEHOLDERS[new_mode]

    @property
    def value(self) -> str:
        """Get the current input value."""
        return self._input.value if self._input else ""

    @value.setter
    def value(self, new_value: str) -> None:
        """Set the input value."""
        if self._input:
            self._input.value = new_value

    @property
    def disabled(self) -> bool:
        """Check if input is disabled."""
        return self._input.disabled if self._input else False

    @disabled.setter
    def disabled(self, value: bool) -> None:
        """Set input disabled state."""
        if self._input:
            self._input.disabled = value

    def focus(self) -> None:
        """Focus the input."""
        if self._input:
            self._input.focus()
