"""Loading spinner widget for Kader CLI."""

from textual.reactive import reactive
from textual.widgets import Static


class LoadingSpinner(Static):
    """Animated loading spinner shown during LLM response generation."""

    DEFAULT_CSS = """
    LoadingSpinner {
        width: 100%;
        height: auto;
        padding: 1 2;
        color: $text-muted;
        text-style: italic;
    }

    LoadingSpinner.hidden {
        display: none;
    }
    """

    SPINNER_FRAMES = ["=   ", "==  ", "=== ", " ===", "  ==", "   =", "    "]

    frame_index: reactive[int] = reactive(0)
    is_spinning: reactive[bool] = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self.add_class("hidden")
        self._timer = None

    def on_mount(self) -> None:
        """Start the animation timer when mounted."""
        self._timer = self.set_interval(0.1, self._advance_frame)

    def _advance_frame(self) -> None:
        """Advance to the next spinner frame."""
        if self.is_spinning:
            self.frame_index = (self.frame_index + 1) % len(self.SPINNER_FRAMES)

    def watch_frame_index(self, frame_index: int) -> None:
        """Update display when frame changes."""
        if self.is_spinning:
            spinner = self.SPINNER_FRAMES[frame_index]
            self.update(f"{spinner} Kader is thinking...")

    def start(self) -> None:
        """Start the loading animation."""
        self.is_spinning = True
        self.remove_class("hidden")
        self.frame_index = 0

    def stop(self) -> None:
        """Stop the loading animation."""
        self.is_spinning = False
        self.add_class("hidden")
        self.update("")
