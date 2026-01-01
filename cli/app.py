"""Kader CLI - Modern Vibe Coding CLI with Textual."""

import asyncio
from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    DirectoryTree,
    Footer,
    Header,
    Input,
    Markdown,
    Static,
)

from kader.providers import OllamaProvider, Message

from .utils import (
    HELP_TEXT,
    THEME_NAMES,
    DEFAULT_MODEL,
    get_models_text,
)
from .widgets import ConversationView, LoadingSpinner


WELCOME_MESSAGE = """# Welcome to Kader CLI! 🚀

Your **modern AI-powered coding assistant**.

Type a message below to start chatting, or use one of the commands:

- `/help` - Show available commands
- `/models` - View available LLM models  
- `/theme` - Change the color theme
- `/clear` - Clear the conversation
- `/exit` - Exit the application
"""


class KaderApp(App):
    """Main Kader CLI application."""

    TITLE = "Kader CLI"
    SUB_TITLE = "Modern Vibe Coding Assistant"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+t", "cycle_theme", "Theme"),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._current_theme_index = 0
        self._is_processing = False
        self._provider = OllamaProvider(DEFAULT_MODEL)
        self._conversation_history: list[Message] = []

    def compose(self) -> ComposeResult:
        """Create the application layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Sidebar with directory tree
            with Vertical(id="sidebar"):
                yield Static("📁 Files", id="sidebar-title")
                yield DirectoryTree(Path.cwd(), id="directory-tree")

            # Main content area
            with Vertical(id="content-area"):
                # Conversation view
                with Container(id="conversation"):
                    yield ConversationView(id="conversation-view")
                    yield LoadingSpinner()

                # Input area
                with Container(id="input-container"):
                    yield Input(
                        placeholder="Enter your prompt or /help for commands...",
                        id="prompt-input",
                    )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        # Show welcome message
        conversation = self.query_one("#conversation-view", ConversationView)
        conversation.mount(Markdown(WELCOME_MESSAGE, id="welcome"))
        
        # Focus the input
        self.query_one("#prompt-input", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        user_input = event.value.strip()
        if not user_input:
            return

        # Clear the input
        event.input.value = ""

        # Check if it's a command
        if user_input.startswith("/"):
            await self._handle_command(user_input)
        else:
            await self._handle_chat(user_input)

    async def _handle_command(self, command: str) -> None:
        """Handle CLI commands."""
        cmd = command.lower().strip()
        conversation = self.query_one("#conversation-view", ConversationView)

        if cmd == "/help":
            conversation.add_message(HELP_TEXT, "assistant")
        elif cmd == "/models":
            models_text = get_models_text()
            conversation.add_message(models_text, "assistant")
        elif cmd == "/theme":
            self._cycle_theme()
            theme_name = THEME_NAMES[self._current_theme_index]
            conversation.add_message(
                f"🎨 Theme changed to **{theme_name}**!", "assistant"
            )
        elif cmd == "/clear":
            conversation.clear_messages()
            self._conversation_history.clear()
            self.notify("Conversation cleared!", severity="information")
        elif cmd == "/new":
            conversation.clear_messages()
            self._conversation_history.clear()
            conversation.mount(Markdown(WELCOME_MESSAGE, id="welcome"))
            self.notify("New conversation started!", severity="information")
        elif cmd == "/exit":
            self.exit()
        else:
            conversation.add_message(
                f"❌ Unknown command: `{command}`\n\nType `/help` to see available commands.",
                "assistant",
            )

    async def _handle_chat(self, message: str) -> None:
        """Handle regular chat messages with OllamaProvider streaming."""
        if self._is_processing:
            self.notify("Please wait for the current response...", severity="warning")
            return

        self._is_processing = True
        conversation = self.query_one("#conversation-view", ConversationView)
        spinner = self.query_one(LoadingSpinner)

        # Add user message to UI and history
        conversation.add_message(message, "user")
        self._conversation_history.append(Message.user(message))

        # Show loading spinner
        spinner.start()

        try:
            # Stream response from Ollama using asyncio.to_thread
            full_response = ""
            
            def stream_response():
                nonlocal full_response
                for chunk in self._provider.stream(self._conversation_history):
                    full_response = chunk.content
                    
            await asyncio.to_thread(stream_response)
            
            # Add assistant response to history
            self._conversation_history.append(Message.assistant(full_response))
            
            # Hide spinner and show response
            spinner.stop()
            conversation.add_message(full_response, "assistant")

        except Exception as e:
            spinner.stop()
            error_msg = f"❌ **Error:** {str(e)}\n\nMake sure Ollama is running and the model `{DEFAULT_MODEL}` is available."
            conversation.add_message(error_msg, "assistant")
            self.notify(f"Error: {e}", severity="error")

        finally:
            self._is_processing = False

    def _cycle_theme(self) -> None:
        """Cycle through available themes."""
        # Remove current theme class if it's not dark
        current_theme = THEME_NAMES[self._current_theme_index]
        if current_theme != "dark":
            self.remove_class(f"theme-{current_theme}")

        # Move to next theme
        self._current_theme_index = (self._current_theme_index + 1) % len(THEME_NAMES)
        new_theme = THEME_NAMES[self._current_theme_index]

        # Apply new theme class (dark is default, no class needed)
        if new_theme != "dark":
            self.add_class(f"theme-{new_theme}")

    def action_clear(self) -> None:
        """Clear the conversation (Ctrl+L)."""
        conversation = self.query_one("#conversation-view", ConversationView)
        conversation.clear_messages()
        self._conversation_history.clear()
        self.notify("Conversation cleared!", severity="information")

    def action_cycle_theme(self) -> None:
        """Cycle theme (Ctrl+T)."""
        self._cycle_theme()
        theme_name = THEME_NAMES[self._current_theme_index]
        self.notify(f"Theme: {theme_name}", severity="information")


def main() -> None:
    """Run the Kader CLI application."""
    app = KaderApp()
    app.run()


if __name__ == "__main__":
    main()

