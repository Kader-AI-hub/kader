"""Kader CLI - Modern Vibe Coding CLI with Textual."""

import asyncio
import threading
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Footer,
    Header,
    Input,
    Markdown,
    Static,
    Tree,
)

from kader.agent.agents import ReActAgent
from kader.memory import (
    FileSessionManager,
    MemoryConfig,
    SlidingWindowConversationManager,
)
from kader.tools import get_default_registry

from .utils import (
    DEFAULT_MODEL,
    HELP_TEXT,
)
from .widgets import ConversationView, InlineSelector, LoadingSpinner, ModelSelector

WELCOME_MESSAGE = """
<div align="center">

```
    ██╗ ██╗  ██╗ █████╗ ██████╗ ███████╗██████╗
   ██╔╝ ██║ ██╔╝██╔══██╗██╔══██╗██╔════╝██╔══██╗
  ██╔╝  █████╔╝ ███████║██║  ██║█████╗  ██████╔╝
 ██╔╝   ██╔═██╗ ██╔══██║██║  ██║██╔══╝  ██╔══██╗
██╔╝    ██║  ██╗██║  ██║██████╔╝███████╗██║  ██║
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
```

</div>

Type a message below to start chatting, or use one of the commands:

- `/help` - Show available commands
- `/models` - View available LLM models
- `/clear` - Clear the conversation
- `/save` - Save current session
- `/load` - Load a saved session
- `/sessions` - List saved sessions
- `/cost` - Show the cost of the conversation
- `/exit` - Exit the application
"""


# Minimum terminal size to prevent UI breakage
MIN_WIDTH = 89
MIN_HEIGHT = 29


class ASCIITree(Tree):
    """A Tree widget that uses no icons."""

    ICON_NODE = ""
    ICON_NODE_EXPANDED = ""


class KaderApp(App):
    """Main Kader CLI application."""

    TITLE = "Kader CLI"
    SUB_TITLE = f"v{get_version('kader')}"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+r", "refresh_tree", "Refresh"),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._is_processing = False
        self._current_model = DEFAULT_MODEL
        self._current_session_id: str | None = None
        # Session manager with sessions stored in ~/.kader/sessions/
        self._session_manager = FileSessionManager(
            MemoryConfig(memory_dir=Path.home() / ".kader")
        )
        # Tool confirmation coordination
        self._confirmation_event: Optional[threading.Event] = None
        self._confirmation_result: tuple[bool, Optional[str]] = (True, None)
        self._inline_selector: Optional[InlineSelector] = None
        self._model_selector: Optional[ModelSelector] = None
        self._update_info: Optional[str] = None  # Latest version if update available

        self._agent = self._create_agent(self._current_model)

    def _create_agent(self, model_name: str) -> ReActAgent:
        """Create a new ReActAgent with the specified model."""
        registry = get_default_registry()
        memory = SlidingWindowConversationManager(window_size=10)
        return ReActAgent(
            name="kader_cli",
            tools=registry,
            memory=memory,
            model_name=model_name,
            use_persistence=True,
            interrupt_before_tool=True,
            tool_confirmation_callback=self._tool_confirmation_callback,
        )

    def _tool_confirmation_callback(self, message: str) -> tuple[bool, Optional[str]]:
        """
        Callback for tool confirmation - called from agent thread.

        Shows inline selector with arrow key navigation.
        """
        # Set up synchronization
        self._confirmation_event = threading.Event()
        self._confirmation_result = (True, None)  # Default

        # Schedule selector to be shown on main thread
        # Use call_from_thread to safely call from background thread
        self.call_from_thread(self._show_inline_selector, message)

        # Wait for user response (blocking in agent thread)
        # This is safe because we're in a background thread
        self._confirmation_event.wait()

        # Return the result
        return self._confirmation_result

    def _show_inline_selector(self, message: str) -> None:
        """Show the inline selector in the conversation view."""
        # Stop spinner while waiting for confirmation
        try:
            spinner = self.query_one(LoadingSpinner)
            spinner.stop()
        except Exception:
            pass

        conversation = self.query_one("#conversation-view", ConversationView)

        # Create and mount the selector
        self._inline_selector = InlineSelector(message, id="tool-selector")
        conversation.mount(self._inline_selector)
        conversation.scroll_end()

        # Disable input and focus selector
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = True

        # Force focus on the selector widget
        self.set_focus(self._inline_selector)

        # Force refresh
        self.refresh()

    def on_inline_selector_confirmed(self, event: InlineSelector.Confirmed) -> None:
        """Handle confirmation from inline selector."""
        conversation = self.query_one("#conversation-view", ConversationView)

        # Set result
        self._confirmation_result = (event.confirmed, None)

        # Remove selector and show result message
        tool_message = None
        if self._inline_selector:
            tool_message = self._inline_selector.message
            self._inline_selector.remove()
            self._inline_selector = None

        if event.confirmed:
            if tool_message:
                conversation.add_message(tool_message, "assistant")
            conversation.add_message("(+) Executing tool...", "assistant")
            # Restart spinner
            try:
                spinner = self.query_one(LoadingSpinner)
                spinner.start()
            except Exception:
                pass
        else:
            conversation.add_message("(-) Tool execution skipped.", "assistant")

        # Re-enable input
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = False

        # Signal the waiting thread BEFORE focusing input
        # This ensures the agent thread can continue
        if self._confirmation_event:
            self._confirmation_event.set()

        # Now focus input
        prompt_input.focus()

    async def _show_model_selector(self, conversation: ConversationView) -> None:
        """Show the model selector widget."""
        from kader.providers import OllamaProvider

        try:
            models = OllamaProvider.get_supported_models()
            if not models:
                conversation.add_message(
                    "## Models (^^)\n\n*No models found. Is Ollama running?*",
                    "assistant",
                )
                return

            # Create and mount the model selector
            self._model_selector = ModelSelector(
                models=models, current_model=self._current_model, id="model-selector"
            )
            conversation.mount(self._model_selector)
            conversation.scroll_end()

            # Disable input and focus selector
            prompt_input = self.query_one("#prompt-input", Input)
            prompt_input.disabled = True
            self.set_focus(self._model_selector)

        except Exception as e:
            conversation.add_message(
                f"## Models (^^)\n\n*Error fetching models: {e}*", "assistant"
            )

    def on_model_selector_model_selected(
        self, event: ModelSelector.ModelSelected
    ) -> None:
        """Handle model selection."""
        conversation = self.query_one("#conversation-view", ConversationView)

        # Remove selector
        if self._model_selector:
            self._model_selector.remove()
            self._model_selector = None

        # Update model and recreate agent
        old_model = self._current_model
        self._current_model = event.model
        self._agent = self._create_agent(self._current_model)

        conversation.add_message(
            f"(+) Model changed from `{old_model}` to `{self._current_model}`",
            "assistant",
        )

        # Re-enable input
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = False
        prompt_input.focus()

    def on_model_selector_model_cancelled(
        self, event: ModelSelector.ModelCancelled
    ) -> None:
        """Handle model selection cancelled."""
        conversation = self.query_one("#conversation-view", ConversationView)

        # Remove selector
        if self._model_selector:
            self._model_selector.remove()
            self._model_selector = None

        conversation.add_message(
            f"Model selection cancelled. Current model: `{self._current_model}`",
            "assistant",
        )

        # Re-enable input
        prompt_input = self.query_one("#prompt-input", Input)
        prompt_input.disabled = False
        prompt_input.focus()

    def compose(self) -> ComposeResult:
        """Create the application layout."""
        yield Header()

        with Horizontal(id="main-container"):
            # Sidebar with directory tree
            with Vertical(id="sidebar"):
                yield Static("Files", id="sidebar-title")
                yield ASCIITree(str(Path.cwd().name), id="directory-tree")

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

        # Check initial size
        self._check_terminal_size()

        # Start background update check
        # Start background update check
        threading.Thread(target=self._check_for_updates, daemon=True).start()

        # Initial tree population
        self._refresh_directory_tree()

    def _populate_tree(self, node, path: Path) -> None:
        """Recursively populate the tree with ASCII symbols."""
        try:
            # Sort: directories first, then files
            items = sorted(
                path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
            )
            for child in items:
                if child.name.startswith((".", "__pycache__")):
                    continue

                if child.is_dir():
                    new_node = node.add(f"[+] {child.name}", expand=False)
                    self._populate_tree(new_node, child)
                else:
                    node.add(f"{child.name}")
        except Exception:
            pass

    def _check_for_updates(self) -> None:
        """Check for package updates in background thread."""
        try:
            from outdated import check_outdated

            current_version = get_version("kader")
            is_outdated, latest_version = check_outdated("kader", current_version)

            if is_outdated:
                self._update_info = latest_version
                # Schedule UI update on main thread
                self.call_from_thread(self._show_update_notification)
        except Exception:
            # Silently ignore update check failures
            pass

    def _show_update_notification(self) -> None:
        """Show update notification as a toast."""
        if not self._update_info:
            return

        try:
            current = get_version("kader")
            message = (
                f">> Update available! v{current} → v{self._update_info} "
                f"Run: uv tool upgrade kader"
            )
            self.notify(message, severity="information", timeout=10)
        except Exception:
            pass

    def on_resize(self) -> None:
        """Handle terminal resize events."""
        self._check_terminal_size()

    def _check_terminal_size(self) -> None:
        """Check if terminal is large enough and show warning if not."""
        width = self.console.size.width
        height = self.console.size.height

        # Check if we need to show/hide the size warning
        too_small = width < MIN_WIDTH or height < MIN_HEIGHT

        try:
            warning = self.query_one("#size-warning", Static)
            if not too_small:
                warning.remove()
        except Exception:
            if too_small:
                # Show warning overlay
                warning_text = f"""<!>  Terminal Too Small

Current: {width}x{height}
Minimum: {MIN_WIDTH}x{MIN_HEIGHT}

Please resize your terminal."""
                warning = Static(warning_text, id="size-warning")
                self.mount(warning)

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
            await self._show_model_selector(conversation)
        elif cmd == "/clear":
            conversation.clear_messages()
            self._agent.memory.clear()
            self._agent.provider.reset_tracking()  # Reset usage/cost tracking
            self._current_session_id = None
            self.notify("Conversation cleared!", severity="information")
        elif cmd == "/save":
            self._handle_save_session(conversation)
        elif cmd == "/sessions":
            self._handle_list_sessions(conversation)
        elif cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                conversation.add_message(
                    "❌ Usage: `/load <session_id>`\n\nUse `/sessions` to see available sessions.",
                    "assistant",
                )
            else:
                self._handle_load_session(parts[1], conversation)
        elif cmd == "/refresh":
            self._refresh_directory_tree()
            self.notify("Directory tree refreshed!", severity="information")
        elif cmd == "/cost":
            self._handle_cost(conversation)
        elif cmd == "/exit":
            self.exit()
        else:
            conversation.add_message(
                f"❌ Unknown command: `{command}`\n\nType `/help` to see available commands.",
                "assistant",
            )

    async def _handle_chat(self, message: str) -> None:
        """Handle regular chat messages with ReActAgent."""
        if self._is_processing:
            self.notify("Please wait for the current response...", severity="warning")
            return

        self._is_processing = True
        conversation = self.query_one("#conversation-view", ConversationView)
        spinner = self.query_one(LoadingSpinner)

        # Add user message to UI
        conversation.add_message(message, "user")

        # Show loading spinner
        spinner.start()

        # Use run_worker to run agent in background without blocking event loop
        self.run_worker(
            self._invoke_agent_worker(message),
            name="agent_worker",
            exclusive=True,
        )

    async def _invoke_agent_worker(self, message: str) -> None:
        """Worker to invoke agent in background."""
        conversation = self.query_one("#conversation-view", ConversationView)
        spinner = self.query_one(LoadingSpinner)

        try:
            # Run the agent invoke in a thread
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: self._agent.invoke(message)
            )

            # Hide spinner and show response (this runs on main thread via await)
            spinner.stop()
            if response and response.content:
                conversation.add_message(response.content, "assistant")

        except Exception as e:
            spinner.stop()
            error_msg = f"(-) **Error:** {str(e)}\n\nMake sure Ollama is running and the model `{self._current_model}` is available."
            conversation.add_message(error_msg, "assistant")
            self.notify(f"Error: {e}", severity="error")

        finally:
            self._is_processing = False
            # Auto-refresh directory tree in case agent created/modified files
            self._refresh_directory_tree()

    def action_clear(self) -> None:
        """Clear the conversation (Ctrl+L)."""
        conversation = self.query_one("#conversation-view", ConversationView)
        conversation.clear_messages()
        self._agent.memory.clear()
        self.notify("Conversation cleared!", severity="information")

    def action_save_session(self) -> None:
        """Save session (Ctrl+S)."""
        conversation = self.query_one("#conversation-view", ConversationView)
        self._handle_save_session(conversation)

    def action_refresh_tree(self) -> None:
        """Refresh directory tree (Ctrl+R)."""
        self._refresh_directory_tree()
        self.notify("Directory tree refreshed!", severity="information")

    def _refresh_directory_tree(self) -> None:
        """Refresh the directory tree with ASCII symbols."""
        try:
            tree = self.query_one("#directory-tree", ASCIITree)
            tree.clear()
            tree.root.label = str(Path.cwd().name)
            self._populate_tree(tree.root, Path.cwd())
            tree.root.expand()
        except Exception:
            pass  # Silently ignore if tree not found

    def _handle_save_session(self, conversation: ConversationView) -> None:
        """Save the current session."""
        try:
            # Create a new session if none exists
            if not self._current_session_id:
                session = self._session_manager.create_session("kader_cli")
                self._current_session_id = session.session_id

            # Get messages from agent memory and save
            messages = [msg.message for msg in self._agent.memory.get_messages()]
            self._session_manager.save_conversation(self._current_session_id, messages)

            conversation.add_message(
                f"(+) Session saved!\n\n**Session ID:** `{self._current_session_id}`",
                "assistant",
            )
            self.notify("Session saved!", severity="information")
        except Exception as e:
            conversation.add_message(f"(-) Error saving session: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")

    def _handle_load_session(
        self, session_id: str, conversation: ConversationView
    ) -> None:
        """Load a saved session by ID."""
        try:
            # Check if session exists
            session = self._session_manager.get_session(session_id)
            if not session:
                conversation.add_message(
                    f"(-) Session `{session_id}` not found.\n\nUse `/sessions` to see available sessions.",
                    "assistant",
                )
                return

            # Load conversation history
            messages = self._session_manager.load_conversation(session_id)

            # Clear current state
            conversation.clear_messages()
            self._agent.memory.clear()

            # Add loaded messages to memory and UI
            for msg in messages:
                self._agent.memory.add_message(msg)
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    conversation.add_message(content, role)

            self._current_session_id = session_id
            conversation.add_message(
                f"(+) Session `{session_id}` loaded with {len(messages)} messages.",
                "assistant",
            )
            self.notify("Session loaded!", severity="information")
        except Exception as e:
            conversation.add_message(f"(-) Error loading session: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")

    def _handle_list_sessions(self, conversation: ConversationView) -> None:
        """List all saved sessions."""
        try:
            sessions = self._session_manager.list_sessions()

            if not sessions:
                conversation.add_message(
                    "[ ] No saved sessions found.\n\nUse `/save` to save the current session.",
                    "assistant",
                )
                return

            lines = [
                "## Saved Sessions [=]\n",
                "| Session ID | Created | Updated |",
                "|------------|---------|---------|",
            ]
            for session in sessions:
                # Shorten UUID for display
                created = session.created_at[:10]  # Just date
                updated = session.updated_at[:10]
                lines.append(f"| `{session.session_id}` | {created} | {updated} |")

            lines.append("\n*Use `/load <session_id>` to load a session.*")
            conversation.add_message("\n".join(lines), "assistant")
        except Exception as e:
            conversation.add_message(f"❌ Error listing sessions: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")

    def _handle_cost(self, conversation: ConversationView) -> None:
        """Display LLM usage costs."""
        try:
            # Get cost and usage from the provider
            cost = self._agent.provider.total_cost
            usage = self._agent.provider.total_usage
            model = self._agent.provider.model

            lines = [
                "## Usage Costs ($)\n",
                f"**Model:** `{model}`\n",
                "### Cost Breakdown",
                "| Type | Amount |",
                "|------|--------|",
                f"| Input Cost | ${cost.input_cost:.6f} |",
                f"| Output Cost | ${cost.output_cost:.6f} |",
                f"| **Total Cost** | **${cost.total_cost:.6f}** |",
                "",
                "### Token Usage",
                "| Type | Tokens |",
                "|------|--------|",
                f"| Prompt Tokens | {usage.prompt_tokens:,} |",
                f"| Completion Tokens | {usage.completion_tokens:,} |",
                f"| **Total Tokens** | **{usage.total_tokens:,}** |",
            ]

            if cost.total_cost == 0.0:
                lines.append(
                    "\n> (!) *Note: Ollama runs locally, so there are no API costs.*"
                )

            conversation.add_message("\n".join(lines), "assistant")
        except Exception as e:
            conversation.add_message(f"(-) Error getting costs: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")


def main() -> None:
    """Run the Kader CLI application."""
    app = KaderApp()
    app.run()


if __name__ == "__main__":
    main()
