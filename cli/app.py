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

from kader.agent.agents import ReActAgent
from kader.tools import get_default_registry
from kader.memory import SlidingWindowConversationManager, FileSessionManager, MemoryConfig

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
- `/save` - Save current session
- `/load` - Load a saved session
- `/sessions` - List saved sessions
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
        Binding("ctrl+s", "save_session", "Save"),
        Binding("ctrl+r", "refresh_tree", "Refresh"),
        Binding("tab", "focus_next", "Next", show=False),
        Binding("shift+tab", "focus_previous", "Previous", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._current_theme_index = 0
        self._is_processing = False
        self._current_model = DEFAULT_MODEL
        self._current_session_id: str | None = None
        # Session manager with sessions stored in ~/.kader/sessions/
        self._session_manager = FileSessionManager(
            MemoryConfig(memory_dir=Path.home() / ".kader")
        )
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
            use_persistence=True
        )


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
            self._agent.memory.clear()
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
                    "assistant"
                )
            else:
                self._handle_load_session(parts[1], conversation)
        elif cmd == "/refresh":
            self._refresh_directory_tree()
            self.notify("Directory tree refreshed!", severity="information")
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

        try:
            # Invoke agent using asyncio.to_thread for sync invoke
            def invoke_agent():
                return self._agent.invoke(message)
                    
            response = await asyncio.to_thread(invoke_agent)
            
            # Hide spinner and show response
            spinner.stop()
            conversation.add_message(response.content, "assistant")

        except Exception as e:
            spinner.stop()
            error_msg = f"❌ **Error:** {str(e)}\n\nMake sure Ollama is running and the model `{self._current_model}` is available."
            conversation.add_message(error_msg, "assistant")
            self.notify(f"Error: {e}", severity="error")

        finally:
            self._is_processing = False
            # Auto-refresh directory tree in case agent created/modified files
            self._refresh_directory_tree()

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
        self._agent.memory.clear()
        self.notify("Conversation cleared!", severity="information")

    def action_cycle_theme(self) -> None:
        """Cycle theme (Ctrl+T)."""
        self._cycle_theme()
        theme_name = THEME_NAMES[self._current_theme_index]
        self.notify(f"Theme: {theme_name}", severity="information")

    def action_save_session(self) -> None:
        """Save session (Ctrl+S)."""
        conversation = self.query_one("#conversation-view", ConversationView)
        self._handle_save_session(conversation)

    def action_refresh_tree(self) -> None:
        """Refresh directory tree (Ctrl+R)."""
        self._refresh_directory_tree()
        self.notify("Directory tree refreshed!", severity="information")

    def _refresh_directory_tree(self) -> None:
        """Refresh the directory tree to show new/modified files."""
        try:
            tree = self.query_one("#directory-tree", DirectoryTree)
            tree.reload()
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
                f"✅ Session saved!\n\n**Session ID:** `{self._current_session_id}`",
                "assistant"
            )
            self.notify("Session saved!", severity="information")
        except Exception as e:
            conversation.add_message(f"❌ Error saving session: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")

    def _handle_load_session(self, session_id: str, conversation: ConversationView) -> None:
        """Load a saved session by ID."""
        try:
            # Check if session exists
            session = self._session_manager.get_session(session_id)
            if not session:
                conversation.add_message(
                    f"❌ Session `{session_id}` not found.\n\nUse `/sessions` to see available sessions.",
                    "assistant"
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
                f"✅ Session `{session_id}` loaded with {len(messages)} messages.",
                "assistant"
            )
            self.notify("Session loaded!", severity="information")
        except Exception as e:
            conversation.add_message(f"❌ Error loading session: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")

    def _handle_list_sessions(self, conversation: ConversationView) -> None:
        """List all saved sessions."""
        try:
            sessions = self._session_manager.list_sessions()
            
            if not sessions:
                conversation.add_message(
                    "📭 No saved sessions found.\n\nUse `/save` to save the current session.",
                    "assistant"
                )
                return
            
            lines = ["## Saved Sessions 📂\n", "| Session ID | Created | Updated |", "|------------|---------|---------|"]
            for session in sessions:
                # Shorten UUID for display
                short_id = session.session_id[:8] + "..."
                created = session.created_at[:10]  # Just date
                updated = session.updated_at[:10]
                lines.append(f"| `{session.session_id}` | {created} | {updated} |")
            
            lines.append(f"\n*Use `/load <session_id>` to load a session.*")
            conversation.add_message("\n".join(lines), "assistant")
        except Exception as e:
            conversation.add_message(f"❌ Error listing sessions: {e}", "assistant")
            self.notify(f"Error: {e}", severity="error")


def main() -> None:
    """Run the Kader CLI application."""
    app = KaderApp()
    app.run()


if __name__ == "__main__":
    main()

