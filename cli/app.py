"""Kader CLI - Modern AI Coding CLI with Rich.

A Rich-based interactive CLI for the Kader AI coding agent. Uses Rich for
beautiful terminal output and prompt_toolkit for async input handling.
"""

import asyncio
import atexit
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.theme import Theme

from kader.memory import FileSessionManager, MemoryConfig
from kader.workflows import PlannerExecutorWorkflow

from .commands import InitializeCommand
from .llm_factory import LLMProviderFactory
from .utils import DEFAULT_MODEL, HELP_TEXT

# Suppress warnings
warnings.filterwarnings("ignore")

# Custom Rich theme matching the Kader brand
KADER_THEME = Theme(
    {
        "kader.primary": "bold magenta",
        "kader.cyan": "bold cyan",
        "kader.yellow": "bold yellow",
        "kader.green": "bold green",
        "kader.red": "bold red",
        "kader.orange": "bold dark_orange",
        "kader.muted": "dim",
        "kader.user": "bold magenta",
        "kader.assistant": "bold cyan",
        "kader.success": "bold green",
        "kader.error": "bold red",
        "kader.info": "bold yellow",
    }
)

WELCOME_BANNER = """\
[bold magenta]
  _  __    _    ____  _____ ____
 | |/ /   / \\  |  _ \\| ____|  _ \\
 | ' /   / _ \\ | | | |  _| | |_) |
 | . \\  / ___ \\| |_| | |___|  _ <
 |_|\\_\\/_/   \\_\\____/|_____|_| \\_\\
[/bold magenta]"""


class KaderApp:
    """Main Kader CLI application using Rich."""

    def __init__(self) -> None:
        self.console = Console(theme=KADER_THEME)
        self._is_processing = False
        self._current_model = DEFAULT_MODEL
        self._current_session_id: str | None = None
        self._running = True

        # Session manager
        self._session_manager = FileSessionManager(
            MemoryConfig(memory_dir=Path.home() / ".kader")
        )

        # Tool confirmation coordination
        self._confirmation_event: Optional[threading.Event] = None
        self._confirmation_result: tuple[bool, Optional[str]] = (True, None)
        self._update_info: Optional[str] = None
        self._awaiting_rejection_context: bool = False

        # Spinner state for Live display
        self._spinner_live: Optional[Live] = None

        # Dedicated thread pool for agent invocation
        self._agent_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="kader_agent"
        )
        atexit.register(self._agent_executor.shutdown, wait=False)

        self._workflow = self._create_workflow(self._current_model)

        # Prompt session for input
        self._prompt_session: PromptSession = PromptSession()

    def _create_workflow(self, model_name: str) -> PlannerExecutorWorkflow:
        """Create a new PlannerExecutorWorkflow with the specified model."""
        provider = LLMProviderFactory.create_provider(model_name)

        workflow = PlannerExecutorWorkflow(
            name="kader_cli",
            provider=provider,
            model_name=model_name,
            interrupt_before_tool=True,
            tool_confirmation_callback=self._tool_confirmation_callback,
            direct_execution_callback=self._direct_execution_callback,
            tool_execution_result_callback=self._tool_execution_result_callback,
            use_persistence=True,
            executor_names=["executor"],
        )

        if not self._current_session_id:
            self._current_session_id = workflow.session_id

        return workflow

    # ── Callbacks (called from agent thread) ───────────────────────────

    def _direct_execution_callback(self, message: str, tool_name: str) -> None:
        """Callback for direct execution tools - called from agent thread."""
        self._stop_spinner()
        self.console.print(f"  [kader.cyan]⚡ {tool_name}:[/kader.cyan] {message}")
        self._start_spinner()

    def _tool_execution_result_callback(
        self, tool_name: str, success: bool, result: str
    ) -> None:
        """Callback for tool execution results - called from agent thread."""
        self._stop_spinner()
        if success:
            self.console.print(
                rf"  [kader.green]\[+] {tool_name}[/kader.green] completed successfully"
            )
        else:
            error_preview = result[:100] + "..." if len(result) > 100 else result
            self.console.print(
                rf"  [kader.red]\[-] {tool_name}[/kader.red] failed: {error_preview}"
            )
        self._start_spinner()

    def _tool_confirmation_callback(self, message: str) -> tuple[bool, Optional[str]]:
        """Callback for tool confirmation - called from agent thread."""
        self._stop_spinner()
        self.console.print()
        self.console.print(
            Panel(
                message,
                title=r"[kader.yellow]\[?] Tool Confirmation[/kader.yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        # Set up synchronization
        self._confirmation_event = threading.Event()
        self._confirmation_result = (True, None)

        # Prompt for confirmation in the main thread context
        # We use an event to wait for the answer from the input loop
        self._awaiting_confirmation = True

        # Wait for user response (blocking in agent thread)
        if not self._confirmation_event.wait(timeout=300):
            return (False, "Tool confirmation timed out after 5 minutes")

        self._start_spinner()
        return self._confirmation_result

    # ── Spinner helpers ───────────────────────────────────────────────

    def _start_spinner(self) -> None:
        """Start the thinking spinner."""
        if self._spinner_live is None:
            spinner = Spinner("dots", text="[kader.yellow] Kader is thinking...[/]")
            self._spinner_live = Live(
                spinner, console=self.console, transient=True, refresh_per_second=10
            )
            self._spinner_live.start()

    def _stop_spinner(self) -> None:
        """Stop the thinking spinner."""
        if self._spinner_live is not None:
            self._spinner_live.stop()
            self._spinner_live = None

    # ── Display helpers ──────────────────────────────────────────────

    def _print_user_message(self, message: str) -> None:
        """Display a user message."""
        self.console.print()
        self.console.print(
            Panel(
                Markdown(message),
                title=r"[kader.user]\[>] You[/kader.user]",
                border_style="magenta",
                padding=(0, 1),
            )
        )

    def _print_assistant_message(
        self,
        message: str,
        model_name: str | None = None,
        usage_cost: float | None = None,
    ) -> None:
        """Display an assistant message."""
        subtitle_parts = []
        if model_name:
            subtitle_parts.append(f"[dim]{model_name}[/dim]")
        if usage_cost is not None:
            subtitle_parts.append(f"[kader.green]${usage_cost:.6f}[/kader.green]")
        subtitle = " · ".join(subtitle_parts) if subtitle_parts else None

        self.console.print()
        self.console.print(
            Panel(
                Markdown(message),
                title=r"[kader.assistant]\[=] Kader[/kader.assistant]",
                subtitle=subtitle,
                border_style="cyan",
                padding=(0, 1),
            )
        )

    def _print_system_message(self, message: str, style: str = "kader.info") -> None:
        """Display a system/info message."""
        self.console.print(rf"  [{style}]\[>][/{style}] {message}")

    # ── Command handlers ─────────────────────────────────────────────

    async def _handle_command(self, command: str) -> None:
        """Handle CLI commands."""
        cmd = command.lower().strip()

        if cmd == "/help":
            self.console.print()
            self.console.print(Markdown(HELP_TEXT))

        elif cmd == "/models":
            await self._handle_models()

        elif cmd == "/clear":
            self._workflow.planner.memory.clear()
            self._workflow.planner.provider.reset_tracking()
            self._current_session_id = self._workflow.session_id
            self.console.clear()
            self._print_welcome()
            self._print_system_message("Conversation cleared!", "kader.green")

        elif cmd == "/save":
            self._handle_save_session()

        elif cmd == "/sessions":
            self._handle_list_sessions()

        elif cmd == "/skills":
            self._handle_skills()

        elif cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.console.print(
                    r"  [kader.red]\[-][/kader.red] Usage: `/load <session_id>` "
                    "— Use `/sessions` to see available sessions."
                )
            else:
                self._handle_load_session(parts[1])

        elif cmd == "/cost":
            self._handle_cost()

        elif cmd == "/init":
            init_cmd = InitializeCommand(self)
            await init_cmd.execute()

        elif cmd == "/exit":
            self._running = False
            self.console.print("  [kader.muted]Goodbye! [!][/kader.muted]")

        else:
            self.console.print(
                rf"  [kader.red]\[-][/kader.red] Unknown command: `{command}` "
                "— Type `/help` to see available commands."
            )

    async def _handle_models(self) -> None:
        """Handle the /models command with interactive selection."""
        try:
            models = LLMProviderFactory.get_flat_model_list()
            if not models:
                self.console.print(
                    r"  [kader.red]\[-][/kader.red] No models found. "
                    "Check provider configurations."
                )
                return

            # Display models as a numbered list
            table = Table(
                title="[kader.cyan]Available Models[/kader.cyan]",
                border_style="cyan",
                show_header=True,
                header_style="bold cyan",
                padding=(0, 1),
            )
            table.add_column("#", style="dim", width=4, justify="right")
            table.add_column("Model", style="white")
            table.add_column("Status", justify="center")

            for i, model in enumerate(models, 1):
                marker = (
                    "[kader.green]● current[/kader.green]"
                    if model == self._current_model
                    else "[dim]available[/dim]"
                )
                table.add_row(str(i), model, marker)

            self.console.print()
            self.console.print(table)
            self.console.print(
                "  [dim]Enter model number to switch, or press Enter to cancel:[/dim]"
            )

            # Get selection
            try:
                choice = await self._prompt_session.prompt_async(
                    HTML("<ansicyan>  model> </ansicyan>")
                )
                choice = choice.strip()
                if not choice:
                    self._print_system_message(
                        f"Model selection cancelled. Current: `{self._current_model}`"
                    )
                    return

                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    old_model = self._current_model
                    self._current_model = models[idx]
                    self._workflow = self._create_workflow(self._current_model)
                    self._print_system_message(
                        f"[kader.green]Model changed from "
                        f"`{old_model}` to `{self._current_model}`[/kader.green]"
                    )
                else:
                    self.console.print("  [kader.red]✗[/kader.red] Invalid selection.")
            except (ValueError, EOFError, KeyboardInterrupt):
                self._print_system_message(
                    f"Model selection cancelled. Current: `{self._current_model}`"
                )

        except Exception as e:
            self.console.print(f"  [kader.red]✗[/kader.red] Error fetching models: {e}")

    async def _handle_chat(self, message: str) -> None:
        """Handle regular chat messages with PlannerExecutorWorkflow."""
        if self._is_processing:
            self.console.print(
                r"  [kader.yellow]\[!][/kader.yellow] "
                "Please wait for the current response..."
            )
            return

        self._is_processing = True
        self._print_user_message(message)

        try:
            # Start spinner
            self._start_spinner()

            # Run the workflow in the dedicated thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._agent_executor, lambda: self._workflow.run(message)
            )

            # Stop spinner and show response
            self._stop_spinner()
            if response:
                self._print_assistant_message(
                    response,
                    model_name=self._workflow.planner.provider.model,
                    usage_cost=self._workflow.planner.provider.total_cost.total_cost,
                )

        except Exception as e:
            self._stop_spinner()
            self.console.print(
                f"\n  [kader.red]\\[-] Error:[/kader.red] {e}\n"
                f"  Make sure the provider for `{self._current_model}` "
                "is configured and available."
            )

        finally:
            self._is_processing = False

    async def _handle_terminal_command(self, command: str) -> None:
        """Handle terminal commands starting with !."""
        cmd = command.strip()
        if not cmd:
            return

        self.console.print(f"\n  [kader.orange]\\[>] Executing:[/kader.orange] `{cmd}`")

        try:
            process = await asyncio.create_subprocess_shell(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            output = ""
            if stdout:
                output += stdout.decode().strip()
            if stderr:
                if output:
                    output += "\n\n"
                output += f"Stderr:\n{stderr.decode().strip()}"

            if not output:
                output = "Command executed successfully with no output."

            self.console.print()
            self.console.print(
                Panel(
                    output,
                    title="[kader.orange]Terminal Output[/kader.orange]",
                    border_style="dark_orange",
                    padding=(0, 1),
                )
            )

        except Exception as e:
            self.console.print(
                rf"  [kader.red]\[-][/kader.red] Error executing command: {e}"
            )

    def _handle_save_session(self) -> None:
        """Save the current session."""
        import shutil

        from kader.memory.types import get_default_memory_dir

        try:
            if not self._current_session_id:
                session = self._session_manager.create_session("kader_cli")
                self._current_session_id = session.session_id

            planner_session_dir = (
                get_default_memory_dir() / "sessions" / self._current_session_id
            )
            target_session_dir = (
                self._session_manager.sessions_dir / self._current_session_id
            )

            if planner_session_dir.exists() and self._current_session_id:
                target_session_dir.parent.mkdir(parents=True, exist_ok=True)
                if target_session_dir.exists():
                    shutil.rmtree(target_session_dir)
                shutil.copytree(planner_session_dir, target_session_dir)

            self._print_system_message(
                f"[kader.green]Session saved! "
                f"ID: `{self._current_session_id}`[/kader.green]"
            )
        except Exception as e:
            self.console.print(f"  [kader.red]✗[/kader.red] Error saving session: {e}")

    def _handle_load_session(self, session_id: str) -> None:
        """Load a saved session by ID."""
        try:
            session = self._session_manager.get_session(session_id)
            if not session:
                self.console.print(
                    rf"  [kader.red]\[-][/kader.red] Session `{session_id}` not found. "
                    "Use `/sessions` to see available sessions."
                )
                return

            messages = self._session_manager.load_conversation(session_id)
            self._workflow.planner.memory.clear()

            for msg in messages:
                self._workflow.planner.memory.add_message(msg)
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ["user", "assistant"] and content:
                    if role == "user":
                        self._print_user_message(content)
                    else:
                        self._print_assistant_message(content)

            self._current_session_id = session_id
            self._print_system_message(
                f"[kader.green]Session `{session_id}` loaded "
                f"with {len(messages)} messages.[/kader.green]"
            )
        except Exception as e:
            self.console.print(f"  [kader.red]✗[/kader.red] Error loading session: {e}")

    def _handle_list_sessions(self) -> None:
        """List all saved sessions."""
        try:
            sessions = self._session_manager.list_sessions()

            if not sessions:
                self.console.print(
                    "  [dim]No saved sessions found. "
                    "Use `/save` to save the current session.[/dim]"
                )
                return

            table = Table(
                title="[kader.cyan]Saved Sessions[/kader.cyan]",
                border_style="cyan",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Session ID", style="white")
            table.add_column("Created", style="dim")
            table.add_column("Updated", style="dim")

            for session in sessions:
                table.add_row(
                    session.session_id,
                    session.created_at[:10],
                    session.updated_at[:10],
                )

            self.console.print()
            self.console.print(table)
            self.console.print(
                "  [dim]Use `/load <session_id>` to load a session.[/dim]"
            )
        except Exception as e:
            self.console.print(
                rf"  [kader.red]\[-][/kader.red] Error listing sessions: {e}"
            )

    def _handle_skills(self) -> None:
        """Handle the /skills command to display loaded skills."""
        try:
            from kader.tools.skills import SkillLoader

            loader = SkillLoader()
            skills = loader.list_skills()

            if not skills:
                self.console.print(
                    "  [dim]No skills found in `~/.kader/skills` "
                    "or `./.kader/skills`[/dim]"
                )
            else:
                table = Table(
                    title="[kader.cyan]Loaded Skills[/kader.cyan]",
                    border_style="cyan",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("Name", style="bold white")
                table.add_column("Description", style="dim")

                for s in skills:
                    table.add_row(s.name, s.description)

                self.console.print()
                self.console.print(table)
        except Exception as e:
            self.console.print(f"  [kader.red]✗[/kader.red] Error loading skills: {e}")

    def _handle_cost(self) -> None:
        """Display LLM usage costs."""
        try:
            cost = self._workflow.planner.provider.total_cost
            usage = self._workflow.planner.provider.total_usage
            model = self._workflow.planner.provider.model

            table = Table(
                title=f"[kader.cyan]Usage Costs — {model}[/kader.cyan]",
                border_style="cyan",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Metric", style="white")
            table.add_column("Value", justify="right", style="bold")

            table.add_row("Input Cost", f"${cost.input_cost:.6f}")
            table.add_row("Output Cost", f"${cost.output_cost:.6f}")
            table.add_row(
                "[bold]Total Cost[/bold]",
                f"[kader.green]${cost.total_cost:.6f}[/kader.green]",
            )
            table.add_row("", "")
            table.add_row("Prompt Tokens", f"{usage.prompt_tokens:,}")
            table.add_row("Completion Tokens", f"{usage.completion_tokens:,}")
            table.add_row(
                "[bold]Total Tokens[/bold]",
                f"[bold]{usage.total_tokens:,}[/bold]",
            )

            self.console.print()
            self.console.print(table)

            if cost.total_cost == 0.0:
                self.console.print("  [dim]ℹ Ollama runs locally — no API costs.[/dim]")
        except Exception as e:
            self.console.print(f"  [kader.red]✗[/kader.red] Error getting costs: {e}")

    # ── Update check ─────────────────────────────────────────────────

    def _check_for_updates(self) -> None:
        """Check for package updates in background thread."""
        try:
            from outdated import check_outdated

            current_version = get_version("kader")
            is_outdated, latest_version = check_outdated("kader", current_version)

            if is_outdated:
                self._update_info = latest_version
        except Exception:
            pass

    def _show_update_notification(self) -> None:
        """Show update notification if available."""
        if not self._update_info:
            return
        try:
            current = get_version("kader")
            self.console.print(
                rf"  [kader.yellow]\[^] Update available! "
                f"v{current} -> v{self._update_info} — "
                f"Run: uv tool upgrade kader[/kader.yellow]"
            )
        except Exception:
            pass

    # ── Welcome banner ───────────────────────────────────────────────

    def _print_welcome(self) -> None:
        """Print the welcome banner."""
        try:
            version = get_version("kader")
        except Exception:
            version = "?"

        self.console.print(WELCOME_BANNER)
        self.console.print(
            f"  [dim]v{version} · "
            f"Model: {self._current_model} · "
            f"Type /help for commands[/dim]\n"
        )

    # ── Main loop ────────────────────────────────────────────────────

    def _get_prompt_text(self) -> HTML:
        """Get the styled prompt text."""
        return HTML("<ansimagenta><b>[>] </b></ansimagenta>")

    def _get_confirmation_prompt(self) -> HTML:
        """Get the confirmation prompt text."""
        return HTML("<ansiyellow><b>  Approve? [Y/n/reason]: </b></ansiyellow>")

    async def _run_async(self) -> None:
        """Async main loop."""
        self._print_welcome()

        # Background update check
        threading.Thread(target=self._check_for_updates, daemon=True).start()

        # Show update notification after a short delay
        await asyncio.sleep(2)
        self._show_update_notification()

        while self._running:
            try:
                # Check if we need to handle a tool confirmation
                if getattr(self, "_awaiting_confirmation", False):
                    self._awaiting_confirmation = False
                    with patch_stdout():
                        answer = await self._prompt_session.prompt_async(
                            self._get_confirmation_prompt()
                        )
                    answer = answer.strip().lower()

                    if answer in ("", "y", "yes"):
                        self._confirmation_result = (True, None)
                        self._print_system_message(
                            "[kader.green][+] Approved — executing tool...[/kader.green]"
                        )
                        if self._confirmation_event:
                            self._confirmation_event.set()
                    elif answer in ("n", "no"):
                        self.console.print(
                            "  [kader.red][-] Rejected.[/kader.red] "
                            "Please provide context for the rejection:"
                        )
                        with patch_stdout():
                            context = await self._prompt_session.prompt_async(
                                HTML("<ansired><b>  reason> </b></ansired>")
                            )
                        self._confirmation_result = (False, context.strip() or None)
                        if self._confirmation_event:
                            self._confirmation_event.set()
                    else:
                        # Treat as rejection with the text as the reason
                        self._confirmation_result = (False, answer)
                        if self._confirmation_event:
                            self._confirmation_event.set()
                    continue

                # Normal input
                with patch_stdout():
                    user_input = await self._prompt_session.prompt_async(
                        self._get_prompt_text()
                    )

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Check if awaiting rejection context
                if self._awaiting_rejection_context:
                    self._awaiting_rejection_context = False
                    self._confirmation_result = (False, user_input)
                    if self._confirmation_event:
                        self._confirmation_event.set()
                    continue

                # Route input
                if user_input.startswith("/"):
                    await self._handle_command(user_input)
                elif user_input.startswith("!"):
                    await self._handle_terminal_command(user_input[1:])
                else:
                    await self._handle_chat(user_input)

            except KeyboardInterrupt:
                self.console.print(
                    "\n  [dim]Press Ctrl+C again to exit, or type /exit[/dim]"
                )
                try:
                    with patch_stdout():
                        await self._prompt_session.prompt_async(self._get_prompt_text())
                except (KeyboardInterrupt, EOFError):
                    self._running = False
                    self.console.print("  [kader.muted]Goodbye! 👋[/kader.muted]")

            except EOFError:
                self._running = False
                self.console.print("  [kader.muted]Goodbye! 👋[/kader.muted]")

    def run(self) -> None:
        """Run the Kader CLI application."""
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self.console.print("\n  [kader.muted]Goodbye! 👋[/kader.muted]")
        finally:
            self._stop_spinner()


def main() -> None:
    """Run the Kader CLI application."""
    app = KaderApp()
    app.run()


if __name__ == "__main__":
    main()
