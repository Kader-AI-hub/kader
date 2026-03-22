"""Kader CLI - Modern AI Coding CLI with Rich.

A Rich-based interactive CLI for the Kader AI coding agent. Uses Rich for
beautiful terminal output and prompt_toolkit for async input handling.
"""

import asyncio
import json
import os
import subprocess
import sys
import warnings
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Optional

from loguru import logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.theme import Theme
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_fixed,
)

from kader.memory import AsyncFileSessionManager, MemoryConfig
from kader.utils import agenerate_session_title
from kader.utils.todo_metadata import TodoMetadataHandler
from kader.workflows import PlannerExecutorWorkflow

from .commands import InitializeCommand
from .llm_factory import LLMProviderFactory
from .utils import COMMAND_NAMES, DEFAULT_MODEL, HELP_TEXT

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


def enter_fullscreen() -> None:
    """Clear screen and hide cursor for fullscreen CLI mode."""
    sys.stdout.write("\033[2J\033[H\033[?25l")
    sys.stdout.flush()


def exit_fullscreen() -> None:
    """Restore terminal: clear screen and show cursor."""
    sys.stdout.write("\033[2J\033[H\033[?25h")
    sys.stdout.flush()


def format_plan_display(items: list[dict]) -> None:
    """Format and print todo items as a readable Plan with colored status icons."""
    status_styles = {
        "completed": ("[x]", "green"),
        "in-progress": ("[-]", "yellow"),
        "not-started": ("[ ]", "dim white"),
    }
    console = Console(theme=KADER_THEME)
    for item in items:
        status = item.get("status", "not-started")
        icon, style = status_styles.get(status, ("[ ]", "dim white"))
        task = item.get("task", "")
        console.print(f"  [{style}]{icon}[/{style}] {task}")


class KaderApp:
    """Main Kader CLI application using Rich."""

    def __init__(self) -> None:
        self.console = Console(theme=KADER_THEME)
        self._is_processing = False
        self._current_model = DEFAULT_MODEL
        self._current_session_id: str | None = None
        self._running = True
        self._session_title: Optional[str] = None

        # Session manager
        self._session_manager = AsyncFileSessionManager(
            MemoryConfig(memory_dir=Path.home() / ".kader")
        )

        # Update check state
        self._update_info: Optional[str] = None

        # Spinner state for Live display
        self._spinner_live: Optional[Live] = None

        self._workflow = self._create_workflow(self._current_model)

        # Prompt session for input with autocomplete
        command_completer = WordCompleter(
            COMMAND_NAMES,
            ignore_case=True,
            sentence=True,
        )
        self._prompt_session: PromptSession = PromptSession(
            completer=command_completer,
            complete_while_typing=True,
        )

    def _create_workflow(self, model_name: str) -> PlannerExecutorWorkflow:
        """Create a new PlannerExecutorWorkflow with the specified model."""
        from kader.tools.exec_commands import CommandExecutorTool

        CommandExecutorTool.set_output_callback(self._command_output_callback)
        CommandExecutorTool.set_input_callback(self._command_input_callback)

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

    def _direct_execution_callback(
        self, message: str, tool_name: str, tool_args: dict | None = None
    ) -> None:
        """Callback for direct execution tools - called from agent thread."""
        self._stop_spinner()
        self.console.print(f"  [kader.cyan]⚡ {tool_name}:[/kader.cyan] {message}")

        if tool_args:
            self._display_tool_content(tool_name, tool_args)

        self._start_spinner()

    def _display_tool_content(self, tool_name: str, tool_args: dict) -> None:
        """Display content for write_file and edit_file tools with toggle capability."""
        if tool_name == "write_file" and "content" in tool_args:
            content = tool_args["content"]
            path = tool_args.get("path", "write_file")
            self._show_content_nonblocking(content, path, "write_file")

        elif (
            tool_name == "edit_file"
            and "old_string" in tool_args
            and "new_string" in tool_args
        ):
            old_string = tool_args["old_string"]
            new_string = tool_args["new_string"]
            path = tool_args.get("path", "edit_file")
            self._show_edit_diff_nonblocking(old_string, new_string, path)

    def _show_content_nonblocking(
        self, content: str, path: str, title: str, max_lines: int = 20
    ) -> None:
        """Show content without blocking - partial view by default."""
        lines = content.split("\n")

        if len(lines) <= max_lines:
            display_content = content
        else:
            display_content = "\n".join(lines[:max_lines])

        from rich.syntax import Syntax

        lexer_name = Syntax.guess_lexer(path, code=display_content)
        display_syntax = Syntax(
            display_content, lexer_name, theme="monokai", word_wrap=True
        )

        self.console.print()
        self.console.print(
            Panel(
                display_syntax,
                title=f"[kader.orange]{title}: {path}[/kader.orange]",
                border_style="dark_orange",
                padding=(0, 1),
            )
        )
        self.console.print(
            f"  [dim]Showing {max_lines} lines | Total: {len(lines)} lines[/dim]"
        )

    def _show_edit_diff_nonblocking(
        self, old_string: str, new_string: str, path: str, max_lines: int = 15
    ) -> None:
        """Show edit_file changes with side-by-side view - no blocking."""
        old_lines = old_string.split("\n")
        new_lines = new_string.split("\n")

        old_display = "\n".join(old_lines[:max_lines])
        new_display = "\n".join(new_lines[:max_lines])

        if len(old_lines) > max_lines:
            old_display += f"\n... ({len(old_lines)} lines total)"
        if len(new_lines) > max_lines:
            new_display += f"\n... ({len(new_lines)} lines total)"

        from rich.syntax import Syntax

        lexer_name = Syntax.guess_lexer(path, code=new_display)
        old_syntax = Syntax(old_display, lexer_name, theme="monokai", word_wrap=True)
        new_syntax = Syntax(new_display, lexer_name, theme="monokai", word_wrap=True)

        old_panel = Panel(
            old_syntax,
            title=f"[kader.red]OLD: {path}[/kader.red]",
            border_style="red",
            padding=(0, 1),
        )
        new_panel = Panel(
            new_syntax,
            title=f"[kader.green]NEW: {path}[/kader.green]",
            border_style="green",
            padding=(0, 1),
        )

        self.console.print()
        self.console.print(old_panel)
        self.console.print(new_panel)
        self.console.print(
            f"  [dim]Showing {max_lines} lines | Old: {len(old_lines)} lines, New: {len(new_lines)} lines[/dim]"
        )

    def _tool_execution_result_callback(
        self, tool_name: str, success: bool, result: str, tool_args: dict | None = None
    ) -> None:
        """Callback for tool execution results - called from agent thread."""
        self._stop_spinner()
        if success:
            self.console.print(
                rf"  [kader.green]\[+] {tool_name}[/kader.green] completed successfully"
            )
            if tool_name == "todo_tool" and result:
                try:
                    items = json.loads(result)
                    if items:
                        self.console.print("")
                        self.console.print("[bold] Plan:[/bold]")
                        format_plan_display(items)
                except (json.JSONDecodeError, TypeError):
                    pass
        else:
            error_preview = result[:100] + "..." if len(result) > 100 else result
            self.console.print(
                rf"  [kader.red]\[-] {tool_name}[/kader.red] failed: {error_preview}"
            )
        self._start_spinner()

    def _command_output_callback(self, output: str) -> None:
        """Callback for streaming command output - called from agent thread."""
        self.console.print(output, end="")

    def _command_input_callback(self) -> str | None:
        """Callback for getting user input during command execution."""
        self._stop_spinner()
        self.console.print()
        self.console.print(
            r"[kader.yellow]\[?] Command needs input:[/kader.yellow] ",
            end="",
        )
        try:
            user_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            user_input = None
        self._start_spinner()
        return user_input

    def _tool_confirmation_callback(self, message: str) -> tuple[bool, Optional[str]]:
        """Callback for tool confirmation - called from agent thread.

        Prompts the user directly via synchronous input since this runs
        in the agent thread while the main loop is blocked on chat input.
        """
        from prompt_toolkit.shortcuts import choice
        from prompt_toolkit.styles import Style

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

        custom_style = Style.from_dict(
            {
                "selected": "reverse",
                "radio-selected": "reverse",
                "radio-checked": "reverse",
                "selected-option": "reverse",
                "pointer": "reverse",
            }
        )

        try:
            answer = choice(
                message="  Approve?",
                options=[
                    ("y", "Yes - Execute tool"),
                    ("n", "No - Reject and provide reason..."),
                ],
                style=custom_style,
            )
        except (Exception, KeyboardInterrupt):
            answer = "n"

        if answer == "y":
            self._print_system_message(
                "[kader.green][+] Approved -- executing tool...[/kader.green]"
            )
            self._start_spinner()
            return (True, None)
        else:
            self.console.print(
                "  [kader.red][-] Rejected.[/kader.red] "
                "Please provide context for the rejection:"
            )
            self.console.print("  [kader.red]reason> [/kader.red] ", end="")
            try:
                context = input().strip()
            except (EOFError, KeyboardInterrupt):
                context = ""
            self._start_spinner()
            return (False, context or None)

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
        session_title: str | None = None,
    ) -> None:
        """Display an assistant message."""
        subtitle_parts = []
        if session_title:
            subtitle_parts.append(f"[yellow]{session_title}[/yellow]")
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

    def _get_command_suggestions(self, cmd: str) -> list[str]:
        """Get list of commands that start with the given input."""
        if not cmd.startswith("/"):
            cmd = "/" + cmd
        return [c for c in COMMAND_NAMES if c.startswith(cmd.lower())]

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
            self._session_title = None
            self.console.clear()
            self._print_welcome()
            self._print_system_message("Conversation cleared!", "kader.green")

        elif cmd == "/save":
            await self._handle_save_session()

        elif cmd == "/sessions":
            await self._handle_list_sessions()

        elif cmd == "/skills":
            self._handle_skills()

        elif cmd == "/commands":
            await self._handle_commands()

        elif cmd.startswith("/load"):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.console.print(
                    r"  [kader.red]\[-][/kader.red] Usage: `/load <session_id>` "
                    "— Use `/sessions` to see available sessions."
                )
            else:
                await self._handle_load_session(parts[1])

        elif cmd == "/cost":
            self._handle_cost()

        elif cmd == "/init":
            init_cmd = InitializeCommand(self)
            await init_cmd.execute()

        elif cmd == "/exit":
            self._running = False
            self.console.print("  [kader.muted]Goodbye! [!][/kader.muted]")

        else:
            special_cmd = self._get_special_command(command)
            if special_cmd:
                command_name, user_task = special_cmd
                await self._handle_special_command(command_name, user_task, command)
                return

            partial = cmd.lstrip("/")
            suggestions = self._get_command_suggestions(partial)

            # Filter: if exactly 1 suggestion and it's a sub-command (contains / after the command name),
            # don't auto-complete - show all options instead to avoid infinite loops
            # We check if suggestion has pattern like /command/subcommand (2+ slashes)
            if len(suggestions) == 1:
                # Count slashes - if more than 1, it's a sub-command
                slash_count = suggestions[0].count("/")
                if slash_count >= 2:
                    # Don't auto-complete to sub-commands directly
                    suggestions = []

            if len(suggestions) == 1:
                self.console.print(
                    rf"  [kader.cyan]→[/kader.cyan] Auto-completing to `{suggestions[0]}`"
                )
                await self._handle_command(suggestions[0])
            elif len(suggestions) > 1:
                table = Table(
                    title="[kader.cyan]Available Commands[/kader.cyan]",
                    border_style="cyan",
                    show_header=True,
                    header_style="bold cyan",
                    padding=(0, 1),
                )
                table.add_column("Command", style="bold white")
                table.add_column("Description", style="dim")

                from .utils import COMMANDS, SPECIAL_COMMANDS

                all_cmds = COMMANDS + SPECIAL_COMMANDS
                cmd_map = {c.name: c.description for c in all_cmds}

                for sugg in suggestions:
                    table.add_row(sugg, cmd_map.get(sugg, ""))

                self.console.print()
                self.console.print(table)
                self.console.print(
                    rf"  [kader.yellow]\[!][/kader.yellow] Did you mean: `{command}`? "
                    "Type the full command to execute."
                )
            else:
                self.console.print(
                    rf"  [kader.red]\[-][/kader.red] Unknown command: `{command}` "
                    "— Type `/help` to see available commands, or `/commands` for special commands."
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

    async def _send_continuation(self, max_attempts: int = 3) -> str:
        """Send continuation messages with tenacity retry until plan is complete.

        Uses tenacity for structured retry with wait between attempts.
        Stops when the plan is complete or max_attempts is reached.

        Args:
            max_attempts: Maximum number of continuation attempts.

        Returns:
            The latest workflow response string.
        """
        response = ""
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(max_attempts),
            wait=wait_fixed(3),
            reraise=True,
        ):
            with attempt:
                self._stop_spinner()
                self.console.print(
                    r"  [kader.yellow]\→[/kader.yellow] "
                    "Plan not complete. Sending continuation..."
                )
                self._start_spinner()
                response = await self._workflow.arun("continue")
                self._stop_spinner()

                is_complete = await self._should_send_continuation()
                if is_complete:
                    return response
                # Not complete — raise to trigger next retry attempt
                raise RuntimeError("Plan not yet complete")

        return response

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
            is_first_message = len(self._workflow.planner.memory.get_messages()) == 0

            if is_first_message and self._session_title is None:
                asyncio.create_task(self._generate_session_title(message))
                self.console.print("  [dim]Session: Generating...[/dim]")
            elif self._session_title:
                self.console.print(f"  [dim]Session: {self._session_title}[/dim]")

            # Start spinner
            self._start_spinner()

            # Run the workflow asynchronously
            response = await self._workflow.arun(message)
            self._stop_spinner()

            # Check completeness and last message upfront
            is_complete = await self._should_send_continuation()
            last_msg = self._get_last_assistant_message()

            # ── Handle response with todo-based continuation logic ──
            if is_complete:
                if not last_msg:
                    # Response empty but plan completed — single retry
                    self.console.print(
                        r"  [kader.red]\→[/kader.red] "
                        "[kader.red]Sending continuation...[/kader.red]"
                    )
                    self._start_spinner()
                    await asyncio.sleep(3)
                    response = await self._workflow.arun("continue")
                    self._stop_spinner()
            else:
                # Plan not complete — use tenacity-based continuation
                try:
                    response = await self._send_continuation(max_attempts=3)
                except (RetryError, RuntimeError):
                    logger.warning("Plan continuation exhausted after max retries")

            self._print_assistant_message(
                response,
                model_name=self._workflow.planner.provider.model,
                usage_cost=self._workflow.planner.provider.total_cost.total_cost,
                session_title=self._session_title,
            )

        except ConnectionError as e:
            self._stop_spinner()
            logger.error(f"Connection error during chat: {e}")
            self.console.print(
                f"\n  [kader.red]\\[-] Connection Error:[/kader.red] {e}\n"
                f"  Check your network and provider for `{self._current_model}`."
            )

        except RetryError as e:
            self._stop_spinner()
            logger.error(f"Retry exhausted during chat: {e}")
            self.console.print(
                f"\n  [kader.red]\\[-] Retry Exhausted:[/kader.red] {e}\n"
                f"  The provider for `{self._current_model}` may be overloaded."
            )

        except Exception as e:
            self._stop_spinner()
            logger.error(f"Unexpected error during chat: {e}")
            self.console.print(
                f"\n  [kader.red]\\[-] Error:[/kader.red] {e}\n"
                f"  Make sure the provider for `{self._current_model}` "
                "is configured and available."
            )

        finally:
            self._is_processing = False

    async def _should_send_continuation(self) -> bool:
        """
        Check if we should send a continuation message based on todo metadata.

        Returns:
            True if the current todo is complete and we should continue,
            False if the todo is not complete and we should stop.
        """
        if not self._workflow.session_id:
            return True

        todo_tool = self._workflow.planner.tools_map.get("todo_tool")
        if not todo_tool:
            return True

        current_todo_id = getattr(todo_tool, "current_todo_id", None)
        if not current_todo_id:
            return True

        metadata_handler = TodoMetadataHandler()
        return await metadata_handler.is_todo_complete(
            self._workflow.session_id, current_todo_id
        )

    def _get_last_assistant_message(self) -> str | None:
        """
        Get the last assistant message from memory.

        Returns:
            The content of the last assistant message, or None if not found.
        """
        messages = self._workflow.planner.memory.get_messages()
        for msg in reversed(messages):
            if msg.role == "assistant":
                content = msg.content
                if content and (isinstance(content, str) and content.strip()):
                    return content
                return None
        return None

    async def _generate_session_title(self, message: str) -> None:
        """Generate session title in background."""
        try:
            title = await agenerate_session_title(
                provider=self._workflow.planner.provider,
                query=message,
            )
            self._session_title = title
        except Exception:
            pass

    async def _handle_terminal_command(self, command: str) -> None:
        """Handle terminal commands starting with !."""
        cmd = command.strip()
        if not cmd:
            return

        self.console.print(f"\n  [kader.orange]\\[>] Executing:[/kader.orange] `{cmd}`")

        try:
            output = await self._run_terminal_command_direct(cmd)

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

    async def _run_terminal_command_direct(self, command: str) -> str:
        """Run a terminal command directly without PTY."""

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await process.communicate()

            output = stdout.decode("utf-8", errors="replace").strip()
            return output
        except Exception as e:
            return f"Error: {str(e)}"

    async def _handle_save_session(self) -> None:
        """Save the current session."""
        import shutil

        from kader.memory.types import get_default_memory_dir

        try:
            if not self._current_session_id:
                session = await self._session_manager.async_create_session("kader_cli")
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

    async def _handle_load_session(self, session_id: str) -> None:
        """Load a saved session by ID."""
        try:
            session = await self._session_manager.async_get_session(session_id)
            if not session:
                self.console.print(
                    rf"  [kader.red]\[-][/kader.red] Session `{session_id}` not found. "
                    "Use `/sessions` to see available sessions."
                )
                return

            messages = await self._session_manager.async_load_conversation(session_id)
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

    async def _handle_list_sessions(self) -> None:
        """List all saved sessions."""
        try:
            sessions = await self._session_manager.async_list_sessions()

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

    async def _handle_commands(self) -> None:
        """Handle the /commands command to display special commands."""
        try:
            from kader.tools.commands import CommandLoader

            loader = CommandLoader()
            commands = loader.list_commands()

            if not commands:
                self.console.print(
                    "  [dim]No special commands found in `~/.kader/commands` "
                    "or `./.kader/commands`[/dim]"
                )
                self.console.print(
                    "  [dim]Create a command directory with CONTENT.md to add a special command.[/dim]"
                )
            else:
                table = Table(
                    title="[kader.cyan]Special Commands[/kader.cyan]",
                    border_style="cyan",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("Command", style="bold white")
                table.add_column("Description", style="dim")

                for cmd in commands:
                    table.add_row(f"/{cmd.name}", cmd.description)

                self.console.print()
                self.console.print(table)
                self.console.print("  [dim]Usage: /<command> <task>[/dim]")
        except Exception as e:
            self.console.print(
                f"  [kader.red]✗[/kader.red] Error loading commands: {e}"
            )

    def _get_special_command(self, command_input: str) -> tuple[str, str] | None:
        """Parse command input and check if it's a special command.

        Supports formats:
        - /command task
        - /command/subcommand task

        Args:
            command_input: The raw command input (e.g., "/mycommand do something")

        Returns:
            Tuple of (command_name, user_task) if it's a special command, None otherwise
        """
        if not command_input.startswith("/"):
            return None

        # Remove leading / and get the rest
        rest = command_input[1:].strip()

        # Determine the full command name (including sub-command)
        # Format: command/subcommand task or command task
        if " " in rest:
            # Has space: command task or command/subcommand task
            cmd_with_sub = rest.split()[0]
        elif "/" in rest:
            # Has slash but no space: command/subcommand
            cmd_with_sub = rest
        else:
            # Just command name
            cmd_with_sub = rest

        # Parse command name and user task
        if "/" in cmd_with_sub:
            cmd_parts = cmd_with_sub.split("/", 1)
            command_name = cmd_parts[0]
            remaining = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if remaining:
                task_parts = remaining.split(maxsplit=1)
                user_task = task_parts[1] if len(task_parts) > 1 else ""
            else:
                user_task = ""
        else:
            parts = rest.split(maxsplit=1)
            command_name = parts[0] if parts else ""
            user_task = parts[1] if len(parts) > 1 else ""

        if not command_name:
            return None

        try:
            from kader.tools.commands import CommandLoader

            loader = CommandLoader()

            # Try to load with full command name (including sub-command)
            full_command_name = cmd_with_sub if "/" in cmd_with_sub else command_name
            command = loader.load_command(full_command_name)
            if command:
                return (full_command_name, user_task)

            # Also try without sub-command part for top-level commands
            if "/" in full_command_name:
                top_level = full_command_name.split("/")[0]
                command = loader.load_command(top_level)
                if command:
                    return (full_command_name, user_task)
        except Exception:
            pass

        return None

    async def _handle_special_command(
        self, command_name: str, user_task: str, full_input: str
    ) -> None:
        """Handle a special command by executing it via AgentTool.

        Args:
            command_name: The name of the special command
            user_task: The task to execute
            full_input: The full original input for display
        """
        if self._is_processing:
            self.console.print(
                r"  [kader.yellow]\[!][/kader.yellow] "
                "Please wait for the current response..."
            )
            return

        self._is_processing = True
        self._print_user_message(full_input)

        try:
            from kader.tools import AgentTool
            from kader.tools.commands import CommandLoader

            loader = CommandLoader()
            command = loader.load_command(command_name)

            if not command:
                self.console.print(
                    rf"  [kader.red]\[-][/kader.red] Command `{command_name}` not found."
                )
                return

            self._start_spinner()

            agent_tool = AgentTool(
                name=command_name,
                description=command.description,
                provider=self._workflow.planner.provider,
                model_name=self._workflow.planner.provider.model,
                interrupt_before_tool=True,
                tool_confirmation_callback=self._tool_confirmation_callback,
                direct_execution_callback=self._direct_execution_callback,
                tool_execution_result_callback=self._tool_execution_result_callback,
                custom_system_prompt=command.content,
            )

            result = await agent_tool.aexecute(
                task=user_task or "Execute the command with no specific task",
                context="",
            )

            self._stop_spinner()

            if result:
                self._print_assistant_message(
                    result,
                    model_name=self._workflow.planner.provider.model,
                    usage_cost=self._workflow.planner.provider.total_cost.total_cost,
                    session_title=self._session_title,
                )
            else:
                self.console.print(
                    r"  [kader.yellow]\[!][/kader.yellow] Command completed with no output."
                )

        except Exception as e:
            self._stop_spinner()
            self.console.print(
                rf"  [kader.red]\[-] Error executing command `{command_name}`:[/kader.red] {e}"
            )

        finally:
            self._is_processing = False

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

        cwd = os.getcwd()
        git_info = ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                if branch:
                    git_info = f" · {branch}"
        except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
            pass

        self.console.print(WELCOME_BANNER)
        self.console.print(
            f"  [dim]v{version} · "
            f"Model: {self._current_model} · "
            f"Type /help for commands[/dim]\n"
        )
        self.console.print(f"  [dim]{cwd}{git_info}[/dim]")

    # ── Main loop ────────────────────────────────────────────────────

    def _get_prompt_text(self) -> HTML:
        """Get the styled prompt text."""
        return HTML("<ansimagenta><b>[>] </b></ansimagenta>")

    async def _run_async(self) -> None:
        """Async main loop."""
        self._check_for_updates()
        self._print_welcome()
        self._show_update_notification()

        while self._running:
            try:
                # Normal input
                with patch_stdout():
                    user_input = await self._prompt_session.prompt_async(
                        self._get_prompt_text()
                    )

                user_input = user_input.strip()
                if not user_input:
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
        from kader.tools.exec_commands import CommandExecutorTool

        enter_fullscreen()
        try:
            asyncio.run(self._run_async())
        except KeyboardInterrupt:
            self.console.print("\n  [kader.muted]Goodbye! 👋[/kader.muted]")
        finally:
            self._stop_spinner()
            CommandExecutorTool.clear_callbacks()
            exit_fullscreen()


def main() -> None:
    """Run the Kader CLI application."""
    app = KaderApp()
    app.run()


if __name__ == "__main__":
    main()
