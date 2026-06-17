"""Kader CLI - Typer-based command-line interface.

Usage:
  kader                    Launch the interactive Kader AI coding agent
  kader chat -q "..."      Send a one-shot query to the AI agent (no persistence)
  kader connect            Connect an LLM provider by setting its API key
  kader init               Initialize .kader directory and generate KADER.md
  kader model              Show and switch LLM models
  kader sessions           List saved sessions and resume one
  kader sessions --resume  Resume a session by ID
  kader update             Check for and install updates
  kader --version          Show the installed version
  kader --help             Show this help message
"""

import asyncio
import subprocess
import sys
from importlib.metadata import version as get_version
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
from tenacity import RetryError

from cli.callbacks import load_callbacks_from_settings
from cli.commands.update import check_outdated
from cli.sessions_metadata import SessionsMetadataManager
from cli.settings import load_settings, save_settings
from cli.tools import load_tools_from_settings
from kader.config import ENV_FILE_PATH, initialize_kader_config, save_env_var
from kader.prompts.cli_prompts import InitCommandPrompt
from kader.providers import LLMProviderFactory
from kader.tools.agent import AgentTool
from kader.workflows import PlannerExecutorWorkflow

KADER_THEME = Theme(
    {
        "kader.primary": "bold magenta",
        "kader.cyan": "bold cyan",
        "kader.yellow": "bold yellow",
        "kader.green": "bold green",
        "kader.red": "bold red",
        "kader.orange": "bold dark_orange",
        "kader.muted": "dim",
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


class BannerTyper(typer.Typer):
    """Typer app that prints the Kader welcome banner before any --help output."""

    def __call__(self, *args: object, **kwargs: object) -> None:
        if "--help" in sys.argv or "-h" in sys.argv:
            console.print(WELCOME_BANNER)
            console.print()
        super().__call__(*args, **kwargs)


app = BannerTyper(
    name="kader",
    help="Kader - AI coding agent framework.",
    add_completion=False,
    context_settings={"help_option_names": ["--help", "-h"]},
)

console = Console(theme=KADER_THEME)


def _load_model_string() -> str:
    """Load the main agent model string from ~/.kader/settings.json."""
    settings = load_settings()
    return settings.get_main_model_string()


@app.callback(invoke_without_command=True)
def app_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show the installed version of Kader.",
    ),
) -> None:
    """Kader - AI coding agent framework."""
    if version:
        try:
            ver = get_version("kader")
        except Exception:
            ver = "unknown"
        console.print(f"kader {ver}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        from cli.app import KaderApp

        interactive_app = KaderApp()
        interactive_app.run()


@app.command(name="init")
def init_cmd() -> None:
    """Initialize .kader directory and generate KADER.md."""
    initialize_kader_config()

    kader_dir = Path.cwd() / ".kader"

    # Step 1: Create .kader directory
    try:
        kader_dir.mkdir(exist_ok=True)
        console.print(
            f"  [kader.cyan]\u25b6[/kader.cyan] Created directory: `{kader_dir}`"
        )
    except Exception as e:
        console.print(
            f"  [kader.red]\u2717[/kader.red] Failed to create directory: {e}"
        )
        raise typer.Exit(code=1)

    # Step 2: Check if KADER.md already exists
    kader_md_path = kader_dir / "KADER.md"
    if kader_md_path.exists():
        console.print(
            f"  [kader.yellow][!][/kader.yellow] "
            f"KADER.md already exists at `{kader_md_path}`"
        )
        console.print(
            "  [kader.cyan]\u25b6[/kader.cyan] "
            "Re-generating KADER.md with updated analysis..."
        )

    # Step 3: Generate KADER.md using AgentTool
    console.print(
        "  [kader.cyan]\u25b6[/kader.cyan] Analyzing codebase and generating KADER.md..."
    )

    model_string = _load_model_string()

    try:
        provider = LLMProviderFactory.create_provider(model_string)

        prompt = InitCommandPrompt()
        task = str(prompt)

        agent_tool = AgentTool(
            name="init_agent",
            description="Agent to initialize KADER.md by analyzing codebase",
            provider=provider,
            model_name=model_string,
            interrupt_before_tool=False,
        )

        context = (
            f"Working directory: {Path.cwd()}\n"
            f"Target file: {kader_md_path}\n"
            f"Use filesystem tools to explore the codebase "
            f"and create the KADER.md file."
        )

        result = asyncio.run(agent_tool.aexecute(task=task, context=context))

        console.print(
            f"  [kader.green]\u2713[/kader.green] "
            f"Successfully created KADER.md at `{kader_md_path}`"
        )

        preview = result[:500] + "..." if len(result) > 500 else result
        console.print(f"\n```markdown\n{preview}\n```")

    except Exception as e:
        console.print(f"  [kader.red]\u2717[/kader.red] Error generating KADER.md: {e}")
        raise typer.Exit(code=1)


@app.command(name="model")
def model_cmd(
    agent: str | None = typer.Option(
        None,
        "--agent",
        "-a",
        help="Agent type to switch models for: 'main' (orchestrator) or 'sub' (executor).",
    ),
) -> None:
    """Show and switch LLM models for an agent."""
    settings = load_settings()
    current_main = settings.get_main_model_string()
    current_sub = settings.get_sub_model_string()

    # Step 1: Determine agent type
    if agent and agent.lower() in ("main", "sub"):
        agent_type = agent.lower()
    else:
        console.print()
        console.print(
            f"  [dim]Current main agent model:[/dim] [bold]{current_main}[/bold]"
        )
        console.print(
            f"  [dim]Current sub agent model:[/dim]  [bold]{current_sub}[/bold]"
        )
        console.print()
        console.print("  [bold]Which agent to update?[/bold]")
        console.print("  [1] Main Agent \u2014 planner / orchestrator")
        console.print("  [2] Sub Agent \u2014 executor / worker")
        console.print()
        choice = input("  choice (1/2): ").strip()
        if choice == "2":
            agent_type = "sub"
        else:
            agent_type = "main"

    is_main = agent_type == "main"
    agent_label = "Main Agent" if is_main else "Sub Agent"
    current_for_agent = current_main if is_main else current_sub

    # Step 2: Fetch and display models
    try:
        models = LLMProviderFactory.get_flat_model_list()
        if not models:
            console.print(
                "  [kader.red]\u2717[/kader.red] No models found. "
                "Check provider configurations."
            )
            raise typer.Exit(code=1)

        table = Table(
            title=f"[kader.cyan]Available Models \u2014 {agent_label}[/kader.cyan]",
            border_style="cyan",
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("Model", style="white")
        table.add_column("Status", justify="center")

        for i, model in enumerate(models, 1):
            if model == current_for_agent:
                marker = "[kader.green]\u25cf current[/kader.green]"
            else:
                marker = "[dim]available[/dim]"
            table.add_row(str(i), model, marker)

        console.print()
        console.print(table)
        console.print(
            "  [dim]Enter model number to switch, or press Enter to cancel:[/dim]"
        )

        model_choice = input("  model> ").strip()
        if not model_choice:
            console.print(
                f"  [dim]{agent_label} model selection cancelled. "
                f"Current: `{current_for_agent}`[/dim]"
            )
            return

        idx = int(model_choice) - 1
        if 0 <= idx < len(models):
            selected_model = models[idx]
            provider_name, model_name = LLMProviderFactory.parse_model_name(
                selected_model
            )

            if is_main:
                old_model = current_main
                settings.main_agent_provider = provider_name
                settings.main_agent_model = model_name
            else:
                old_model = current_sub
                settings.sub_agent_provider = provider_name
                settings.sub_agent_model = model_name

            save_settings(settings)
            console.print(
                f"  [kader.green]\u2713[/kader.green] "
                f"{agent_label} model changed from "
                f"`{old_model}` to `{selected_model}`"
            )
        else:
            console.print("  [kader.red]\u2717[/kader.red] Invalid selection.")
            raise typer.Exit(code=1)

    except ValueError:
        console.print(
            f"  [dim]{agent_label} model selection cancelled. "
            f"Current: `{current_for_agent}`[/dim]"
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        console.print(
            f"  [dim]{agent_label} model selection cancelled. "
            f"Current: `{current_for_agent}`[/dim]"
        )
    except Exception as e:
        console.print(f"  [kader.red]\u2717[/kader.red] Error fetching models: {e}")
        raise typer.Exit(code=1)


@app.command(name="update")
def update_cmd() -> None:
    """Check for updates and update Kader if a newer version is available."""
    try:
        current_version = get_version("kader")
    except Exception:
        current_version = "unknown"

    is_outdated, latest_version = check_outdated("kader", current_version)

    if is_outdated:
        console.print(
            f"  [kader.yellow]Updating Kader from v{current_version} "
            f"to v{latest_version}...[/kader.yellow]"
        )
        result = subprocess.run(["uv", "tool", "upgrade", "kader"], capture_output=True)
        if result.returncode == 0:
            console.print(
                f"  [kader.green]\u2713[/kader.green] Updated to v{latest_version}"
            )
        else:
            console.print(
                "  [kader.red]\u2717[/kader.red] Update failed.\n"
                "  Try running: uv tool upgrade kader"
            )
            raise typer.Exit(code=1)
    else:
        console.print(
            f"  [kader.green]\u2713[/kader.green] "
            f"You are running the latest version v{current_version}"
        )


@app.command(name="chat")
def chat_cmd(
    query: str = typer.Option(
        ...,
        "--query",
        "-q",
        help="The query to send to the AI agent.",
    ),
) -> None:
    """Send a one-shot query to the Kader AI agent (no session persistence)."""
    initialize_kader_config()
    settings = load_settings()
    model_string = settings.get_main_model_string()
    executor_model = settings.get_sub_model_string()

    provider = LLMProviderFactory.create_provider(model_string)
    executor_provider = LLMProviderFactory.create_provider(executor_model)

    callbacks = load_callbacks_from_settings(settings)
    planner_tools, executor_tools = load_tools_from_settings(settings)
    enabled_subagents = settings.subagents

    def _direct_execution_callback(
        message: str, tool_name: str, tool_args: dict | None = None
    ) -> None:
        console.print(f"  [kader.cyan]\u26a1 {tool_name}:[/kader.cyan] {message}")

    def _tool_result_callback(
        tool_name: str,
        success: bool,
        result: str,
        tool_args: dict | None = None,
    ) -> None:
        if success:
            console.print(f"  [kader.green]\u2713[/kader.green] {tool_name} completed")
        else:
            preview = result[:200] + "..." if len(result) > 200 else result
            console.print(
                f"  [kader.red]\u2717[/kader.red] {tool_name} failed: {preview}"
            )

    def _tool_confirmation_callback(message: str) -> tuple[bool, str | None]:
        """Prompt user for tool approval via stdin."""
        console.print()
        console.print(
            Panel(
                message,
                title="[kader.yellow]Tool Confirmation[/kader.yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        try:
            answer = input("  Approve? (y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return (False, None)
        if answer in ("y", "yes"):
            console.print("  [kader.green]\u2713[/kader.green] Approved")
            return (True, None)
        else:
            try:
                reason = input("  Reason for rejection: ").strip()
            except (EOFError, KeyboardInterrupt):
                reason = ""
            return (False, reason or None)

    workflow = PlannerExecutorWorkflow(
        name="kader_chat",
        provider=provider,
        model_name=model_string,
        interrupt_before_tool=True,
        tool_confirmation_callback=_tool_confirmation_callback,
        direct_execution_callback=_direct_execution_callback,
        tool_execution_result_callback=_tool_result_callback,
        use_persistence=False,
        executor_model_name=executor_model,
        executor_provider=executor_provider,
        callbacks=callbacks,
        planner_tools=planner_tools,
        executor_tools=executor_tools,
        enabled_subagents=enabled_subagents,
    )

    console.print(f"  [kader.cyan]\u25b6[/kader.cyan] Processing: {query[:100]}...")

    try:
        response = asyncio.run(workflow.arun(query))

        console.print()
        console.print(
            Panel(
                Markdown(response),
                title="[kader.assistant]Kader[/kader.assistant]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    except ConnectionError as e:
        console.print(f"  [kader.red]\u2717[/kader.red] Connection Error: {e}")
        raise typer.Exit(code=1)
    except RetryError as e:
        console.print(f"  [kader.red]\u2717[/kader.red] Retry Exhausted: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"  [kader.red]\u2717[/kader.red] Error: {e}")
        raise typer.Exit(code=1)


@app.command(name="connect")
def connect_cmd(
    provider: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Provider name (e.g., 'openai', 'google', 'anthropic'). Skips selection prompt.",
    ),
) -> None:
    """Connect an LLM provider by setting its API key."""
    provider_names = list(LLMProviderFactory.PROVIDERS.keys())
    provider_names_lower = {p.lower(): p for p in provider_names}

    # Step 1: Determine provider
    if provider and provider.lower() in provider_names_lower:
        provider_name = provider_names_lower[provider.lower()]
    else:
        console.print()
        console.print("  [bold cyan]Connect a Provider[/bold cyan]")
        console.print("  [dim]Select a provider below to set its API key.[/dim]")
        console.print()
        for i, name in enumerate(provider_names, 1):
            env_key = LLMProviderFactory.get_provider_env_key(name)
            console.print(f"  [{i}] {name.title()} \u2014 {env_key}")
        console.print()
        try:
            choice = input("  Provider (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(provider_names):
                provider_name = provider_names[idx]
            else:
                console.print("  [kader.red]\u2717[/kader.red] Invalid selection.")
                raise typer.Exit(code=1)
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("  [dim]Provider selection cancelled.[/dim]")
            raise typer.Exit()

    env_key = LLMProviderFactory.get_provider_env_key(provider_name)

    # Step 2: Prompt for API key
    console.print()
    console.print(
        f"  [dim]Enter API key for[/dim] [bold]{provider_name.title()}[/bold]"
        f" [dim]({env_key})[/dim]:"
    )
    try:
        api_key = input(f"  {env_key}> ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print(f"\n  [dim]{provider_name.title()} not connected.[/dim]")
        raise typer.Exit()

    if not api_key:
        console.print(
            f"  [dim]No API key entered. {provider_name.title()} not connected.[/dim]"
        )
        raise typer.Exit()

    # Step 3: Save API key
    success = save_env_var(ENV_FILE_PATH, env_key, api_key)
    if success:
        console.print()
        console.print(
            f"  [kader.green]\u2713[/kader.green] [bold]{provider_name.title()}[/bold]"
            " connected successfully!"
        )
        console.print(f"  [dim]API key saved to {ENV_FILE_PATH}[/dim]")
        console.print(
            "  [dim]Use [/dim][bold]kader model[/bold][dim] to browse available models.[/dim]"
        )
    else:
        console.print(
            f"  [kader.red]\u2717[/kader.red] Failed to save API key for {provider_name.title()}."
        )
        raise typer.Exit(code=1)


@app.command(name="sessions")
def sessions_cmd(
    resume: str | None = typer.Option(
        None,
        "--resume",
        "-r",
        help="Resume a specific session by ID (skips selection prompt).",
    ),
) -> None:
    """List saved sessions and resume a session."""
    import json

    from cli.app import KaderApp

    manager = SessionsMetadataManager()
    memory_dir = Path.home() / ".kader" / "memory"
    lock_file = memory_dir / "sessions.json.lock"

    if resume:
        if not lock_file.exists():
            manager.update()

        try:
            metadata = json.loads(lock_file.read_text())
        except (json.JSONDecodeError, OSError):
            console.print(
                "  [kader.red]\u2717[/kader.red] "
                f"Session `{resume}` not found \u2014 no valid metadata."
            )
            raise typer.Exit(code=1)

        if resume not in metadata:
            console.print(
                f"  [kader.red]\u2717[/kader.red] Session `{resume}` not found."
            )
            raise typer.Exit(code=1)

        console.print(
            f"  [kader.cyan]\u25b6[/kader.cyan] Resuming session `{resume}`..."
        )
        interactive_app = KaderApp(session_id=resume)
        interactive_app.run()
        return

    if not lock_file.exists():
        manager.update()

    if not lock_file.exists():
        console.print(
            "  [dim]No saved sessions found. Start a conversation to create one.[/dim]"
        )
        return

    try:
        metadata = json.loads(lock_file.read_text())
    except (json.JSONDecodeError, OSError):
        console.print(
            "  [kader.red]\u2717[/kader.red] Failed to read session metadata."
        )
        raise typer.Exit(code=1)

    if not metadata:
        console.print(
            "  [dim]No saved sessions found. Start a conversation to create one.[/dim]"
        )
        return

    sorted_sessions = sorted(
        metadata.items(),
        key=lambda x: x[1].get("creation-date", ""),
        reverse=True,
    )

    table = Table(
        title="[kader.cyan]Saved Sessions[/kader.cyan]",
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Title", style="white")
    table.add_column("Date", style="dim")
    table.add_column("Session ID", style="dim")

    session_ids: list[str] = []
    for i, (session_id, data) in enumerate(sorted_sessions, 1):
        title = data.get("session-title") or session_id
        created = data.get("creation-date", "")[:10]
        if not created:
            created = data.get("update-date", "")[:10]
        table.add_row(str(i), title, created, session_id)
        session_ids.append(session_id)

    console.print()
    console.print(table)
    console.print(
        "  [dim]Enter session number to resume, or press Enter to cancel:[/dim]"
    )

    try:
        choice = input("  sessions> ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        console.print("  [dim]Session selection cancelled.[/dim]")
        return

    if not choice:
        console.print("  [dim]Session selection cancelled.[/dim]")
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(session_ids):
            selected_id = session_ids[idx]
            console.print(
                f"  [kader.cyan]\u25b6[/kader.cyan] Resuming session `{selected_id}`..."
            )
            interactive_app = KaderApp(session_id=selected_id)
            interactive_app.run()
        else:
            console.print("  [kader.red]\u2717[/kader.red] Invalid selection.")
            raise typer.Exit(code=1)
    except ValueError:
        console.print("  [kader.red]\u2717[/kader.red] Invalid input.")
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the kader command."""
    app()


if __name__ == "__main__":
    main()
