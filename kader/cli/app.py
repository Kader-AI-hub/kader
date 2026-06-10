"""Kader CLI - Typer-based command-line interface.

Usage:
  kader              Launch the interactive Kader AI coding agent
  kader init         Initialize .kader directory and generate KADER.md
  kader model        Show and switch LLM models
  kader update       Check for and install updates
  kader --version    Show the installed version
  kader --help       Show this help message
"""

import asyncio
import subprocess
from importlib.metadata import version as get_version
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.theme import Theme

from cli.commands.update import check_outdated
from cli.settings import load_settings, save_settings
from kader.config import initialize_kader_config
from kader.prompts.cli_prompts import InitCommandPrompt
from kader.providers import LLMProviderFactory
from kader.tools.agent import AgentTool

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

app = typer.Typer(
    name="kader",
    help="Kader - AI coding agent framework.",
    add_completion=False,
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


def main() -> None:
    """Entry point for the kader command."""
    app()


if __name__ == "__main__":
    main()
