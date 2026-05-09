"""Kader CLI - Typer-based command-line interface.

Provides two commands:
  kader-cli --help   Show help
  kader-cli init     Initialize .kader directory and generate KADER.md
"""

import asyncio
import json
from importlib.metadata import version as get_version
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme

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
    name="kader-cli",
    help="Kader CLI - AI coding agent framework.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console(theme=KADER_THEME)


def _load_model_string() -> str:
    """Load the main agent model string from ~/.kader/settings.json."""
    settings_path = Path.home() / ".kader" / "settings.json"
    if not settings_path.exists():
        return "ollama:glm-5:cloud"

    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        provider = data.get("main-agent-provider", "ollama")
        model = data.get("main-agent-model", "glm-5:cloud")
        return f"{provider}:{model}"
    except (json.JSONDecodeError, ValueError, TypeError):
        return "ollama:glm-5:cloud"


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
    """Kader CLI - AI coding agent framework."""
    if version:
        try:
            ver = get_version("kader")
        except Exception:
            ver = "unknown"
        console.print(f"kader {ver}")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        console.print(Markdown("# Kader CLI"))
        console.print()
        console.print(
            "Run [bold]kader-cli init[/bold] to initialize a .kader directory "
            "and generate KADER.md."
        )
        console.print()
        console.print("Commands:")
        console.print()
        console.print(
            "  [bold]init[/bold]  Initialize .kader directory and generate KADER.md"
        )
        console.print()


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


def main() -> None:
    """Entry point for the kader-cli command."""
    app()


if __name__ == "__main__":
    main()
