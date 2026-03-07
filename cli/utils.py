"""Utility constants and helpers for Kader CLI."""

from dataclasses import dataclass

from .llm_factory import LLMProviderFactory

DEFAULT_MODEL = "minimax-m2.5:cloud"


@dataclass
class CLICommand:
    name: str
    description: str
    has_args: bool = False
    arg_hint: str = ""


def get_special_commands() -> list[CLICommand]:
    """Get special commands from ~/.kader/commands and ./.kader/commands."""
    try:
        from kader.tools.commands import CommandLoader
    except Exception:
        return []

    try:
        loader = CommandLoader()
        commands = loader.list_commands()
        return [
            CLICommand(
                name=f"/{cmd.name}",
                description=cmd.description,
                has_args=True,
                arg_hint="<task>",
            )
            for cmd in commands
        ]
    except Exception:
        return []


COMMANDS: list[CLICommand] = [
    CLICommand(name="/help", description="Show help message"),
    CLICommand(name="/models", description="Show and switch LLM models"),
    CLICommand(name="/clear", description="Clear the conversation"),
    CLICommand(name="/save", description="Save current session"),
    CLICommand(
        name="/load",
        description="Load a saved session",
        has_args=True,
        arg_hint="<session_id>",
    ),
    CLICommand(name="/sessions", description="List saved sessions"),
    CLICommand(name="/skills", description="List loaded skills"),
    CLICommand(name="/commands", description="List special commands"),
    CLICommand(name="/cost", description="Show usage costs"),
    CLICommand(name="/init", description="Initialize .kader directory with KADER.md"),
    CLICommand(name="/exit", description="Exit the CLI"),
]

SPECIAL_COMMANDS: list[CLICommand] = get_special_commands()

ALL_COMMANDS: list[CLICommand] = COMMANDS + SPECIAL_COMMANDS

COMMAND_NAMES: list[str] = [cmd.name for cmd in ALL_COMMANDS]


def get_commands_text() -> str:
    """Get formatted text of available commands."""
    lines = [
        "## Kader CLI Commands\n",
        "| Command | Description |",
        "|---------|-------------|",
    ]
    for cmd in COMMANDS:
        arg = f" {cmd.arg_hint}" if cmd.has_args else ""
        lines.append(f"| `{cmd.name}`{arg} | {cmd.description} |")

    if SPECIAL_COMMANDS:
        lines.append("\n### Special Commands")
        lines.append("| Command | Description |")
        lines.append("|---------|-------------|")
        for cmd in SPECIAL_COMMANDS:
            arg = f" {cmd.arg_hint}" if cmd.has_args else ""
            lines.append(f"| `{cmd.name}`{arg} | {cmd.description} |")

    lines.append("\n### Tips:")
    lines.append("- Type any question to chat with the AI")
    lines.append("- Use `/` to access command menu (arrow keys to navigate)")
    lines.append("- Model format: `provider:model` (e.g., `google:gemini-2.5-flash`)")
    lines.append("- Use `Ctrl+C` to cancel, `Ctrl+D` to exit")
    return "\n".join(lines)


HELP_TEXT = """## Kader CLI Commands

| Command | Description |
|---------|-------------|
| `/models` | Show and switch LLM models |
| `/help` | Show this help message |
| `/clear` | Clear the conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/commands` | List special commands |
| `/cost` | Show usage costs |
| `/init` | Initialize .kader directory with KADER.md |
| `/exit` | Exit the CLI |
| `!cmd` | Run terminal command |

### Tips:
- Type any question to chat with the AI
- Model format: `provider:model` (e.g., `google:gemini-2.5-flash`)
- Use `Ctrl+C` to cancel, `Ctrl+D` to exit
"""


def get_models_text() -> str:
    """Get formatted text of available models from all providers."""
    try:
        all_models = LLMProviderFactory.get_all_models()
        flat_list = LLMProviderFactory.get_flat_model_list()

        if not flat_list:
            return "No models found. Check provider configurations."

        lines = [
            "## Available Models\n",
            "| Provider | Model | Status |",
            "|----------|-------|--------|",
        ]
        for provider_name, provider_models in all_models.items():
            for model in provider_models:
                lines.append(f"| {provider_name.title()} | `{model}` | ✓ Available |")

        lines.append(f"\n*Currently using: **{DEFAULT_MODEL}***")
        lines.append(
            "\n> Tip: Use `provider:model` format (e.g., `google:gemini-2.5-flash`)"
        )
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching models: {e}"
