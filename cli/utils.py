"""Utility constants and helpers for Kader CLI."""

from .llm_factory import LLMProviderFactory

# Default model (with provider prefix for clarity)
DEFAULT_MODEL = "minimax-m2.5:cloud"

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
