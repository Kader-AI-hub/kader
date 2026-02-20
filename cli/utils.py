"""Utility constants and helpers for Kader CLI."""

from .llm_factory import LLMProviderFactory

# Default model (with provider prefix for clarity)
DEFAULT_MODEL = "ollama:kimi-k2.5:cloud"

HELP_TEXT = """## Kader CLI Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available LLM models |
| `/help` | Show this help message |
| `/clear` | Clear the conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/cost` | Show usage costs |
| `/refresh` | Refresh file tree |
| `/init` | Initialize .kader directory with KADER.md |
| `/exit` | Exit the CLI |
| `!cmd` | Run terminal command |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+L` | Clear conversation |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Refresh file tree |
| `Ctrl+Q` | Quit |

### Input Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Copy selected text |
| `Ctrl+V` | Paste from clipboard |
| `Ctrl+A` | Select all text |
| Click+Drag | Select text |

### Tips:
- Type any question to chat with the AI
- Use **Tab** to navigate between panels
- Model format: `provider:model` (e.g., `google:gemini-2.5-flash`)
"""


def get_models_text() -> str:
    """Get formatted text of available models from all providers."""
    try:
        all_models = LLMProviderFactory.get_all_models()
        flat_list = LLMProviderFactory.get_flat_model_list()

        if not flat_list:
            return "## Available Models (^^)\n\n*No models found. Check provider configurations.*"

        lines = [
            "## Available Models (^^)\n",
            "| Provider | Model | Status |",
            "|----------|-------|--------|",
        ]
        for provider_name, provider_models in all_models.items():
            for model in provider_models:
                lines.append(f"| {provider_name.title()} | `{model}` | (+) Available |")

        lines.append(f"\n*Currently using: **{DEFAULT_MODEL}***")
        lines.append(
            "\n> (!) Tip: Use `provider:model` format (e.g., `google:gemini-2.5-flash`)"
        )
        return "\n".join(lines)
    except Exception as e:
        return f"## Available Models (^^)\n\n*Error fetching models: {e}*"
