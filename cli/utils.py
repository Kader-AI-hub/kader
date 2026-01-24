"""Utility constants and helpers for Kader CLI."""

from kader.providers import OllamaProvider

# Default model
DEFAULT_MODEL = "qwen3-coder:480b-cloud"

HELP_TEXT = """## Kader CLI Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available LLM models |
| `/help` | Show this help message |
| `/clear` | Clear the conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/cost` | Show usage costs |
| `/refresh` | Refresh file tree |
| `/exit` | Exit the CLI |

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
"""


def get_models_text() -> str:
    """Get formatted text of available Ollama models."""
    try:
        models = OllamaProvider.get_supported_models()
        if not models:
            return "## Available Models (^^)\n\n*No models found. Is Ollama running?*"

        lines = [
            "## Available Models (^^)\n",
            "| Model | Status |",
            "|-------|--------|",
        ]
        for model in models:
            lines.append(f"| {model} | (+) Available |")
        lines.append(f"\n*Currently using: **{DEFAULT_MODEL}***")
        return "\n".join(lines)
    except Exception as e:
        return f"## Available Models (^^)\n\n*Error fetching models: {e}*"
