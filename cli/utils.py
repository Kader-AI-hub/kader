"""Utility constants and helpers for Kader CLI."""

from kader.providers import OllamaProvider

# Theme names for cycling
THEME_NAMES = ["dark", "ocean", "forest", "sunset"]

# Default model
DEFAULT_MODEL = "gpt-oss:120b-cloud"

HELP_TEXT = """## Kader CLI Commands 📖

| Command | Description |
|---------|-------------|
| `/models` | Show available LLM models |
| `/theme` | Cycle through color themes |
| `/help` | Show this help message |
| `/clear` | Clear the conversation |
| `/new` | Start a new conversation |
| `/exit` | Exit the CLI |

### Tips:
- Type any question to chat with the AI
- Use **Tab** to navigate between panels
- Press **Ctrl+C** to cancel current operation"""


def get_models_text() -> str:
    """Get formatted text of available Ollama models."""
    try:
        models = OllamaProvider.get_supported_models()
        if not models:
            return "## Available Models 🤖\n\n*No models found. Is Ollama running?*"
        
        lines = ["## Available Models 🤖\n", "| Model | Status |", "|-------|--------|"]
        for model in models:
            lines.append(f"| {model} | ✅ Available |")
        lines.append(f"\n*Currently using: **{DEFAULT_MODEL}***")
        return "\n".join(lines)
    except Exception as e:
        return f"## Available Models 🤖\n\n*Error fetching models: {e}*"

