# Kader CLI

A modern terminal-based AI coding assistant built with Python's [Textual](https://textual.textualize.io/) framework, powered by **ReActAgent** with tool execution capabilities.

## Features

- 🤖 **ReAct Agent** - Intelligent agent with reasoning and tool execution
- 🛠️ **Built-in Tools** - File system, command execution, web search
- 📁 **Directory Tree** - Auto-refreshing sidebar showing current working directory
- 💬 **Conversation View** - Markdown-rendered chat history
- 💾 **Session Persistence** - Save and load conversation sessions
- 🎨 **Color Themes** - 4 themes (dark, ocean, forest, sunset)
- 🔧 **Tool Confirmation** - Interactive approval for tool execution
- 🤖 **Model Selection** - Dynamic model switching interface
- 📝 **File Operations** - Integrated file system tools for coding tasks
- ☁️ **Multi-Provider Support** - Ollama, Google Gemini, Mistral, OpenAI, Moonshot (Kimi), Z.ai (GLM), OpenRouter, OpenCode, Groq
- 🖥️ **CLI Message Display** - Enhanced display showing agent reasoning and tool execution

## Prerequisites

- [Ollama](https://ollama.ai/) running locally (for local models)
- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or [pip](https://pypi.org/project/pip/)
- API keys for cloud providers (optional, based on model selection)

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install dependencies
uv sync

# Run the CLI
uv run python -m cli
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install dependencies
pip install -e .

# Run the CLI
python -m cli
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command reference |
| `/models` | Show available Ollama models |
| `/theme` | Cycle color themes |
| `/clear` | Clear conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/refresh` | Refresh file tree |
| `/cost` | Show usage costs |
| `/exit` | Exit the CLI |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear conversation |
| `Ctrl+T` | Cycle theme |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Refresh file tree |
| `Tab` | Navigate panels |

## Input Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Copy selected text |
| `Ctrl+V` | Paste from clipboard |
| `Ctrl+A` | Select all text |
| Click+Drag | Select text |

## Session Management

Sessions are saved to `~/.kader/sessions/`. Use:

- `/save` to save current conversation
- `/sessions` to list all saved sessions
- `/load <session_id>` to restore a session

## Tool Confirmation System

Kader includes an interactive tool confirmation system that prompts for approval before executing tools. This provides:

- Safe execution of potentially destructive operations
- Interactive approval with arrow keys and Enter
- Quick confirmation with Y/N keys
- Visual feedback during tool execution

## Skills System

Kader supports skills - specialized instructions for specific domains or tasks. Skills are loaded from:

- `~/.kader/skills/` (user-level skills)
- `./.kader/skills/` (project-level skills)

Use `/skills` to list all available skills. Each skill is a directory containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: python-expert
description: Expert in Python programming and best practices
---

# Python Expert Skill

You are an expert Python developer...
```

Agents can load skills dynamically using the SkillsTool to get specialized instructions for specific domains.

## Model Selection Interface

The model selection interface allows you to:

- Browse available Ollama models
- Switch models on the fly during conversation
- See which models are currently installed
- Cancel selection without changing the current model

## Project Structure

```
cli/
├── app.py          # Main application (ReActAgent integration)
├── app.tcss        # Styles (TCSS)
├── utils.py        # Constants and helpers
├── __init__.py     # Package exports
├── __main__.py     # Entry point
└── widgets/        # Custom UI components
    ├── __init__.py
    ├── conversation.py  # Chat display
    ├── loading.py       # Spinner animation
    └── confirmation.py  # Tool and model selection widgets
```

## Changing the Model

The default model is set in `utils.py`. Kader supports multiple providers with the format `provider:model`:

```python
DEFAULT_MODEL = "minimax-m2.5:cloud"  # Default model
```

### Supported Providers

| Provider | Format | Example |
|----------|--------|---------|
| Ollama | `ollama:model` | `ollama:llama3` |
| Google Gemini | `google:model` | `google:gemini-2.5-flash` |
| Mistral | `mistral:model` | `mistral:small-3.1` |
| OpenAI | `openai:model` | `openai:gpt-4o` |
| Moonshot (Kimi) | `moonshot:model` | `moonshot:kimi-k2.5` |
| Z.ai (GLM) | `zai:model` | `zai:glm-5` |
| OpenRouter | `openrouter:model` | `openrouter:anthropic/claude-3.5-sonnet` |
| OpenCode | `opencode:model` | `opencode:claude-3.5-sonnet` |
| Groq | `groq:model` | `groq:llama-3.3-70b-versatile` |

### Environment Variables

Set API keys for cloud providers:

```bash
export GOOGLE_API_KEY="your-google-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export MOONSHOT_API_KEY="your-kimi-api-key"
export ZAI_API_KEY="your-glm-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENCODE_API_KEY="your-opencode-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

## Development

Run with live CSS reloading:

```bash
uv run textual run --dev cli.app:KaderApp
```

## Configuration

Kader automatically creates a `.kader` directory in your home directory on first run. This stores:

- Session data in `~/.kader/sessions/`
- Configuration files in `~/.kader/`
- Memory files in `~/.kader/memory/`

## Troubleshooting

### Common Issues

- **No models found**: Make sure Ollama is running and you have at least one model installed (e.g., `ollama pull minimax-m2.5:cloud`)
- **Connection errors**: Verify that Ollama service is accessible at the configured endpoint
- **Theme not changing**: Some terminal emulators may not support all color themes

### Debugging

If you encounter issues:

1. Check that Ollama is running: `ollama serve`
2. Verify your model is pulled: `ollama list`
3. Ensure your terminal supports the required features
4. Check the logs for specific error messages
