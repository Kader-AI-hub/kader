# Kader CLI

A modern terminal-based AI coding assistant built with [Rich](https://github.com/Textualize/rich) and [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/), powered by **PlannerExecutorWorkflow** with tool execution capabilities.

## Features

- 🤖 **Planner-Executor Workflow** — Intelligent agent with reasoning, planning, and tool execution
- 🛠️ **Built-in Tools** — File system, command execution, web search
- 💬 **Rich Conversation** — Beautiful markdown-rendered chat with styled panels
- 💾 **Session Persistence** — Save and load conversation sessions
- 🔧 **Tool Confirmation** — Interactive approval for tool execution
- 🤖 **Model Selection** — Dynamic model switching interface
- 📝 **File Operations** — Integrated file system tools for coding tasks
- ☁️ **Multi-Provider Support** — Ollama, Google Gemini, Anthropic, Mistral, OpenAI, Moonshot (Kimi), Z.ai (GLM), OpenRouter, OpenCode, Groq
- 🖥️ **CLI Message Display** — Enhanced display showing agent reasoning and tool execution

## Prerequisites

- [Ollama](https://ollama.ai/) running locally (for local models) or [Ollama Cloud](https://ollama.com) (for cloud models)
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
| `/models` | Show and switch available models |
| `/clear` | Clear conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/commands` | List special commands |
| `/cost` | Show usage costs |
| `/init` | Initialize .kader directory with KADER.md |
| `/exit` | Exit the CLI |
| `!cmd` | Run terminal command |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current operation |
| `Ctrl+D` | Exit the CLI |

## Session Management

Sessions are saved to `~/.kader/sessions/`. Use:

- `/save` to save current conversation
- `/sessions` to list all saved sessions
- `/load <session_id>` to restore a session

## Tool Confirmation System

Kader includes an interactive tool confirmation system that prompts for approval before executing tools. This provides:

- Safe execution of potentially destructive operations
- Simple `[Y/n/reason]` prompt for quick approval
- Ability to provide context when rejecting a tool
- Visual feedback during tool execution

## Skills System

Kader supports skills — specialized instructions for specific domains or tasks. Skills are loaded from:

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

## Special Commands

Kader supports special commands — custom command agents that can be invoked from the CLI. Commands are loaded from:

- `./.kader/commands/` (project-level commands, higher priority)
- `~/.kader/commands/` (user-level commands)

Use `/commands` to list all available special commands.

### Creating a Command

Create a command directory with a `CONTENT.md` file:

```bash
mkdir -p ~/.kader/commands/mycommand
```

**CONTENT.md format:**
```yaml
---
description: What this command does
---

# Command Instructions

Your command agent instructions here...

## Guidelines
- Guideline 1
- Guideline 2
```

### Using Commands

Execute a command with:
```
/mycommand
/mycommand do something specific
```

### Example: Lint and Test Command

```yaml
---
description: Lint and test the codebase following AGENTS.md
---

You are a Lint and Test Agent specialized in maintaining code quality.

## Instructions

1. Run: uv run ruff check .
2. Run: uv run ruff format --check .
3. Run: uv run pytest -v
4. Report results
```

Usage: `/lint-test` or `/lint-test run full check`

## Model Selection Interface

The model selection interface allows you to:

- Browse all available models from configured providers
- Switch models on the fly during conversation
- See which model is currently active
- Cancel selection without changing the current model

## Project Structure

```
cli/
├── app.py          # Main application (Rich + prompt_toolkit)
├── utils.py        # Constants and helpers
├── llm_factory.py  # Multi-provider LLM factory
├── __init__.py     # Package exports
├── __main__.py     # Entry point
└── commands/       # CLI command handlers
    ├── __init__.py
    ├── base.py          # Base command class
    └── initialize.py    # /init command
```

## Changing the Model

The default model is set in `utils.py`. Kader supports multiple providers with the format `provider:model`:

```python
DEFAULT_MODEL = "minimax-m2.5:cloud"  # Default model
```

### Supported Providers

| Provider | Format | Example |
|----------|--------|---------|
| Ollama (local) | `ollama:model` | `ollama:llama3` |
| Ollama (cloud) | `ollama:model:cloud` | `ollama:minimax-m2.5:cloud` |
| Google Gemini | `google:model` | `google:gemini-2.5-flash` |
| Mistral | `mistral:model` | `mistral:small-3.1` |
| Anthropic | `anthropic:model` | `anthropic:claude-3.5-sonnet` |
| OpenAI | `openai:model` | `openai:gpt-4o` |
| Moonshot (Kimi) | `moonshot:model` | `moonshot:kimi-k2.5` |
| Z.ai (GLM) | `zai:model` | `zai:glm-5` |
| OpenRouter | `openrouter:model` | `openrouter:anthropic/claude-3.5-sonnet` |
| OpenCode | `opencode:model` | `opencode:claude-3.5-sonnet` |
| Groq | `groq:model` | `groq:llama-3.3-70b-versatile` |

### Environment Variables

Set API keys for cloud providers:

```bash
# Ollama Cloud (get from https://ollama.com/settings)
export OLLAMA_API_KEY="your-ollama-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Other providers...
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export MISTRAL_API_KEY="your-mistral-api-key"
export OPENAI_API_KEY="your-openapi-key"
export MOONSHOT_API_KEY="your-kimi-api-key"
export ZAI_API_KEY="your-glm-api-key"
export OPENROUTER_API_KEY="your-openrouter-api-key"
export OPENCODE_API_KEY="your-opencode-api-key"
export GROQ_API_KEY="your-groq-api-key"
```

## Configuration

Kader automatically creates a `.kader` directory in your home directory on first run. This stores:

- Session data in `~/.kader/sessions/`
- Configuration files in `~/.kader/`
- Memory files in `~/.kader/memory/`

## Troubleshooting

### Common Issues

- **No models found**: Make sure your provider is configured and API keys are set
- **Connection errors**: Verify that the provider service is accessible
- **Import errors**: Run `uv sync` to ensure all dependencies are installed

### Debugging

If you encounter issues:

1. Check that your provider is configured: API keys set or Ollama running
2. Verify your model is available: `/models` to list
3. Check the logs for specific error messages
