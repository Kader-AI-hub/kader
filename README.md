# Kader

Kader is an intelligent coding agent designed to assist with software development tasks. It provides a comprehensive framework for building AI-powered agents with advanced reasoning capabilities and tool integration.

## Features

- 🤖 **AI-powered Code Assistance** - Support for multiple LLM providers:
  - **Ollama**: Local LLM execution for privacy and speed.
  - **Google Gemini**: Cloud-based powerful models via the Google GenAI SDK.
- 🖥️ **Interactive CLI** - Modern TUI interface built with Textual:
  - **Lazy Loading**: Efficient directory tree loading for large projects.
  - **TODO Management**: Integrated TODO list widget with automatic updates.
- 🛠️ **Tool Integration** - File system, command execution, web search, and more.
- 🧠 **Memory Management** - State persistence, conversation history, and isolated sub-agent memory.
- 🔁 **Session Management** - Save and load conversation sessions.
- ⌨️ **Keyboard Shortcuts** - Efficient navigation and operations.
- 📝 **YAML Configuration** - Agent configuration via YAML files.
- 🔄 **Planner-Executor Framework** - Sophisticated reasoning and acting architecture using task planning and delegation.
- 🗂️ **File System Tools** - Read, write, search, and edit files with automatic `.gitignore` filtering.
- 🤝 **Agent-As-Tool** - Spawn sub-agents for specific tasks with isolated memory and automated context aggregation.
- 🎯 **Agent Skills** - Modular skill system for specialized domain knowledge and task-specific instructions.

## Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (optional, for local LLMs)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or [pip](https://pypi.org/project/pip/)

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install dependencies with uv
uv sync

# Run the CLI
uv run python -m cli
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install in development mode
pip install -e .

# Run the CLI
python -m cli
```

## Quick Start

### Running the CLI

```bash
# Run the Kader CLI using uv
uv run python -m cli

# Or using pip
python -m cli
```

### First Steps in CLI

Once the CLI is running:

1. Type any question to start chatting with the agent.
2. Use `/help` to see available commands.
3. Use `/models` to check available models from all providers.
4. The directory tree on the left features **lazy loading**, expanding only when needed.
5. The **TODO list** on the right tracks tasks identified by the planner.

## Configuration

When the kader module is imported for the first time, it automatically creates a `.kader` directory in your home directory and a `.env` file.

### Environment Variables

The application automatically loads environment variables from `~/.kader/.env`:
- `OLLAMA_API_KEY`: API key for Ollama service (if applicable).
- `GOOGLE_API_KEY`: API key for Google Gemini (required for Google Provider).
- Additional variables can be added to the `.env` file and will be automatically loaded.

### Memory and Sessions

Kader stores data in `~/.kader/`:
- Sessions: `~/.kader/memory/sessions/`
- Configuration: `~/.kader/`
- Memory files: `~/.kader/memory/`
- Checkpoints: `~/.kader/memory/sessions/<session-id>/executors/` (Aggregated context from sub-agents)

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command reference |
| `/models` | Show available models (Ollama & Google) |
| `/clear` | Clear conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/refresh` | Refresh file tree |
| `/exit` | Exit the CLI |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear conversation |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Refresh file tree |
| `Tab` | Navigate panels |

## Project Structure

```
kader/
├── cli/                    # Interactive command-line interface
│   ├── app.py             # Main application entry point
│   ├── app.tcss           # Textual CSS for styling
│   ├── llm_factory.py     # Provider selection logic
│   ├── widgets/           # Custom Textual widgets
│   │   ├── conversation.py # Chat display widget
│   │   ├── loading.py     # Loading spinner widget
│   │   ├── confirmation.py # Tool/model selection widgets
│   │   └── todo_list.py    # TODO tracking widget
│   └── README.md          # CLI documentation
├── examples/              # Example implementations
│   ├── memory_example.py  # Memory management examples
│   ├── google_example.py  # Google Gemini provider examples
│   ├── planner_executor_example.py # Advanced workflow examples
│   ├── skills/           # Agent skills examples
│   │   ├── hello/        # Greeting skill with instructions
│   │   ├── calculator/   # Math calculation skill
│   │   └── react_agent.py # Skills demo with ReAct agent
│   └── README.md         # Examples documentation
├── kader/                # Core framework
│   ├── agent/            # Agent implementations (Planning, ReAct)
│   ├── memory/           # Memory management & persistence
│   ├── providers/        # LLM providers (Ollama, Google)
│   ├── tools/            # Tools (File System, Web, Command, AgentTool)
│   ├── prompts/          # Prompt templates (Jinja2)
│   └── utils/            # Utilities (Checkpointer, ContextAggregator)
├── pyproject.toml        # Project dependencies
├── README.md             # This file
└── uv.lock               # Dependency lock file
```

## Core Components

### Agents

Kader provides a robust agent architecture:

- **ReActAgent**: Reasoning and Acting agent that combines thoughts with actions.
- **PlanningAgent**: High-level agent that breaks complex tasks into manageable plans.
- **BaseAgent**: Abstract base class for creating custom agent behaviors.

### LLM Providers

Kader supports multiple backends:
- **OllamaProvider**: Connects to locally running Ollama instances.
- **GoogleProvider**: High-performance access to Gemini models.

### Agent-As-Tool (AgentTool)

The `AgentTool` allows a `PlanningAgent` (Architect) to delegate work to a `ReActAgent` (Worker). It features:
- **Persistent Memory**: Sub-agent conversations are saved to JSON.
- **Context Aggregation**: Sub-agent research and actions are automatically merged into the main session's `checkpoint.md` via `ContextAggregator`.

### Agent Skills

Kader supports a modular skill system for domain-specific knowledge and specialized instructions:

- **Skill Structure**: Skills are defined as directories containing `SKILL.md` files with YAML frontmatter
- **Skill Loading**: Skills are loaded from `~/.kader/skills` (high priority) and `./.kader/` directories
- **Skill Injection**: Available skills are automatically injected into the system prompt
- **Skills Tool**: Agents can load skills dynamically using the `skills_tool`

### File System Tools with Gitignore Filtering

The file system tools (`read_directory`, `grep`, `glob`) automatically filter out files and directories that match patterns defined in `.gitignore` files.

You can disable this filtering by passing `apply_gitignore_filter=False` when creating tools:

```python
from pathlib import Path
from kader.tools.filesys import get_filesystem_tools

# With filtering (default)
tools = get_filesystem_tools(base_path=Path.cwd())

# Without filtering
tools = get_filesystem_tools(base_path=Path.cwd(), apply_gitignore_filter=False)
```

Example skill structure:
```
~/.kader/skills/hello/
├── SKILL.md
└── scripts/
    └── hello.py
```

Example skill (`SKILL.md`):
```yaml
---
name: hello
description: Skill for ALL greeting requests
---

# Hello Skill

This skill provides the greeting format you must follow.

## How to greet

Always greet the user with:
- A warm welcome
- Their name if mentioned
- A friendly emoji
```

### Memory Management

- **SlidingWindowConversationManager**: Maintains context within token limits.
- **PersistentSlidingWindowConversationManager**: Auto-saves sub-agent history.
- **Checkpointer**: Generates markdown summaries of agent actions.

## Development

### Setting up for Development

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install in development mode with uv
uv sync

# Run the CLI with hot reload for development
uv run textual run --dev cli.app:KaderApp
```

### Running Tests

```bash
# Run tests with uv
uv run pytest

# Run tests with specific options
uv run pytest --verbose
```

### Code Quality

Kader uses various tools for maintaining code quality:

```bash
# Run linter
uv run ruff check .

# Format code
uv run ruff format .
```

## Troubleshooting

### Common Issues

- **No models found**: Ensure your providers are correctly configured. For Ollama, run `ollama serve`. For Google, ensure `GOOGLE_API_KEY` is set.
- **Connection errors**: Verify internet access for cloud providers and local service availability for Ollama.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines on:

- Setting up your development environment
- Code style guidelines
- Running tests
- Submitting pull requests

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/your-username/kader.git
cd kader

# Install dependencies
uv sync

# Run tests
uv run pytest

# Run linter
uv run ruff check .
```

### Coding with AI

This project includes a specialized skill for AI coding agents. When working with AI assistants on this codebase, they should use the `contributing-to-kader` skill located in [`.kader/skills/contributing-to-kader`](.kader/skills/contributing-to-kader). This skill provides AI agents with essential guidelines including:

- Core development rules (linting, formatting, testing)
- Key commands for development workflow
- Project structure overview
- Best practices for contributing

AI assistants can load this skill using the skills_tool to get specialized instructions for working with this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Textual](https://textual.textualize.io/) for the beautiful CLI interface.
- Uses [Ollama](https://ollama.ai/) for local LLM execution.
- Powered by [Google Gemini](https://ai.google.dev/) for advanced cloud-based reasoning.
