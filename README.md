# Kader

Kader is an intelligent coding agent designed to assist with software development tasks. It provides a comprehensive framework for building AI-powered agents with advanced reasoning capabilities and tool integration.

## Features

- 🤖 **AI-powered Code Assistance** - Using Ollama for local LLM execution
- 🖥️ **Interactive CLI** - Modern TUI interface built with Textual
- 🛠️ **Tool Integration** - File system, command execution, web search, and more
- 🧠 **Memory Management** - State persistence and conversation history
- 🔁 **Session Management** - Save and load conversation sessions
- 🎨 **Theming** - Multiple color themes for the CLI interface
- ⌨️ **Keyboard Shortcuts** - Efficient navigation and operations
- 📝 **YAML Configuration** - Agent configuration via YAML files
- 🔄 **ReAct Agent Framework** - Reasoning and Acting agent architecture
- 🗂️ **File System Tools** - Read, write, search, and edit files
- 🔍 **Planning Agent** - Task planning and execution capabilities

## Installation

### Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) running locally to use LLMs
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

1. Type any question to start chatting with the agent
2. Use `/help` to see available commands
3. Use `/models` to check available models
4. Use `/theme` to cycle through color themes

## Configuration

When the kader module is imported for the first time, it automatically:
1. Creates a `.kader` directory in your home directory (`~/.kader` on Unix systems, `%USERPROFILE%\.kader` on Windows)
2. Creates a `.env` file with the required configuration (including `OLLAMA_API_KEY=''`)
3. Loads all environment variables from the `.env` file into the application environment

### Environment Variables

The application automatically loads environment variables from `~/.kader/.env`:
- `OLLAMA_API_KEY`: API key for Ollama service (default: empty)
- Additional variables can be added to the `.env` file and will be automatically loaded

### Memory and Sessions

Kader stores data in `~/.kader/`:
- Sessions: `~/.kader/sessions/`
- Configuration: `~/.kader/`
- Memory files: `~/.kader/memory/`

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command reference |
| `/models` | Show available Ollama models |
| `/theme` | Cycle color themes |
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
| `Ctrl+T` | Cycle theme |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Refresh file tree |
| `Tab` | Navigate panels |

## Project Structure

```
kader/
├── cli/                    # Interactive command-line interface
│   ├── app.py             # Main application entry point
│   ├── app.tcss           # Textual CSS for styling
│   ├── utils.py           # Utility functions and constants
│   ├── widgets/           # Custom Textual widgets
│   │   ├── conversation.py # Chat display widget
│   │   ├── loading.py     # Loading spinner widget
│   │   └── confirmation.py # Tool/model selection widgets
│   └── README.md          # CLI documentation
├── examples/              # Example implementations
│   ├── memory_example.py  # Memory management examples
│   ├── ollama_example.py  # Ollama provider examples
│   ├── react_agent_example.py # ReAct agent examples
│   ├── planning_agent_example.py # Planning agent examples
│   ├── python_developer/  # Python expert agent example
│   ├── todo_agent/       # Todo management agent example
│   └── README.md         # Examples documentation
├── kader/                # Core framework
│   ├── agent/            # Agent implementations
│   ├── memory/           # Memory management
│   ├── providers/        # LLM providers
│   ├── tools/            # Tools and utilities
│   └── prompts/          # Prompt templates
├── pyproject.toml        # Project dependencies
├── README.md             # This file
└── uv.lock               # Dependency lock file
```

## Core Components

### Agents

Kader provides several agent types:

- **ReActAgent**: Reasoning and Acting agent that combines thoughts with actions
- **PlanningAgent**: Agent that plans multi-step tasks
- **BaseAgent**: Base agent class for creating custom agents

### Memory Management

Kader's memory system includes:

- **AgentState**: Persistent key-value storage for agents
- **RequestState**: Ephemeral request-scoped state
- **FileSessionManager**: Session persistence to disk
- **SlidingWindowConversationManager**: Conversation windowing

### Tools

Kader includes a rich set of tools:

- **File System Tools**: Read, write, edit, search files
- **Command Executor**: Execute shell commands safely
- **Web Tools**: Search and fetch web content
- **RAG Tools**: Retrieval Augmented Generation capabilities

## Examples

Check out the examples directory for comprehensive demonstrations of Kader's features:

### Basic Examples

- `memory_example.py`: Shows memory management capabilities
- `ollama_example.py`: Demonstrates how to use the Ollama provider for LLM interactions
- `tools_example.py`: Demonstrates the various tools available in Kader
- `simple_agent.py`: Basic agent implementation example

### Advanced Examples

- `react_agent_example.py`: Interactive ReAct agent with tool integration
- `planning_agent_example.py`: Planning agent for multi-step tasks
- `python_developer/`: Specialized Python expert agent (YAML-configured)
- `todo_agent/`: Task management agent with TodoTool

### Running Examples

Use uv to run examples:

```bash
uv run python -m examples.memory_example
uv run python -m examples.ollama_example
uv run python -m examples.tools_example
uv run python -m examples.react_agent_example
uv run python -m examples.python_developer.main
uv run python -m examples.todo_agent.main
```

## Architecture

### Agent Architecture

Kader uses a modular architecture where:

1. **Agents** define the behavior and reasoning strategy
2. **Tools** provide capabilities for external interactions
3. **Memory** manages state and conversation history
4. **Providers** handle LLM interactions

### Tool Architecture

Tools in Kader follow a standardized interface:

- Each tool has a schema defining its inputs and outputs
- Tools can be registered in a ToolRegistry
- Tools can be executed synchronously or asynchronously
- Tools can be configured with parameters

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

- **No models found**: Make sure Ollama is running and you have at least one model installed (e.g., `ollama pull gpt-oss:120b-cloud`)
- **Connection errors**: Verify that Ollama service is accessible at the configured endpoint
- **Theme not changing**: Some terminal emulators may not support all color themes

### Debugging

If you encounter issues:

1. Check that Ollama is running: `ollama serve`
2. Verify your model is pulled: `ollama list`
3. Ensure your terminal supports the required features
4. Check the logs for specific error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Textual](https://textual.textualize.io/) for the beautiful CLI interface
- Uses [Ollama](https://ollama.ai/) for local LLM execution
- Inspired by ReAct (Reasoning and Acting) agent architecture