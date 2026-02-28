---
name: contributing-to-kader
description: This skill provides AI coding agents with the guidelines and reference material needed to contribute effectively to the Kader repository. Use this skill when writing code, fixing bugs, adding features, or making any modifications to the Kader project.
---

# Contributing To Kader

## Overview

This skill enables AI coding agents to work effectively in the Kader repository by providing essential guidelines, commands, and project structure information. It should be loaded whenever the agent needs to make code contributions, whether bug fixes, features, or documentation improvements.

## Core Guidelines

AI agents working in the Kader repository must follow these fundamental rules:

1. **Always run linting and formatting before committing** - Use `uv run ruff check . --fix` and `uv run ruff format .`
2. **Mock filesystem operations in tests** - Never modify real files during test execution
3. **Use `uv` for dependency management** - Not pip; all commands should use `uv run` or `uv sync`
4. **Target Python 3.11+** - Use modern Python features compatible with py311

## Build/Lint/Test Commands

### Installation and Running
```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_base_agent.py -v

# Run a specific test function
uv run pytest tests/test_base_agent.py::test_agent_mock_invocation -v

# Run tests with verbose output
uv run pytest -v

# Run the CLI
uv run python -m cli
```

### Linting and Formatting
```bash
# Lint code
uv run ruff check .

# Fix linting issues automatically
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without modifying
uv run ruff format --check .
```

## Code Style Guidelines

### Python Version
- Use Python 3.11+ features
- Target version: py311 (configured in pyproject.toml)

### Formatting
- Line length: 88 characters (Black-compatible)
- Use double quotes for strings
- Use Ruff for all linting and formatting

### Ruff Configuration
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"examples/*" = ["E402", "F841", "F402"]
"tests/*" = ["E402"]
```

### Naming Conventions
- Use `snake_case` for functions, methods, variables
- Use `SCREAMING_SNAKE_CASE` for constants
- Use `PascalCase` for classes and exceptions
- Use `kebab-case` for CLI commands

### Type Hints
- Use type hints for all function signatures
- Prefer `X | None` over `Optional[X]`
- Prefer `list[X]` over `List[X]`

## Project Structure

```
kader/
├── kader/                # Core framework (agents, providers, tools, memory)
│   ├── agent/            # Agent implementations (Planning, ReAct)
│   ├── memory/           # Memory management & persistence
│   ├── providers/       # LLM providers (Ollama, Google, Mistral, Anthropic)
│   ├── tools/            # Tools (File System, Web, Command, AgentTool)
│   ├── prompts/          # Prompt templates
│   ├── utils/            # Utilities (Checkpointer, ContextAggregator)
│   └── workflows/        # Workflow executors (PlannerExecutorWorkflow)
├── cli/                  # Interactive CLI implementation
│   ├── app.py            # Main application (Rich + prompt_toolkit)
│   ├── commands/        # CLI commands (initialize, base)
│   ├── llm_factory.py    # LLM provider factory
│   └── utils.py          # CLI utilities and constants
├── tests/                # Test files mirroring source structure
│   ├── providers/
│   ├── tools/
│   └── conftest.py       # Shared fixtures
├── examples/             # Example usage scripts
├── pyproject.toml        # Project dependencies and configuration
└── uv.lock              # Dependency lock file
```

## CLI Commands

The Kader CLI supports these commands:

| Command | Description |
|---------|-------------|
| `/models` | Show and switch LLM models |
| `/help` | Show help message |
| `/clear` | Clear the conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/cost` | Show usage costs |
| `/init` | Initialize .kader directory |
| `/exit` | Exit the CLI |
| `!cmd` | Run terminal command |

## LLM Providers

The CLI supports multiple LLM providers via the factory pattern:

- **ollama**: Local models via Ollama (default)
- **google**: Google Gemini models
- **mistral**: Mistral AI models
- **anthropic**: Anthropic Claude models
- **openai**: OpenAI models (GPT-4, GPT-4o)
- **moonshot**: Moonshot AI models (kimi)
- **zai**: Z.ai models (GLM)
- **openrouter**: OpenRouter models (200+ models)
- **opencode**: OpenCode Zen models
- **groq**: Groq models (ultra-fast inference)

Model format: `provider:model` (e.g., `google:gemini-2.5-flash`, `ollama:llama3`)

## Important Notes

- **Never edit `uv.lock` manually** - This is an auto-generated dependency lock file
- **Store API keys in `~/.kader/.env`** - LLM providers require API keys in this location
- **Configuration directory**: `.kader/` in home stores config, sessions, and memory
- **CLI uses Rich + prompt_toolkit** - The CLI renders beautiful terminal output using Rich library

## Testing Requirements

- All new functionality must include tests
- Tests should be placed in the `tests/` directory, mirroring the source structure
- Use pytest as the testing framework
- Mock filesystem operations to avoid modifying real files during tests
- Run `uv run pytest` before committing to ensure all tests pass

## Pull Request Process

1. **Fork and Branch**: Fork the repository and create a feature branch from `main`
2. **Make Changes**: Implement changes following code style guidelines
3. **Add Tests**: Write tests for new functionality or bug fixes
4. **Run Tests**: Ensure all tests pass locally
5. **Lint and Format**: Run Ruff to check and fix code style
6. **Commit**: Write clear, descriptive commit messages
7. **Push**: Push your branch to your fork
8. **Open PR**: Open a pull request with a clear description of the changes

### PR Checklist
- [ ] Code follows the style guidelines
- [ ] Tests pass locally (`uv run pytest`)
- [ ] Linting passes (`uv run ruff check .`)
- [ ] Formatting passes (`uv run ruff format --check .`)
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

## Resources

This skill includes additional reference materials in the following directories:

### scripts/
Contains example scripts demonstrating Kader functionality. These can be executed directly or read for reference when implementing similar features.

### references/
Contains detailed API reference documentation and supplementary materials that provide deeper context about Kader's architecture and capabilities.

### assets/
Contains template files and other resources that may be useful when creating new components or extending Kader's functionality.

---

**Note**: This skill should be loaded whenever an AI agent needs to work on the Kader codebase. The guidelines and commands in this skill are essential for maintaining code quality and consistency across the project.
