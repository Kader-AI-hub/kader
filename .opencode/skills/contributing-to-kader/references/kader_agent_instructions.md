# Kader Agent Instructions

This document provides guidelines for AI coding agents working in the Kader repository.

## Build/Lint/Test Commands

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

# Lint code
uv run ruff check .

# Fix linting issues automatically
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without modifying
uv run ruff format --check .

# Run the CLI
uv run python -m cli

# Run CLI with hot reload (for development)
uv run textual run --dev cli.app:KaderApp
```

## Code Style Guidelines

### Python Version
- Use Python 3.11+ features
- Target version: py311 (configured in pyproject.toml)

### Formatting
- Line length: 88 characters (Black-compatible)
- Use double quotes for strings
- Use trailing commas
- Use explicit string concatenation

### Naming Conventions
- Variables/functions: snake_case
- Classes: PascalCase
- Constants: SCREAMING_SNAKE_CASE
- Private methods: _leading_underscore

### Type Hints
- Always use type hints for function signatures
- Use `typing` module for complex types
- Prefer `list`, `dict` over `List`, `Dict` (Python 3.9+)

### Import Organization
Order imports as:
1. Standard library
2. Third-party packages
3. Local application imports

Use absolute imports within the project.

## Testing Requirements

### Test File Structure
- Place tests in `tests/` directory mirroring source structure
- Name test files as `test_<module>.py`
- Name test functions as `test_<functionality>`

### Mocking Guidelines
- Mock ALL filesystem operations (use `pytest.fixture` with `tmp_path`)
- Mock network calls
- Mock external API calls
- DO NOT create real files or make real network requests in tests

### Fixtures
- Use `conftest.py` for shared fixtures
- Use `tmp_path` fixture for file operations
- Mock LLM providers in tests

## Project Structure

```
kader/
├── kader/                # Core framework (agents, providers, tools, memory)
│   ├── agent/            # Agent implementations (Planning, ReAct)
│   ├── memory/           # Memory management & persistence
│   ├── providers/        # LLM providers (Ollama, Google, Mistral)
│   ├── tools/            # Tools (File System, Web, Command, AgentTool)
│   ├── prompts/          # Prompt templates
│   └── utils/            # Utilities (Checkpointer, ContextAggregator)
├── cli/                  # Interactive CLI implementation
│   ├── app.py            # Main application entry point
│   ├── widgets/          # Custom Textual widgets
│   └── commands/         # CLI commands
├── tests/                # Test files mirroring source structure
│   ├── providers/
│   ├── tools/
│   └── conftest.py       # Shared fixtures
├── examples/             # Example usage scripts
├── pyproject.toml        # Project dependencies and configuration
└── uv.lock              # Dependency lock file
```

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

### Special Files
- `.kader/` directory in home: Stores config, sessions, memory
- `~/.kader/.env`: API keys and environment variables
- `app.tcss`: Textual CSS for CLI styling
- `uv.lock`: Dependency lock file (do not edit manually)

## Important Notes

- When modifying code, always run `uv run ruff check . --fix` before committing
- Always mock filesystem operations in tests to avoid modifying real files
- The project uses `uv` for dependency management, not pip
- LLM providers require API keys in `~/.kader/.env`
