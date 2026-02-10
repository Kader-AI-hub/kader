# Contributing to Kader

Thank you for your interest in contributing to Kader! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Ways to Contribute

- **Bug Reports**: Open an issue with a clear description and reproduction steps.
- **Feature Requests**: Open an issue describing the feature and its use case.
- **Code Contributions**: Fix bugs, add features, or improve documentation.
- **Documentation**: Improve README, docstrings, or add examples.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup Steps

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Install dependencies
uv sync

# Run the CLI to verify setup
uv run python -m cli
```

## Code Style Guidelines

### Python Version

- Use Python 3.11+ features
- Target version: py311

### Formatting

- Line length: 88 characters (Black-compatible)
- Use double quotes for strings
- Use trailing commas in multi-line collections

### Imports

Group imports in this order:

1. Standard library imports
2. Third-party imports
3. Local imports (`from kader.*`)
4. Relative imports (for same package)

Use absolute imports for cross-package references and relative imports within the same package.

Example:

```python
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Optional, Union

import yaml
from tenacity import RetryError, stop_after_attempt, wait_exponential

from kader.memory import ConversationManager
from kader.providers.base import BaseLLMProvider
from kader.tools import BaseTool, ToolRegistry

from .logger import agent_logger
```

### Type Hints

- Use type hints for all function parameters and return types
- Use `|` syntax for unions (Python 3.11+)
- Use `list[...]`, `dict[...]` instead of `typing.List`, `typing.Dict`
- Use `Optional[...]` or `... | None` for nullable types

### Naming Conventions

- Classes: PascalCase (e.g., `BaseAgent`, `OllamaProvider`)
- Functions/Methods: snake_case (e.g., `invoke`, `get_next_response`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`, `MIN_WIDTH`)
- Private methods/attributes: `_leading_underscore`
- Module names: snake_case (e.g., `base.py`, `ollama.py`)

### Error Handling

- Use specific exceptions, not bare `except:`
- Use `try/except/finally` for resource cleanup
- Log errors with context using loguru
- Use tenacity for retry logic with exponential backoff

Example:

```python
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def fetch_data():
    try:
        return api_call()
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        raise
```

### Documentation

- Use docstrings for modules, classes, and functions
- Follow Google-style docstrings
- Include type information in docstrings only when it adds clarity

Example:

```python
def initialize_kader_config():
    """
    Initialize the .kader directory in user's home with required configuration.

    Creates the directory, sets up the .env file, and loads environment variables.

    Returns:
        Tuple of (kader_dir, success) where success is a boolean indicating
        whether initialization succeeded.
    """
```

### Async Patterns

- Use `async def` for I/O-bound operations
- Provide both sync and async versions when appropriate (`method` and `amethod`)
- Use `asyncio` for concurrency
- Use `AsyncIterator` for streaming responses

### Ruff Configuration

The project uses Ruff for linting and formatting:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_base_agent.py -v

# Run a specific test function
uv run pytest tests/test_base_agent.py::test_agent_mock_invocation -v

# Run tests with verbose output
uv run pytest -v
```

### Testing Guidelines

- Use pytest with asyncio support
- Use fixtures in conftest.py for shared setup
- Mock external services (LLM providers, filesystem)
- Tests mock `Path.home()` to avoid touching real ~/.kader directory
- Always clean up resources (temp files, logger handlers)

### Linting and Formatting

Before submitting a pull request, ensure your code passes linting and formatting checks:

```bash
# Run linter
uv run ruff check .

# Fix linting issues automatically
uv run ruff check . --fix

# Format code
uv run ruff format .

# Check formatting without modifying
uv run ruff format --check .
```

## Pull Request Process

1. **Fork and Branch**: Fork the repository and create a feature branch from `main`.
2. **Make Changes**: Implement your changes following the code style guidelines.
3. **Add Tests**: Write tests for new functionality or bug fixes.
4. **Run Tests**: Ensure all tests pass locally.
5. **Lint and Format**: Run Ruff to check and fix code style.
6. **Commit**: Write clear, descriptive commit messages.
7. **Push**: Push your branch to your fork.
8. **Open PR**: Open a pull request with a clear description of the changes.

### PR Checklist

- [ ] Code follows the style guidelines
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Documentation updated if needed
- [ ] Commit messages are clear and descriptive

## Project Structure

```
kader/
├── cli/                    # Interactive command-line interface
├── examples/              # Example implementations
├── kader/                # Core framework
│   ├── agent/            # Agent implementations
│   ├── memory/           # Memory management
│   ├── providers/        # LLM providers
│   ├── tools/            # Tools
│   ├── prompts/          # Prompt templates
│   └── utils/            # Utilities
├── tests/                # Test files
├── pyproject.toml        # Project dependencies
└── uv.lock              # Dependency lock file
```

## Questions?

If you have questions about contributing, feel free to open an issue for discussion.

Thank you for contributing to Kader!
