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
- Use trailing commas in multi-line collections

### Imports
- Group imports in this order:
  1. Standard library imports
  2. Third-party imports
  3. Local imports (from kader.*)
  4. Relative imports (for same package)
- Use absolute imports for cross-package references
- Use relative imports within the same package
- Use `from __future__ import annotations` where needed for forward references

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

### Testing
- Use pytest with asyncio support
- Use fixtures in conftest.py for shared setup
- Mock external services (LLM providers, filesystem)
- Tests mock `Path.home()` to avoid touching real ~/.kader directory
- Always clean up resources (temp files, logger handlers)

### Project Structure
- `kader/`: Core framework (agents, providers, tools, memory)
- `cli/`: Interactive CLI implementation
- `tests/`: Test files mirroring source structure
- `examples/`: Example usage scripts

### Ruff Configuration
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
ignore = ["E501"]
```

### Special Files
- `.kader/` directory in home: Stores config, sessions, memory
- `~/.kader/.env`: API keys and environment variables
- `app.tcss`: Textual CSS for CLI styling
- `uv.lock`: Dependency lock file (do not edit manually)
