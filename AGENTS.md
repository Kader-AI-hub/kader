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

[tool.ruff.lint.per-file-ignores]
"examples/*" = ["E402", "F841", "F402"]
"tests/*" = ["E402"]
```

### Pytest Configuration
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

### Additional Guidelines

- When adding new dependencies, add them to the `dependencies` list in `pyproject.toml`
- Dev dependencies (pytest, ruff) go in `[dependency-groups].dev`
- Use `uv sync` to install dependencies after any changes to pyproject.toml
- When running tests, use `-v` for verbose output to debug failures
- Test file naming: `test_<module_name>.py` (e.g., `test_base_agent.py` for `kader/base_agent.py`)
- Mock `Path.home()` in tests to avoid touching real ~/.kader directory
- Clean up logger handlers in tests to prevent duplicate logs

### Special Files
- `.kader/` directory in home: Stores config, sessions, memory
- `~/.kader/.env`: API keys and environment variables
- Rich library used for CLI terminal output formatting
- `uv.lock`: Dependency lock file (do not edit manually)

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
