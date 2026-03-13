# Tools

Tools extend agent capabilities beyond text generation. They can read files, execute commands, manage tasks, and more.

## Built-in Tools

Kader provides several built-in tools:

```python
from kader.tools import (
    ReadFileTool,
    WriteFileTool,
    GlobTool,
    GrepTool,
    CommandExecutorTool,
    AgentTool,
    TodoTool,
    SkillsTool,
)
```

## File System Tools

### ReadFileTool

Read contents of a file:

```python
from kader.tools import ReadFileTool

read_tool = ReadFileTool()
result = read_tool.execute(path="README.md")
print(result.content)
```

### WriteFileTool

Write or edit files:

```python
from kader.tools import WriteFileTool

write_tool = WriteFileTool()
result = write_tool.execute(
    content="# Hello World\n",
    path="new_file.md"
)
```

### GlobTool

Find files by pattern:

```python
from kader.tools import GlobTool

glob_tool = GlobTool()
result = glob_tool.execute(pattern="**/*.py")
print(result.matches)
```

### GrepTool

Search file contents:

```python
from kader.tools import GrepTool

grep_tool = GrepTool()
result = grep_tool.execute(pattern="def.*main", path=".")
print(result.matches)
```

## Command Execution

### CommandExecutorTool

Execute shell commands:

```python
from kader.tools import CommandExecutorTool

cmd_tool = CommandExecutorTool()

result = cmd_tool.execute(command="ls -la")
result = cmd_tool.execute(command="python -c 'print(1 + 1)'")
```

## Todo Tool

Manage task lists:

```python
from kader.tools import TodoTool

todo = TodoTool()

# Create a todo list
todo.execute(
    action="create",
    todo_id="project-tasks",
    items=[
        {"id": "1", "content": "Setup project", "status": "completed"},
        {"id": "2", "content": "Write tests", "status": "pending"},
        {"id": "3", "content": "Deploy", "status": "pending"},
    ]
)

# Update todo status
todo.execute(
    action="update",
    todo_id="project-tasks",
    item_id="2",
    status="completed"
)

# Read todo list
result = todo.execute(action="read", todo_id="project-tasks")
```

## Agent Tool (Sub-agents)

Spawn sub-agents with isolated memory:

```python
from kader.tools import AgentTool
from kader.agent import BaseAgent
from kader.providers import OllamaProvider

agent_tool = AgentTool()

parent_agent = BaseAgent(
    name="ParentAgent",
    tools=[agent_tool],
    provider=OllamaProvider(model="llama3.2"),
)

response = parent_agent.invoke(
    "Use the agent tool to research Python async patterns"
)
```

## Skills Tool

Load specialized instructions:

```python
from kader.tools import SkillsTool

skills_tool = SkillsTool()

# Or with custom directories
from pathlib import Path
skills_tool = SkillsTool(
    skills_dirs=[Path("./custom_skills")],
    priority_dir=Path("./project_skills"),
)

result = skills_tool.execute(name="python-expert")
```

### Skill File Format

```yaml
---
name: python-expert
description: Expert in Python programming and best practices
---

# Python Expert Skill

You are an expert Python developer with deep knowledge of:
- PEP 8 style guide
- Type hints and annotations
- Async/await patterns

When writing Python code:
1. Use type hints whenever possible
2. Follow PEP 8 conventions
```

## Special Commands

Special commands allow you to create custom command agents that can be invoked from the CLI. Unlike skills which are loaded by agents, special commands are executed directly from the CLI using `/<command-name>`.

```python
from kader.tools import CommandLoader

# Create command loader
loader = CommandLoader()

# Load a specific command
command = loader.load_command("lint-test")
print(command.name)        # Command name
print(command.description)  # Command description
print(command.content)     # Command instructions
print(command.base_dir)    # Command directory path

# List all available commands
all_commands = loader.list_commands()

# Get formatted description
description = loader.get_description()
```

### Command File Format

Commands can be defined in three formats:

**Option 1: Directory format** (with additional files)
```
~/.kader/commands/lint-test/
├── CONTENT.md          # Main command instructions
├── templates/          # Optional - templates
└── assets/            # Optional - files
```

**Option 2: Simple file format**
```
~/.kader/commands/lint-test.md
```

**Option 3: Directory with sub-commands**
```
~/.kader/commands/mycommand/
├── CONTENT.md           # Main command (/mycommand)
├── subcommand1.md      # Sub-command (/mycommand/subcommand1)
├── subcommand2.md      # Sub-command (/mycommand/subcommand2)
├── templates/           # Optional - shared templates
└── assets/             # Optional - shared assets
```

All formats use the same CONTENT.md or .md file format:

```yaml
---
description: Lint and test the codebase
---

# Lint and Test Agent

You are specialized in maintaining code quality.

## Instructions

1. Run linting: uv run ruff check .
2. Run tests: uv run pytest -v
3. Report results
```

Commands are loaded from:
- `./.kader/commands/` (project-level, higher priority)
- `~/.kader/commands/` (user-level)

### Using Commands from CLI

```
/lint-test
/lint-test run full check
/lint-test/lint    # Sub-command
/commands  # List all available commands
```

## Custom Tools

Create your own tools by subclassing `BaseTool`:

```python
from kader.tools import BaseTool, ParameterSchema, ToolResult

class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="weather",
            description="Get weather information for a city",
            parameters=[
                ParameterSchema(
                    name="city",
                    type="string",
                    description="City name",
                    required=True,
                ),
                ParameterSchema(
                    name="units",
                    type="string",
                    description="Temperature units",
                    required=False,
                ),
            ],
        )

    def execute(self, city: str, units: str = "celsius") -> ToolResult:
        weather_data = get_weather(city, units)
        return ToolResult(
            status="success",
            content=f"Weather in {city}: {weather_data}",
        )

weather_tool = WeatherTool()
agent = BaseAgent(
    name="WeatherBot",
    tools=[weather_tool],
    provider=OllamaProvider(model="llama3.2"),
)
```

## Gitignore Filtering

File system tools automatically filter files matching `.gitignore` patterns:

```python
from pathlib import Path
from kader.tools.filesys import get_filesystem_tools

# With filtering (default)
tools = get_filesystem_tools(base_path=Path.cwd())

# Without filtering
tools = get_filesystem_tools(base_path=Path.cwd(), apply_gitignore_filter=False)
```
