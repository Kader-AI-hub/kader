# Kader Core Framework

A comprehensive Python framework for building intelligent AI coding agents with tool integration, memory management, and multi-provider LLM support.

## Overview

Kader is a flexible framework that enables developers to create AI agents capable of reasoning, planning, and executing tasks using various Large Language Models (LLMs). It provides a unified API for working with different LLM providers while maintaining a consistent interface for tools and memory management.

### Key Features

- **Multi-Provider Support**: Connect to Ollama, Google Gemini, Anthropic, Mistral, OpenAI, and any OpenAI-compatible API
- **Agent Types**: BaseAgent, ReActAgent, and PlanningAgent for different reasoning patterns
- **Tool System**: Build custom tools with automatic schema generation for any LLM provider
- **Memory Management**: Persistent sessions, conversation history with sliding windows, and state management
- **Workflows**: Planner-Executor pattern for complex multi-step tasks
- **Session Persistence**: Save and resume agent conversations

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip

### Install Kader

```bash
# Clone the repository
git clone https://github.com/your-repo/kader.git
cd kader

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Provider-Specific Setup

For cloud providers, create a `.env` file in `~/.kader/.env`:

```bash
# Google Gemini
GEMINI_API_KEY='your-api-key'

# Mistral
MISTRAL_API_KEY='your-api-key'

# Anthropic
ANTHROPIC_API_KEY='your-api-key'

# OpenAI
OPENAI_API_KEY='your-api-key'

# OpenAI-Compatible Providers
MOONSHOT_API_KEY='your-api-key'
GROQ_API_KEY='your-api-key'
OPENROUTER_API_KEY='your-api-key'
```

For local LLM inference with Ollama, install from https://ollama.ai and pull your desired model:

```bash
ollama pull llama3.2
ollama pull qwen2.5
```

## Core Concepts

### 1. Providers

Providers are the bridge between Kader and LLM services. Each provider implements a common interface:

```python
from kader.providers import Message, OllamaProvider

provider = OllamaProvider(model="llama3.2")
response = provider.invoke([Message.user("Hello!")])
print(response.content)
```

### 2. Tools

Tools extend agent capabilities beyond text generation. They can read files, execute commands, manage tasks, and more:

```python
from kader.tools import ReadFileTool, WriteFileTool, CommandExecutorTool

tools = [
    ReadFileTool(),
    WriteFileTool(),
    CommandExecutorTool(),
]
```

### 3. Agents

Agents combine providers, tools, and memory to accomplish tasks:

```python
from kader.agent import BaseAgent
from kader.providers import OllamaProvider
from kader.tools import ReadFileTool

agent = BaseAgent(
    name="Assistant",
    system_prompt="You are a helpful coding assistant.",
    tools=[ReadFileTool()],
    provider=OllamaProvider(model="llama3.2"),
)
```

### 4. Memory

Memory systems manage conversation history and agent state:

```python
from kader.memory import SlidingWindowConversationManager, FileSessionManager

# Session persistence
session_mgr = FileSessionManager()
session = session_mgr.create_session("my-agent")

# Conversation windowing
conv_mgr = SlidingWindowConversationManager(window_size=20)
conv_mgr.add_message(Message.user("Hello"))
```

## Quick Start

### Basic Agent Setup

```python
import sys
from pathlib import Path

# Add kader to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kader.agent import BaseAgent
from kader.providers import OllamaProvider
from kader.tools import ReadFileTool, WriteFileTool

# 1. Initialize provider
provider = OllamaProvider(model="llama3.2")

# 2. Create agent with tools
agent = BaseAgent(
    name="CodingAssistant",
    system_prompt="You are a helpful coding assistant. Use tools when needed.",
    tools=[ReadFileTool(), WriteFileTool()],
    provider=provider,
)

# 3. Invoke the agent
response = agent.invoke("Read the file README.md and summarize it")
print(response.content)
```

### Full Example with Todo Tool

```python
from kader.agent import BaseAgent
from kader.tools import TodoTool
from kader.providers import OllamaProvider

# Initialize agent with TodoTool
agent = BaseAgent(
    name="PlannerBot",
    system_prompt=(
        "You are a helpful planning assistant. "
        "Use the 'todo_tool' to manage tasks. "
        "Always check the todo list before adding new items."
    ),
    tools=[TodoTool()],
    model_name="llama3.2",
    use_persistence=True,
)

# Create and manage tasks
response = agent.invoke(
    "Create a todo list with ID 'my-tasks' containing: "
    "1. Learn Python, 2. Build an agent, 3. Deploy to production"
)
print(response.content)
```

## Agent Types

### BaseAgent

The foundation class for all agents. Provides core functionality for tool execution, conversation management, and session persistence.

```python
from kader.agent import BaseAgent
from kader.providers import OllamaProvider, Message

# Create a basic agent
agent = BaseAgent(
    name="MyAgent",
    system_prompt="You are a helpful assistant.",
    tools=[],  # Add your tools here
    provider=OllamaProvider(model="llama3.2"),
    use_persistence=True,  # Enable session saving
    interrupt_before_tool=True,  # Confirm before tool execution
)

# Synchronous invocation
response = agent.invoke("What is Python?")
print(response.content)

# Streaming invocation
for chunk in agent.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Asynchronous invocation
import asyncio
response = await agent.ainvoke("What is asyncio?")
```

#### YAML Configuration

BaseAgent supports loading configuration from YAML files:

```yaml
# agent.yaml
name: ConfiguredAgent
system_prompt: "You are configured via YAML."
tools:
  - read_file
  - write_file
provider:
  model: llama3.2
persistence: true
retry_attempts: 3
```

```python
agent = BaseAgent.from_yaml("agent.yaml")
```

### ReActAgent

ReAct (Reasoning + Acting) agents use a prompt strategy that interleaves reasoning traces with actions. Better for single-step tool usage.

```python
from kader.agent import ReActAgent
from kader.providers import OllamaProvider
from kader.tools import ReadFileTool, TodoTool

# ReAct agent with tools
react_agent = ReActAgent(
    name="ReActBot",
    tools=[ReadFileTool(), TodoTool()],
    provider=OllamaProvider(model="llama3.2"),
    interrupt_before_tool=True,
)

# The agent will reason about the task and use tools as needed
response = react_agent.invoke(
    "Read the file example.py and create a todo list for the tasks mentioned"
)
```

### PlanningAgent

Planning agents break down complex tasks into plans and execute them step by step using the TodoTool.

```python
from kader.agent import PlanningAgent
from kader.providers import OllamaProvider
from kader.tools import TodoTool

# Planning agent automatically creates and manages task lists
planner = PlanningAgent(
    name="PlannerBot",
    tools=[TodoTool()],
    provider=OllamaProvider(model="llama3.2"),
    use_persistence=True,
)

# The agent will:
# 1. Analyze the request
# 2. Create a plan using TodoTool
# 3. Execute each step
# 4. Update task status as it completes
response = planner.invoke(
    "Create a Python web project with: "
    "1. Setup Flask app, "
    "2. Add routes, "
    "3. Write tests"
)
```

## LLM Providers

### OllamaProvider

For local LLM inference. Best for privacy, speed, and offline capability.

```python
from kader.providers import OllamaProvider, Message

provider = OllamaProvider(
    model="llama3.2",          # Model name
    base_url="http://localhost:11434",  # Custom endpoint
    timeout=120,               # Request timeout
)

# Basic invocation
messages = [
    Message.system("You are helpful."),
    Message.user("What is Ollama?"),
]
response = provider.invoke(messages)
print(response.content)

# Streaming
for chunk in provider.stream(messages):
    print(chunk.content, end="", flush=True)

# Async
import asyncio
response = await provider.ainvoke(messages)
```

### GoogleProvider

Google Gemini models via the Google GenAI SDK.

```python
from kader.providers import GoogleProvider, Message

provider = GoogleProvider(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048,
)

# Requires GEMINI_API_KEY in environment
response = provider.invoke([Message.user("Hello from Gemini!")])
```

### AnthropicProvider

Anthropic Claude models via the Anthropic SDK.

```python
from kader.providers import AnthropicProvider, Message

provider = AnthropicProvider(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
)

# Requires ANTHROPIC_API_KEY
response = provider.invoke([Message.user("Hello from Claude!")])
```

Configuration via environment or parameters:
```bash
export GEMINI_API_KEY="your-key"
```

### MistralProvider

Mistral AI models for cloud inference.

```python
from kader.providers import MistralProvider, Message

provider = MistralProvider(
    model="mistral-large-latest",
    temperature=0.7,
)

# Requires MISTRAL_API_KEY
response = provider.invoke([Message.user("Hello from Mistral!")])
```

### OpenAICompatibleProvider

Connect to OpenAI, Groq, OpenRouter, Moonshot AI, and other OpenAI-compatible APIs.

```python
 import OpenAICompatiblefrom kader.providersProvider, Message, OpenAIProviderConfig

# Standard OpenAI
openai_provider = OpenAICompatibleProvider(
    model="gpt-4o",
    config=OpenAIProviderConfig(
        api_key="your-openai-key",
    )
)

# Groq (fast inference)
groq_provider = OpenAICompatibleProvider(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    config=OpenAIProviderConfig(
        api_key="your-groq-key",
    )
)

# OpenRouter (200+ models)
openrouter_provider = OpenAICompatibleProvider(
    model="anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    config=OpenAIProviderConfig(
        api_key="your-openrouter-key",
    )
)
```

## Tools

### Built-in Tools

Kader provides several built-in tools for common tasks:

```python
from kader.tools import (
    ReadFileTool,
    WriteFileTool,
    GlobTool,
    GrepTool,
    CommandExecutorTool,
    AgentTool,
    TodoTool,
)
```

### File System Tools

```python
from kader.tools import ReadFileTool, WriteFileTool, GlobTool

# Read a file
read_tool = ReadFileTool()
result = read_tool.execute(path="README.md")
print(result.content)

# Write or edit files
write_tool = WriteFileTool()
result = write_tool.execute(
    content="# Hello World\n",
    path="new_file.md"
)

# Find files by pattern
glob_tool = GlobTool()
result = glob_tool.execute(pattern="**/*.py")
print(result.matches)
```

### Command Execution

```python
from kader.tools import CommandExecutorTool

cmd_tool = CommandExecutorTool()

# Execute shell commands
result = cmd_tool.execute(command="ls -la")

# Run Python code
result = cmd_tool.execute(command="python -c 'print(1 + 1)'")
```

### Todo Tool

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

### Agent Tool (Sub-agents)

Spawn sub-agents with isolated memory for specific tasks:

```python
from kader.tools import AgentTool
from kader.agent import BaseAgent
from kader.providers import OllamaProvider

# Create an agent tool
agent_tool = AgentTool()

# Use it within another agent
parent_agent = BaseAgent(
    name="ParentAgent",
    tools=[agent_tool],
    provider=OllamaProvider(model="llama3.2"),
)

# The parent can delegate tasks to sub-agents
response = parent_agent.invoke(
    "Use the agent tool to research Python async patterns"
)
```

### Skills Tool

Load specialized instructions (skills) for specific domains or tasks:

```python
from kader.tools import SkillsTool
from kader.agent import BaseAgent
from kader.providers import OllamaProvider

# Create skills tool (loads from ~/.kader/skills and ./.kader/skills)
skills_tool = SkillsTool()

# Or specify custom skill directories
from pathlib import Path
skills_tool = SkillsTool(
    skills_dirs=[Path("./custom_skills")],
    priority_dir=Path("./project_skills"),  # Check this directory first
)

# Use it within an agent
agent = BaseAgent(
    name="SkillfulAgent",
    tools=[skills_tool],
    provider=OllamaProvider(model="llama3.2"),
)

# Load a specific skill
result = skills_tool.execute(name="python-expert")
print(result["content"])  # The skill's instructions
```

#### Skill File Format

Skills are stored in directories named after the skill, each containing a `SKILL.md` file with YAML frontmatter:

```markdown
---
name: python-expert
description: Expert in Python programming and best practices
---

# Python Expert Skill

You are an expert Python developer with deep knowledge of:
- PEP 8 style guide
- Type hints and annotations
- Async/await patterns
- Performance optimization

When writing Python code:
1. Use type hints whenever possible
2. Follow PEP 8 conventions
3. Write docstrings in Google style
4. Prefer list comprehensions over loops
```

#### Available Methods

```python
# Load a specific skill by name
skill = skill_loader.load_skill("python-expert")

# List all available skills
all_skills = skill_loader.list_skills()

# Get formatted description of all skills
description = skill_loader.get_description()
```

### Custom Tools

Create your own tools by subclassing `BaseTool`:

```python
from kader.tools import BaseTool

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
                    description="Temperature units (celsius/fahrenheit)",
                    required=False,
                ),
            ],
        )

    def execute(self, city: str, units: str = "celsius") -> ToolResult:
        # Implement your tool logic
        weather_data = get_weather(city, units)  # Your implementation
        return ToolResult(
            status="success",
            content=f"Weather in {city}: {weather_data}",
        )

# Use the custom tool
weather_tool = WeatherTool()
agent = BaseAgent(
    name="WeatherBot",
    tools=[weather_tool],
    provider=OllamaProvider(model="llama3.2"),
)
```

## Memory Management

### AgentState

Key-value storage for agent data:

```python
from kader.memory import AgentState

state = AgentState(agent_id="my-agent")

# Set values
state.set("user_name", "Alice")
state.set("preferences", {"theme": "dark", "language": "en"})

# Get values
print(state.get("user_name"))
print(state.get("preferences"))

# Check existence
if "user_name" in state:
    print("User name is set")
```

### Session Management

Persist sessions to disk:

```python
from kader.memory import FileSessionManager, AgentState

# Create session manager
session_mgr = FileSessionManager()

# Create new session
session = session_mgr.create_session("my-agent")
print(f"Session ID: {session.session_id}")

# Save agent state
state = AgentState(agent_id="my-agent")
state.set("counter", 0)
session_mgr.save_agent_state(session.session_id, state)

# Load session later
loaded_state = session_mgr.load_agent_state(session.session_id)
```

### Conversation Management

Sliding window for managing conversation history:

```python
from kader.memory import SlidingWindowConversationManager
from kader.providers import Message

# Create conversation manager with window size
conv_mgr = SlidingWindowConversationManager(window_size=10)

# Add messages
conv_mgr.add_message(Message.user("Hello"))
conv_mgr.add_message(Message.assistant("Hi there!"))
conv_mgr.add_message(Message.user("How are you?"))

# Get messages (automatically manages window)
messages = conv_mgr.get_messages()

# Persist conversation
session_mgr.save_conversation(session_id, [msg.message for msg in conv_mgr.get_messages()])
```

### Tool Output Compression

Compress tool outputs to save token space:

```python
from kader.memory import ToolOutputCompressor

compressor = ToolOutputCompressor(
    max_length=1000,  # Max characters per output
    summary_type="truncate",  # or "first_last"
)

compressed = compressor.compress(
    tool_name="read_file",
    output="Very long file content..."
)
```

## Workflows

### PlannerExecutorWorkflow

A sophisticated workflow that combines a PlanningAgent with sub-agent delegation:

```python
from kader.workflows import PlannerExecutorWorkflow
from kader.providers import OllamaProvider

# Create workflow
workflow = PlannerExecutorWorkflow(
    name="project_workflow",
    provider=OllamaProvider(model="llama3.2"),
    interrupt_before_tool=True,
)

# Execute complex task
result = workflow.run(
    "Create a Python project with: "
    "1. A hello world function, "
    "2. A test file, "
    "3. A requirements.txt"
)

print(result)
```

The workflow:
1. Creates a PlanningAgent with TodoTool and AgentTool
2. PlanningAgent breaks down the task into subtasks
3. Uses AgentTool to delegate subtasks to executor agents
4. Updates todo status as tasks complete
5. Summarizes completed work

### Custom Workflows

Extend `BaseWorkflow` for custom patterns:

```python
from kader.workflows import BaseWorkflow

class MyWorkflow(BaseWorkflow):
    def __init__(self, name: str, provider, **kwargs):
        super().__init__(name, provider)
        # Initialize your workflow components

    def run(self, task: str) -> str:
        # Implement your workflow logic
        result = self._execute_steps(task)
        return result

    async def arun(self, task: str) -> str:
        # Async implementation
        return await self._execute_steps_async(task)
```

## Examples

See the `examples/` directory for complete working examples:

| Example | Description |
|---------|-------------|
| `ollama_example.py` | Local LLM with Ollama |
| `google_example.py` | Google Gemini integration |
| `mistral_example.py` | Mistral AI integration |
| `anthropic_example.py` | Anthropic Claude integration |
| `openai_compatible_example.py` | OpenAI, Groq, OpenRouter |
| `memory_example.py` | Memory management |
| `todo_agent/main.py` | Todo-based planning agent |

Run examples:
```bash
cd examples
python ollama_example.py
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `BaseAgent` | Foundation agent class |
| `ReActAgent` | Reasoning + Acting agent |
| `PlanningAgent` | Task planning agent |
| `OllamaProvider` | Local LLM provider |
| `GoogleProvider` | Google Gemini provider |
| `MistralProvider` | Mistral AI provider |
| `AnthropicProvider` | Anthropic Claude provider |
| `OpenAICompatibleProvider` | OpenAI-compatible providers |
| `FileSessionManager` | Session persistence |
| `SlidingWindowConversationManager` | Conversation history |
| `ToolOutputCompressor` | Token optimization |
| `PlannerExecutorWorkflow` | Planning + execution workflow |

### Key Functions

- `Message.system()` - Create system message
- `Message.user()` - Create user message
- `Message.assistant()` - Create assistant message

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `KADER_DIR` | Kader config directory (default: `~/.kader`) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `MISTRAL_API_KEY` | Mistral API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `MOONSHOT_API_KEY` | Moonshot AI key |
| `GROQ_API_KEY` | Groq API key |
| `OPENROUTER_API_KEY` | OpenRouter key |

### YAML Agent Configuration

```yaml
name: MyAgent
system_prompt: "You are a helpful assistant."
tools:
  - read_file
  - write_file
  - todo_tool
provider:
  model: llama3.2
persistence: true
retry_attempts: 3
```

## Troubleshooting

### Common Issues

**No models found (Ollama)**
- Ensure Ollama is running: `ollama serve`
- Pull a model: `ollama pull llama3.2`

**API key errors**
- Check your `.env` file in `~/.kader/.env`
- Verify API key is correct and has sufficient credits

**Import errors**
- Ensure Kader is installed: `pip install -e .`
- Check Python version (3.11+ required)

**Tool execution failures**
- Check tool permissions
- Verify file paths exist for file operations

## License

MIT License - see LICENSE file for details.
