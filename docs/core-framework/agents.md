# Agents

Kader provides several agent types for different reasoning patterns and use cases.

## Agent Types

### BaseAgent

The foundation class for all agents. Provides core functionality for tool execution, conversation management, and session persistence.

```python
from kader.agent import BaseAgent
from kader.providers import OllamaProvider, Message

agent = BaseAgent(
    name="MyAgent",
    system_prompt="You are a helpful assistant.",
    tools=[],
    provider=OllamaProvider(model="llama3.2"),
    use_persistence=True,
    interrupt_before_tool=True,
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

#### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Agent name |
| `system_prompt` | str | System instructions |
| `tools` | list | List of tools |
| `provider` | BaseLLMProvider | LLM provider |
| `use_persistence` | bool | Enable session saving |
| `interrupt_before_tool` | bool | Confirm before tool execution |
| `retry_attempts` | int | Number of retry attempts |

### ReActAgent

ReAct (Reasoning + Acting) agents use a prompt strategy that interleaves reasoning traces with actions. Better for single-step tool usage.

```python
from kader.agent import ReActAgent
from kader.providers import OllamaProvider
from kader.tools import ReadFileTool, TodoTool

react_agent = ReActAgent(
    name="ReActBot",
    tools=[ReadFileTool(), TodoTool()],
    provider=OllamaProvider(model="llama3.2"),
    interrupt_before_tool=True,
)

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

planner = PlanningAgent(
    name="PlannerBot",
    tools=[TodoTool()],
    provider=OllamaProvider(model="llama3.2"),
    use_persistence=True,
)

response = planner.invoke(
    "Create a Python web project with: "
    "1. Setup Flask app, "
    "2. Add routes, "
    "3. Write tests"
)
```

The PlanningAgent will:
1. Analyze the request
2. Create a plan using TodoTool
3. Execute each step
4. Update task status as it completes

## YAML Configuration

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
  provider: ollama
persistence: true
retry_attempts: 3
```

```python
agent = BaseAgent.from_yaml("agent.yaml")
```

## Choosing the Right Agent

| Agent Type | Use Case |
|------------|----------|
| BaseAgent | Simple tasks, custom workflows |
| ReActAgent | Single-step tool usage, reasoning + acting |
| PlanningAgent | Complex multi-step tasks, project creation |
