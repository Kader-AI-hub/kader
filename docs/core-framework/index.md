# Core Framework

The Kader core framework provides a flexible Python library for building intelligent AI coding agents.

## Overview

Kader is a flexible framework that enables developers to create AI agents capable of reasoning, planning, and executing tasks using various Large Language Models (LLMs). It provides a unified API for working with different LLM providers while maintaining a consistent interface for tools and memory management.

## Key Features

- **Multi-Provider Support**: Connect to Ollama, Google Gemini, Anthropic, Mistral, OpenAI, and any OpenAI-compatible API
- **Agent Types**: BaseAgent, ReActAgent, and PlanningAgent for different reasoning patterns
- **Tool System**: Build custom tools with automatic schema generation for any LLM provider
- **Memory Management**: Persistent sessions, conversation history with sliding windows, and state management
- **Workflows**: Planner-Executor pattern for complex multi-step tasks

## Core Concepts

### 1. Providers

Providers are the bridge between Kader and LLM services:

```python
from kader.providers import Message, OllamaProvider

provider = OllamaProvider(model="llama3.2")
response = provider.invoke([Message.user("Hello!")])
print(response.content)
```

### 2. Tools

Tools extend agent capabilities:

```python
from kader.tools import ReadFileTool, WriteFileTool, CommandExecutorTool

tools = [
    ReadFileTool(),
    WriteFileTool(),
    CommandExecutorTool(),
]
```

### 3. Agents

Agents combine providers, tools, and memory:

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

Memory systems manage conversation history:

```python
from kader.memory import SlidingWindowConversationManager, FileSessionManager

session_mgr = FileSessionManager()
session = session_mgr.create_session("my-agent")

conv_mgr = SlidingWindowConversationManager(window_size=20)
conv_mgr.add_message(Message.user("Hello"))
```

## Documentation Sections

- [Agents](agents.md) - BaseAgent, ReActAgent, PlanningAgent
- [Providers](providers.md) - Ollama, Google, Anthropic, Mistral, OpenAI-compatible
- [Tools](tools.md) - Built-in tools and custom tool creation
- [Memory](memory.md) - Session management and conversation history
