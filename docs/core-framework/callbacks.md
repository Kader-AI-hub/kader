# Callbacks

Kader provides a callback system that allows you to hook into various stages of agent execution. Callbacks can be used for logging, monitoring, argument transformation, result modification, and more.

## Overview

Callbacks are triggered at different points during agent execution:

- **Agent Start/End** - When an agent begins or finishes execution
- **LLM Start/End** - Before and after LLM calls
- **Tool Start/End** - Before and after tool execution

## Callback Types

### BaseCallback

The base class for all callbacks. It provides default no-op implementations for all callback methods.

```python
from kader.callbacks import BaseCallback, CallbackContext, CallbackEvent

class MyCallback(BaseCallback):
    def on_agent_start(self, context: CallbackContext) -> None:
        print(f"Agent {context.agent_name} starting!")

    def on_agent_end(self, context: CallbackContext) -> None:
        print(f"Agent {context.agent_name} finished!")
```

### ToolCallback

Callbacks for tool execution events. Supports filtering by tool names.

```python
from kader.callbacks import ToolCallback, CallbackContext

class MyToolCallback(ToolCallback):
    def __init__(self, tool_names: list[str] | None = None):
        super().__init__(tool_names=tool_names)

    def on_tool_before(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict,
    ) -> dict:
        print(f"Calling {tool_name} with {arguments}")
        return arguments  # Can modify arguments

    def on_tool_after(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict,
        result,
    ):
        print(f"{tool_name} returned: {result}")
        return result  # Can modify result
```

### LLMCallback

Callbacks for LLM invocation events. Supports filtering by model names.

```python
from kader.callbacks import LLMCallback, CallbackContext

class MyLLMCallback(LLMCallback):
    def __init__(self, model_names: list[str] | None = None):
        super().__init__(model_names=model_names)

    def on_llm_start(
        self,
        context: CallbackContext,
        messages: list,
        config,
    ) -> tuple:
        print(f"LLM call starting with {len(messages)} messages")
        return messages, config  # Can modify messages and config

    def on_llm_end(
        self,
        context: CallbackContext,
        messages: list,
        response,
    ):
        print(f"LLM response: {response.content}")
        return response  # Can modify response
```

## Using Callbacks with Agents

Pass callbacks to the agent via the `callbacks` parameter:

```python
from kader.agent import BaseAgent
from kader.callbacks import ToolCallback, LLMCallback, BaseCallback, CallbackContext

# Create callbacks
class LoggingToolCallback(ToolCallback):
    def on_tool_before(self, context, tool_name, arguments):
        print(f"[LOG] Calling {tool_name}")
        return arguments

    def on_tool_after(self, context, tool_name, arguments, result):
        print(f"[LOG] {tool_name} -> {result}")
        return result

class LoggingLLMCallback(LLMCallback):
    def on_llm_start(self, context, messages, config):
        print(f"[LOG] LLM call starting")
        return messages, config

    def on_llm_end(self, context, messages, response):
        print(f"[LOG] LLM response: {response.content[:50]}...")
        return response

class AgentEventsCallback(BaseCallback):
    def on_agent_start(self, context):
        print(f"[LOG] Agent starting!")

    def on_agent_end(self, context):
        print(f"[LOG] Agent finished!")

# Initialize agent with callbacks
agent = BaseAgent(
    name="my_agent",
    system_prompt="You are a helpful assistant.",
    callbacks=[
        LoggingToolCallback(),
        LoggingLLMCallback(),
        AgentEventsCallback(),
    ],
)

response = agent.invoke("Hello!")
```

## Callback Execution Order

When multiple callbacks are registered, they are invoked in order:

1. `on_agent_start` - At the beginning of `invoke()`/`ainvoke()`
2. `on_llm_start` - Before each LLM call (inside the agent loop)
3. `on_llm_end` - After each LLM call
4. `on_tool_before` - Before each tool execution
5. `on_tool_after` - After each tool execution
6. `on_agent_end` - At the end of `invoke()`/`ainvoke()`

## Callback Context

All callbacks receive a `CallbackContext` object containing:

```python
@dataclass
class CallbackContext:
    event: CallbackEvent  # The event that triggered the callback
    agent_name: str       # Name of the agent
    extra: dict          # Additional context data
```

## Available Events

| Event | Description |
|-------|-------------|
| `CallbackEvent.AGENT_START` | Agent starts execution |
| `CallbackEvent.AGENT_END` | Agent finishes execution |
| `CallbackEvent.LLM_START` | Before LLM invocation |
| `CallbackEvent.LLM_END` | After LLM invocation |
| `CallbackEvent.TOOL_BEFORE` | Before tool execution |
| `CallbackEvent.TOOL_AFTER` | After tool execution |

## Built-in Callbacks

Kader provides several ready-to-use callbacks:

### LoggingToolCallback

Logs tool execution events to the console.

```python
from kader.callbacks import LoggingToolCallback

agent = BaseAgent(
    callbacks=[LoggingToolCallback(tool_names=["read_file", "write_file"])]
)
```

### LoggingLLMCallback

Logs LLM invocation events to the console.

```python
from kader.callbacks import LoggingLLMCallback

agent = BaseAgent(
    callbacks=[LoggingLLMCallback(model_names=["mistral-vibe-cli"])]
)
```

## Transforming Arguments and Results

Callbacks can modify arguments before execution and results after:

```python
class TransformCallback(ToolCallback):
    def on_tool_before(self, context, tool_name, arguments):
        # Add prefix to all arguments
        arguments["_callback"] = f"Modified by {context.agent_name}"
        return arguments

    def on_tool_after(self, context, tool_name, arguments, result):
        # Wrap result
        result.content = f"[MODIFIED] {result.content}"
        return result
```

## Async Support

All callbacks work with both sync and async agent methods:

```python
# Works with invoke()
response = agent.invoke("Hello")

# Works with ainvoke()
import asyncio
response = await agent.ainvoke("Hello")
```
