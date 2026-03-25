"""
Kader Callbacks - Hook system for agent execution lifecycle.

Provides callback classes that can hook into various stages of agent
execution, enabling custom logic for tool execution, logging, validation,
and more.

Example:
    from kader.callbacks import ToolCallback, LoggingToolCallback, BaseAgent

    class MyCallback(ToolCallback):
        def on_tool_before(self, context, tool_name, arguments):
            print(f"Calling {tool_name}")
            return arguments

    agent = BaseAgent(
        name="my_agent",
        system_prompt="...",
        callbacks=[MyCallback(), LoggingToolCallback()]
    )
"""

from .base import BaseCallback, CallbackContext, CallbackEvent
from .tool_callbacks import (
    LoggingToolCallback,
    ToolCallback,
)

__all__ = [
    "BaseCallback",
    "CallbackContext",
    "CallbackEvent",
    "ToolCallback",
    "LoggingToolCallback",
]
