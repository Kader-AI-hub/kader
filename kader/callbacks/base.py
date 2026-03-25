"""
Base callback classes for Kader framework.

Provides abstract base classes for implementing callbacks that can
hook into various stages of agent execution.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any

from kader.tools.base import ToolResult as BaseToolResult


class CallbackEvent(str, Enum):
    """Events that can trigger callbacks."""

    TOOL_BEFORE = "tool_before"
    TOOL_AFTER = "tool_after"
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_START = "llm_start"
    LLM_END = "llm_end"
    ERROR = "error"


@dataclass
class CallbackContext:
    """Context passed to callback methods."""

    event: CallbackEvent
    agent_name: str
    extra: dict[str, Any]


class BaseCallback(ABC):
    """
    Abstract base class for callbacks.

    Subclasses can override specific methods to handle different
    stages of agent execution. Each callback can be enabled/disabled
    and supports both sync and async execution.

    Example:
        class MyCallback(BaseCallback):
            def on_tool_before(self, context, tool_name, arguments):
                print(f"Calling {tool_name}")
                return arguments
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize the callback.

        Args:
            enabled: Whether this callback is active.
        """
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        """Check if callback is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the callback."""
        self._enabled = value

    def on_tool_before(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Called before a tool is executed.

        Args:
            context: Callback context with event info.
            tool_name: Name of the tool being called.
            arguments: Arguments passed to the tool.

        Returns:
            Modified arguments (can be transformed or validated).
        """
        return arguments

    def on_tool_after(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: BaseToolResult,
    ) -> BaseToolResult:
        """
        Called after a tool is executed.

        Args:
            context: Callback context with event info.
            tool_name: Name of the tool that was called.
            arguments: Arguments that were passed to the tool.
            result: The tool execution result.

        Returns:
            Modified result (can be transformed).
        """
        return result

    def on_agent_start(self, context: CallbackContext) -> None:
        """
        Called when agent starts execution.

        Args:
            context: Callback context with event info.
        """
        pass

    def on_agent_end(self, context: CallbackContext) -> None:
        """
        Called when agent finishes execution.

        Args:
            context: Callback context with event info.
        """
        pass

    def on_llm_start(
        self, context: CallbackContext, messages: list[dict[str, Any]]
    ) -> None:
        """
        Called before LLM is invoked.

        Args:
            context: Callback context with event info.
            messages: Messages being sent to LLM.
        """
        pass

    def on_llm_end(self, context: CallbackContext, response: Any) -> None:
        """
        Called after LLM response is received.

        Args:
            context: Callback context with event info.
            response: LLM response.
        """
        pass

    def on_error(self, context: CallbackContext, error: Exception) -> None:
        """
        Called when an error occurs.

        Args:
            context: Callback context with event info.
            error: The exception that occurred.
        """
        pass
