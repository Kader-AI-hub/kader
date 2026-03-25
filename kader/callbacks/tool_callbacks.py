"""
Tool callback implementations for Kader framework.

Provides callback classes that hook into tool execution lifecycle.
"""

from typing import Any

from kader.tools.base import ToolResult as BaseToolResult

from .base import BaseCallback, CallbackContext


class ToolCallback(BaseCallback):
    """
    Callback for tool execution events.

    Provides hooks before and after tool execution with support for
    filtering by tool names. Can modify arguments before execution
    and transform results after execution.

    Example:
        class LoggingCallback(ToolCallback):
            def on_tool_before(self, context, tool_name, arguments):
                print(f"Calling {tool_name}")
                return arguments

            def on_tool_after(self, context, tool_name, arguments, result):
                print(f"{tool_name} returned: {result.content}")
                return result
    """

    def __init__(
        self,
        tool_names: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize tool callback.

        Args:
            tool_names: List of tool names to respond to. None means all tools.
                       If specified, callback only fires for matching tools.
            enabled: Whether this callback is active.
        """
        super().__init__(enabled=enabled)
        self._tool_names = tool_names

    @property
    def tool_names(self) -> list[str] | None:
        """Get the list of tool names this callback responds to."""
        return self._tool_names

    def _matches_tool(self, tool_name: str) -> bool:
        """Check if this callback should respond to the given tool."""
        if self._tool_names is None:
            return True
        return tool_name in self._tool_names

    def on_tool_before(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Called before a tool is executed.

        Override this method to modify or validate tool arguments
        before execution. The returned arguments will be passed
        to the tool.

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

        Override this method to modify or log tool results
        after execution.

        Args:
            context: Callback context with event info.
            tool_name: Name of the tool that was called.
            arguments: Arguments that were passed to the tool.
            result: The tool execution result.

        Returns:
            Modified result (can be transformed).
        """
        return result


class LoggingToolCallback(ToolCallback):
    """
    Callback that logs tool execution events.

    Useful for debugging and monitoring tool calls.

    Example:
        agent = BaseAgent(
            name="my_agent",
            system_prompt="...",
            callbacks=[LoggingToolCallback()]
        )
    """

    def __init__(
        self,
        tool_names: list[str] | None = None,
        log_arguments: bool = True,
        log_results: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Initialize logging callback.

        Args:
            tool_names: List of tool names to log. None means all tools.
            log_arguments: Whether to log tool arguments.
            log_results: Whether to log tool results.
            enabled: Whether this callback is active.
        """
        super().__init__(tool_names=tool_names, enabled=enabled)
        self._log_arguments = log_arguments
        self._log_results = log_results

    def on_tool_before(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Log tool call before execution."""
        if not self._matches_tool(tool_name):
            return arguments

        if self._log_arguments:
            print(f"[Callback] {context.agent_name}: Calling {tool_name}")
            if arguments:
                print(f"[Callback] Arguments: {arguments}")

        return arguments

    def on_tool_after(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
        result: BaseToolResult,
    ) -> BaseToolResult:
        """Log tool result after execution."""
        if not self._matches_tool(tool_name):
            return result

        if self._log_results:
            status_str = (
                "success"
                if result.status == "success"
                else f"error: {result.error_message}"
            )
            print(
                f"[Callback] {context.agent_name}: {tool_name} -> {status_str}")
            if result.content and len(result.content) < 500:
                print(f"[Callback] Result: {result.content[:200]}...")

        return result
