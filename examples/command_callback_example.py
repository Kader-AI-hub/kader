"""
Command Callback Example.

Demonstrates how to use the GitRtkCallback to modify git commands before execution.
"""

import io
import os
import sys
from typing import Any

from kader.agent.base import BaseAgent
from kader.callbacks.tool_callbacks import CallbackContext, CommandTransformCallback
from kader.providers.mistral import MistralProvider
from kader.tools.exec_commands import CommandExecutorTool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class GitRtkCallback(CommandTransformCallback):
    """
    Callback that prepends 'rtk' to git commands.

    Use this callback to automatically prefix all git commands with 'rtk'.

    Example:
        agent = BaseAgent(
            callbacks=[GitRtkCallback()]
        )
    """

    def on_tool_before(
        self,
        context: CallbackContext,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Add 'rtk' prefix to git commands."""
        if tool_name == "execute_command":
            command = arguments.get("command", "")
            if command.strip().startswith("git"):
                arguments["command"] = f"rtk {command}"
        return arguments


def main():
    print("=== Command Callback Example ===\n")

    cmd_tool = CommandExecutorTool()

    provider = MistralProvider(model="mistral-vibe-cli-with-tools")

    agent = BaseAgent(
        name="git_helper",
        provider=provider,
        system_prompt="You are a helpful assistant that can execute git commands. "
        "Use the execute_command tool to run git commands.",
        tools=[cmd_tool],
        callbacks=[GitRtkCallback()],
        retry_attempts=2,
        interrupt_before_tool=False,
    )

    print(f"Agent '{agent.name}' initialized with GitRtkCallback\n")

    print("Testing via agent invoke:\n")

    try:
        response = agent.invoke("Show me the git status")
        if response:
            print(f"\nAgent Response: {response.content}")
    except Exception as e:
        print(f"\nExecution error: {e}")


if __name__ == "__main__":
    main()
