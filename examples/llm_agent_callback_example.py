"""
LLM and Agent Callback Example.

Demonstrates how to use LLM callbacks and Agent callbacks to monitor
and modify the agent execution lifecycle.
"""

import io
import os
import sys
from typing import Any

from kader.agent.base import BaseAgent
from kader.callbacks import CallbackContext, LLMCallback
from kader.callbacks.base import BaseCallback
from kader.providers.mistral import MistralProvider
from kader.tools.exec_commands import CommandExecutorTool

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MyLLMCallback(LLMCallback):
    """
    Callback that logs and can modify LLM calls.

    Example:
        agent = BaseAgent(
            callbacks=[MyLLMCallback()]
        )
    """

    def on_llm_start(
        self,
        context: CallbackContext,
        messages: list,
        config,
    ):
        """Log LLM call start and optionally modify messages/config."""
        print(f"\n[LLM Callback] {context.agent_name}: LLM call starting...")
        print(f"[LLM Callback] Number of messages: {len(messages)}")

        for i, msg in enumerate(messages):
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", "")[:50]
            print(f"[LLM Callback]   Message {i}: {role} - {content}...")

        if config:
            print(
                f"[LLM Callback] Config: temperature={config.temperature}, max_tokens={config.max_tokens}"
            )

        return messages, config

    def on_llm_end(
        self,
        context: CallbackContext,
        messages: list,
        response,
    ):
        """Log LLM response and optionally modify response."""
        print(f"\n[LLM Callback] {context.agent_name}: LLM call complete!")

        content = getattr(response, "content", "")
        print(f"[LLM Callback] Response: {content[:100]}...")

        model = getattr(response, "model", "unknown")
        print(f"[LLM Callback] Model: {model}")

        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            print(
                f"[LLM Callback] Usage: {usage.prompt_tokens}p / "
                f"{usage.completion_tokens}c / {usage.total_tokens}t"
            )

        return response


class MyAgentCallback(BaseCallback):
    """
    Callback that logs agent start and end events.

    Example:
        agent = BaseAgent(
            callbacks=[MyAgentCallback()]
        )
    """

    def on_agent_start(self, context: CallbackContext) -> None:
        """Log agent start."""
        print(f"\n[Agent Callback] {context.agent_name}: Agent starting!")

    def on_agent_end(self, context: CallbackContext) -> None:
        """Log agent end."""
        print(f"\n[Agent Callback] {context.agent_name}: Agent finished!")


def main():
    print("=== LLM and Agent Callback Example ===\n")

    cmd_tool = CommandExecutorTool()

    provider = MistralProvider(model="mistral-vibe-cli-with-tools")

    agent = BaseAgent(
        name="callback_demo",
        provider=provider,
        system_prompt="You are a helpful assistant.",
        tools=[cmd_tool],
        callbacks=[MyLLMCallback(), MyAgentCallback()],
        retry_attempts=1,
        interrupt_before_tool=False,
    )

    print(f"Agent '{agent.name}' initialized with callbacks\n")

    print("Testing via agent invoke:\n")

    try:
        response = agent.invoke("Hello, how are you?")
        if response:
            print(f"\nAgent Response: {response.content}")
    except Exception as e:
        print(f"\nExecution error: {e}")


if __name__ == "__main__":
    main()
