"""
ReAct Agent Interactive Example.

Demonstrates using ReActAgent with tools and memory in an interactive loop.
"""

import asyncio
import io
import os
import sys

# Force utf-8 output for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.agent.agents import ReActAgent
from kader.memory import SlidingWindowConversationManager
from kader.tools import (
    CommandExecutorTool,
    ToolRegistry,
    WebFetchTool,
    WebSearchTool,
    get_filesystem_tools,
)


async def main():
    print("=== ReAct Agent Interactive Demo ===")
    print("Type '/exit' or '/close' to quit.\n")

    # Initialize Tool Registry
    registry = ToolRegistry()
    fs_tools = get_filesystem_tools()
    for tool in fs_tools:
        registry.register(tool)
    registry.register(CommandExecutorTool())

    # Add Web Tools
    try:
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
    except ImportError:
        print(
            "Warning: Web tools not available (ollama library missing or incompatible)"
        )

    # Initialize Memory
    # Using SlidingWindowConversationManager to keep track of history
    memory = SlidingWindowConversationManager(window_size=10)

    # Initialize Agent
    agent = ReActAgent(
        name="react_assistant",
        tools=registry,
        memory=memory,
        model_name="gpt-oss:120b-cloud",
        use_persistence=True,  # Enable session persistence
        interrupt_before_tool=True,
    )

    print(f"Agent '{agent.name}' initialized with session ID: {agent.session_id}")

    print(
        f"Agent '{agent.name}' initialized with tools: {list(agent.tools_map.keys())}"
    )

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/close"]:
            print("Goodbye!")
            break

        print("Agent thinking...", end="", flush=True)

        # We use ainvoke for async main, or invoke for sync.
        # Let's use invoke for simplicity unless streaming is requested specifically
        # (User said "example should be interactive", implies chat)

        try:
            response = agent.invoke(user_input)
            print(f"\rAgent: {response.content}\n")

        except Exception as e:
            print(f"\rError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
