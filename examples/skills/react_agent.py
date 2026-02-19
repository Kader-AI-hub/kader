"""
ReAct Agent with Skills Example.

This example demonstrates using ReActAgent with skills loaded from the skills directory.
The agent has access to the 'hello' skill which provides greeting instructions.
"""

import asyncio
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kader.agent.agents import ReActAgent
from kader.memory import SlidingWindowConversationManager
from kader.tools import get_default_registry


async def main():
    print("=== ReAct Agent with Skills Demo ===")
    print("This agent has access to the 'hello' skill.")
    print("Type '/exit' or '/close' to quit.\n")

    # Get skills directory (parent of react_agent directory)
    skills_dir = Path(__file__).parent

    # Initialize Tool Registry with default tools and skills
    registry = get_default_registry(skills_dirs=[skills_dir])

    # Initialize Memory
    memory = SlidingWindowConversationManager(window_size=10)

    # Initialize Agent with skills
    agent = ReActAgent(
        name="hello_assistant",
        model_name="minimax-m2.5:cloud",
        tools=registry,
        memory=memory,
        skills_dirs=[skills_dir],
        use_persistence=True,
        interrupt_before_tool=True,
    )

    print(f"Agent '{agent.name}' initialized with session ID: {agent.session_id}")
    print(f"Skills loaded: {agent._skills_description}")
    print(f"Tools: {list(agent.tools_map.keys())}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["/exit", "/close"]:
            print("Goodbye!")
            break

        # If user asks to load the hello skill, demonstrate it
        if "load" in user_input.lower() and "skill" in user_input.lower():
            # Call the skills_tool directly to demonstrate
            skills_tool = agent.tools_map.get("skills_tool")
            if skills_tool:
                result = skills_tool.execute(name="hello")
                print("\nLoaded hello skill:")
                print(f"  Description: {result.get('description')}")
                print(f"  Content: {result.get('content')[:200]}...")
            continue

        print("Agent thinking...", end="", flush=True)

        try:
            response = agent.invoke(user_input)
            print(f"\rAgent: {response.content}\n")

        except Exception as e:
            print(f"\rError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
