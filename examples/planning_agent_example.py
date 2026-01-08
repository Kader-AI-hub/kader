import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.agent.agents import PlanningAgent
from kader.memory import SlidingWindowConversationManager
from kader.tools import CommandExecutorTool, ToolRegistry, get_filesystem_tools

# Try importing web tools if available
try:
    from kader.tools import WebFetchTool, WebSearchTool

    HAS_WEB_TOOLS = True
except ImportError:
    HAS_WEB_TOOLS = False


async def main():
    print("=== Planning Agent Interactive Demo ===")
    print("Type '/exit' or '/close' to quit.\n")

    # Initialize Tool Registry
    registry = ToolRegistry()

    # 1. Filesystem Tools
    fs_tools = get_filesystem_tools()
    for tool in fs_tools:
        registry.register(tool)

    # 2. Command Execution Tool
    registry.register(CommandExecutorTool())

    # 3. Web Tools (if available)
    if HAS_WEB_TOOLS:
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
    else:
        print(
            "Warning: Web tools not available (ollama library missing or incompatible)"
        )

    # Initialize Memory
    # Using SlidingWindowConversationManager to keep track of history
    memory = SlidingWindowConversationManager(window_size=10)

    # Initialize Planning Agent
    # We enable persistence to save state between runs locally
    agent = PlanningAgent(
        name="planning_assistant",
        tools=registry,
        memory=memory,
        model_name="gpt-oss:120b-cloud",
        use_persistence=True,
        interrupt_before_tool=True,
    )

    print(f"Agent '{agent.name}' initialized with session ID: {agent.session_id}")
    print(f"Tools available: {list(agent.tools_map.keys())}")

    # Interactive Loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/close", "exit", "quit"]:
                print("Goodbye!")
                break

            # Invoke Agent
            # PlanningAgent uses a plan-and-execute strategy typically driven by the LLM
            # responding to the prompt.
            print("\nPlanning Agent is thinking...")
            try:
                response = agent.invoke(user_input)
                print(f"\rAgent: {response.content}\n")
            except Exception as e:
                print(f"\nError during invocation: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
