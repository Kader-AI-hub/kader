import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kader.agent.agents import ReActAgent


async def main():
    print("=== Python Expert Agent (YAML Config) ===")
    print("Type '/exit' or '/close' to quit.\n")

    # 1. Load Agent from YAML
    # We use ReActAgent.from_yaml which inherits from BaseAgent.from_yaml
    # No need to explicitly pass tool_registry, it will use default if missing.
    yaml_path = Path(__file__).parent / "template.yaml"

    try:
        agent = ReActAgent.from_yaml(yaml_path)
    except Exception as e:
        print(f"Error loading agent from YAML: {e}")
        return

    print(f"Agent '{agent.name}' loaded from YAML.")
    print(f"Session ID: {agent.session_id}")
    print(f"Loaded Tools: {list(agent.tools_map.keys())}")

    # 4. Interactive Loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/close", "exit", "quit"]:
                print("Goodbye!")
                break

            print("\nThinking...")

            # Invoke Agent
            response = await agent.ainvoke(user_input)
            print(f"Agent: {response.content}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
