"""
Todo Agent Example

This example demonstrates how to use the TodoTool with a generic agent.
The agent will be instructed to plan a party, creating a todo list and updating it.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[2]))

# Fix encoding for Windows consoles
sys.stdout.reconfigure(encoding="utf-8")

from kader.agent.base import BaseAgent
from kader.tools.todo import TodoTool


def main():
    # 1. Initialize the agent
    print("Initializing Agent...")
    agent = BaseAgent(
        name="PlannerBot",
        system_prompt=(
            "You are a helpful planning assistant. "
            "Use the 'todo_tool' to manage tasks. "
            "Always check the todo list before adding new items to avoid duplicates. "
            "When a task is done, mark it as completed."
        ),
        tools=[TodoTool()],
        model_name="gpt-oss:120b-cloud",  # Using the user's preferred model
        use_persistence=True,  # Enable session persistence
        interrupt_before_tool=True,
    )

    print(f"Agent Session ID: {agent.session_id}")

    # 2. Define a scenario
    prompts = [
        "Create a plan for a surprise birthday party. Create a todo list with ID 'party-plan' and at least 3 initial items.",
        "I've bought the cake and balloons. Please update the todo list 'party-plan' to mark those as completed.",
        "Read the 'party-plan' todo list and show me the status of all items.",
    ]

    # 3. Run the agent
    for prompt in prompts:
        print(f"\nUser: {prompt}")
        print("-" * 50)

        try:
            response = agent.invoke(prompt)
            print(f"Agent: {response.content}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
