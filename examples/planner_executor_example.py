"""
Planner-Executor Workflow Example.

Demonstrates the PlannerExecutorWorkflow that uses a PlanningAgent with
TodoTool for task management and AgentTool for sub-task delegation.
"""

import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.workflows import PlannerExecutorWorkflow


async def main():
    print("=== Planner-Executor Workflow Demo ===")
    print("This workflow uses a PlanningAgent to:")
    print("  1. Create a plan using TodoTool")
    print("  2. Delegate sub-tasks to executor agents")
    print("  3. Track progress and update todo status")
    print("\nType '/exit' or '/close' to quit.\n")

    # Initialize the workflow
    # Note: The planner (TodoTool, AgentTool) executes without interruption
    # Sub-agents spawned by AgentTool will respect interrupt_before_tool setting
    workflow = PlannerExecutorWorkflow(
        name="demo_workflow",
        model_name="qwen3-coder:480b-cloud",
        interrupt_before_tool=True,  # Applies to sub-agents inside AgentTool
        use_persistence=True,
        executor_names=["code_executor", "research_executor"],
    )

    print(f"Workflow '{workflow.name}' initialized")
    print(f"Planner session ID: {workflow.planner.session_id}")
    print(f"Available tools: {list(workflow.planner.tools_map.keys())}\n")

    # Interactive Loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/exit", "/close", "exit", "quit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "/reset":
                workflow.reset()
                print("Workflow reset. Fresh planner created.")
                continue

            # Run the workflow
            print("\nPlanner is analyzing and creating a plan...")
            try:
                result = workflow.run(user_input)
                print(f"\n[Workflow Result]:\n{result}\n")
            except Exception as e:
                print(f"\nError during workflow execution: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
