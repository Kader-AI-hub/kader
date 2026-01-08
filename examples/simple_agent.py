"""
Simple Agent Example.

Demonstrates how to use the BaseAgent with:
- Custom System Prompt using PromptBase
- Tool Integration
- YAML Saving/Loading
- Execution
"""

import io
import os
import sys
from pathlib import Path

# Force utf-8 output for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.agent.base import BaseAgent
from kader.prompts import PromptBase
from kader.tools import ParameterSchema, tool


# 1. Define a simple tool using the decorator
@tool(
    name="calculator",
    description="Useful for calculating sums",
    parameters=[
        ParameterSchema(name="a", type="number", description="First number"),
        ParameterSchema(name="b", type="number", description="Second number"),
    ],
)
def calculator(a: float, b: float) -> str:
    return str(a + b)


def main():
    print("=== Simple Agent Demo ===\n")

    # 2. Define System Prompt
    system_prompt = PromptBase(
        template="You are a helpful AI assistant named {{ name }}. You have access to tools.",
        name="KaderBot",
    )

    # 3. Initialize Agent
    # Automatically uses OllamaProvider (default model: gpt-oss:120b-cloud)
    # Ensure you have ollama running: `ollama serve` and pulled the model
    # Or pass a specific model in `model_name`
    agent = BaseAgent(
        name="kader_helper",
        system_prompt=system_prompt,
        tools=[calculator],
        retry_attempts=2,
        interrupt_before_tool=True,
    )

    print(f"Agent '{agent.name}' initialized.")
    print(f"System Prompt: {agent.system_prompt}")  # Resolves to string
    print(f"Tools: {list(agent.tools_map.keys())}\n")

    # 4. Save to YAML
    yaml_file = Path("kader_helper.yaml")
    agent.to_yaml(yaml_file)
    print(f"Agent config saved to {yaml_file}")

    # 5. Invoke Agent (Mocking the run here effectively for the example if Ollama isn't up)
    # But intended for real run.
    print("\n[Note] Attempting to invoke agent. Ensure Ollama is running.")
    try:
        response = agent.invoke("What is 5 + 7?")
        if response:
            print(f"\nResponse: {response.content}")
            if response.has_tool_calls:
                print("(Tool calls were made during execution)")
    except Exception as e:
        print(f"\nExecution failed (is Ollama running?): {e}")

    # Clean up
    if yaml_file.exists():
        yaml_file.unlink()


if __name__ == "__main__":
    main()
