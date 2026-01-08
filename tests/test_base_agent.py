import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, List

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.agent.base import BaseAgent
from kader.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
    Usage,
)
from kader.tools import BaseTool, ParameterSchema, ToolRegistry

# --- Mock Components ---


class EchoTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="echo_tool",
            description="Echoes back the input",
            parameters=[
                ParameterSchema(name="text", type="string", description="Text to echo")
            ],
        )

    def execute(self, text: str) -> str:
        return f"Echo: {text}"

    async def aexecute(self, text: str) -> str:
        return f"Async Echo: {text}"

    def get_interruption_message(self, **kwargs: Any) -> str:
        """Get interruption message for user confirmation."""
        text = kwargs.get("text", "")
        if text:
            return f"execute echo_tool: {text}"
        return "execute echo_tool"


class MockLLMProvider(BaseLLMProvider):
    def __init__(self, responses: List[str] = None):
        super().__init__(model="mock-model")
        self.responses = responses or ["Mock response"]
        self.current_response_idx = 0
        self.calls = []

    def _get_next_response(self) -> str:
        if self.current_response_idx >= len(self.responses):
            resp = self.responses[-1]
        else:
            resp = self.responses[self.current_response_idx]
            self.current_response_idx += 1
        return resp

    def invoke(
        self, messages: list[Message], config: ModelConfig | None = None
    ) -> LLMResponse:
        self.calls.append(messages)
        content = self._get_next_response()

        # Simple Logic to simulate tool call if content starts with TOOL_CALL:
        tool_calls = None
        if content.startswith("TOOL_CALL:"):
            # Format: TOOL_CALL:tool_name:json_args
            _, tool_name, json_args = content.split(":", 2)
            tool_calls = [
                {
                    "id": "call_mock_1",
                    "function": {
                        "name": tool_name.strip(),
                        "arguments": json.loads(json_args),
                    },
                }
            ]
            # Reset content for tool response
            content = ""

        return LLMResponse(
            content=content,
            model="mock-model",
            usage=Usage(10, 10),
            tool_calls=tool_calls,
        )

    async def ainvoke(
        self, messages: list[Message], config: ModelConfig | None = None
    ) -> LLMResponse:
        return self.invoke(messages, config)

    def stream(
        self, messages: list[Message], config: ModelConfig | None = None
    ) -> Iterator[StreamChunk]:
        self.calls.append(messages)
        content = self._get_next_response()
        # Simulate simple streaming of the content
        yield StreamChunk(
            content=content, delta=content, finish_reason="stop", usage=Usage(10, 10)
        )

    async def astream(
        self, messages: list[Message], config: ModelConfig | None = None
    ) -> AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        content = self._get_next_response()
        yield StreamChunk(
            content=content, delta=content, finish_reason="stop", usage=Usage(10, 10)
        )

    def count_tokens(self, text):
        return 0

    def estimate_cost(self, usage):
        return 0


# --- Tests ---


def test_agent_structure_and_yaml():
    print("Test 1: Structure and YAML Serialization...")
    tool = EchoTool()
    registry = ToolRegistry()
    registry.register(tool)

    agent = BaseAgent(
        name="test_agent",
        system_prompt="You are a test agent.",
        tools=registry,
        provider=MockLLMProvider(),  # Use mock provider, though for yaml it extracts model name
    )

    # Check proper initialization
    assert "echo_tool" in agent.tools_map
    assert agent.name == "test_agent"

    # Check YAML serialization
    yaml_path = Path("test_agent_mock.yaml")
    agent.to_yaml(yaml_path)

    # Validate YAML content
    with open(yaml_path, "r") as f:
        content = f.read()
        assert "echo_tool" in content
        assert "test_agent" in content

    # Check YAML loading
    loaded_agent = BaseAgent.from_yaml(yaml_path, tool_registry=registry)
    assert loaded_agent.name == "test_agent"
    assert "echo_tool" in loaded_agent.tools_map
    # Note: loaded agent will have default OllamaProvider since we don't serialize custom provider class types currently
    # But for this test we mainly care about config restoration

    print("[OK] Structure and YAML tests passed.")
    if yaml_path.exists():
        yaml_path.unlink()


def test_agent_mock_invocation():
    print("\nTest 2: Mock Invocation...")
    mock_provider = MockLLMProvider(responses=["Hello user!"])
    agent = BaseAgent(
        name="invoker", system_prompt="Sys", provider=mock_provider, retry_attempts=1
    )

    response = agent.invoke("Hi")
    assert response.content == "Hello user!"
    assert len(mock_provider.calls) == 1
    # Check history (System + User)
    assert len(mock_provider.calls[0]) == 2
    print("[OK] Basic invocation passed.")


def test_agent_tool_usage():
    print("\nTest 3: Tool Usage Loop...")
    # 1. First response triggers tool.
    # 2. Second response is final answer.
    responses = ['TOOL_CALL:echo_tool:{"text": "hello_tool"}', "The tool said hello."]
    mock_provider = MockLLMProvider(responses=responses)
    tool = EchoTool()

    agent = BaseAgent(
        name="tool_user", system_prompt="Sys", tools=[tool], provider=mock_provider
    )

    # Mock the input function to return 'yes' automatically during testing
    import builtins

    original_input = builtins.input
    builtins.input = lambda prompt: "yes"

    try:
        final_response = agent.invoke("Use the tool")
    finally:
        # Restore the original input function
        builtins.input = original_input

    assert final_response.content == "The tool said hello."

    # Verify flow:
    # Call 1: [Sys, User] -> Returns Tool Call
    # Call 2: [Sys, User, Assistant(ToolCall), Tool(Result)] -> Returns Final Answer
    assert len(mock_provider.calls) == 2

    last_context = mock_provider.calls[1]
    assert len(last_context) == 4
    assert last_context[2].role == "assistant"
    assert last_context[2].tool_calls is not None
    assert last_context[3].role == "tool"
    assert "Echo: hello_tool" in last_context[3].content

    print("[OK] Tool usage loop passed.")


def test_agent_with_prompt_base():
    print("\nTest 4: PromptBase Usage...")
    from kader.prompts.base import PromptBase

    # Create simple template
    p = PromptBase(template="You are a {{ role }}.", role="tester")

    agent = BaseAgent(name="prompt_agent", system_prompt=p, provider=MockLLMProvider())

    # Check string conversion
    assert str(agent.system_prompt) == "You are a tester."

    # Check invoke uses it (Mock provider captures calls)
    agent.invoke("Hi")
    last_call = agent.provider.calls[0]

    # System prompt should be first
    assert last_call[0].role == "system"
    assert last_call[0].content == "You are a tester."

    print("[OK] PromptBase usage passed.")


def main():
    try:
        test_agent_structure_and_yaml()
        test_agent_mock_invocation()
        test_agent_tool_usage()
        test_agent_with_prompt_base()
        print("\nAll mock agent tests passed!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
