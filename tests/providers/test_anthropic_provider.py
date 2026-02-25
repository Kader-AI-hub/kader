"""
Unit tests for the Anthropic provider.
"""

from kader.providers.anthropic import ANTHROPIC_PRICING, AnthropicProvider
from kader.providers.base import (
    Message,
    ModelConfig,
    Usage,
)


class TestAnthropicProviderInit:
    """Test cases for AnthropicProvider initialization."""

    def test_initialization_with_model(self):
        """Test provider initialization with model only."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        assert provider.model == "claude-3-5-sonnet-20241022"

    def test_initialization_with_config(self):
        """Test provider initialization with default config."""
        config = ModelConfig(temperature=0.7, max_tokens=100)
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-20241022", default_config=config
        )

        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider._default_config == config

    def test_initialization_with_api_key(self):
        """Test provider initialization with explicit api_key."""
        provider = AnthropicProvider(
            model="claude-3-5-sonnet-20241022", api_key="sk-ant-test"
        )
        assert provider._api_key == "sk-ant-test"

    def test_repr(self):
        """Test string representation."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        assert (
            repr(provider) == "AnthropicProvider(model='claude-3-5-sonnet-20241022')"
        )


class TestAnthropicProviderConvertMessages:
    """Test cases for message conversion."""

    def test_convert_simple_user_message(self):
        """Test converting a simple user message."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [Message.user("Hello!")]

        converted, system = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello!"
        assert system is None

    def test_convert_system_message(self):
        """Test that system messages are extracted separately."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
        ]

        converted, system = provider._convert_messages(messages)

        assert system == "You are a helpful assistant."
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_convert_assistant_message(self):
        """Test converting an assistant message."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [Message.assistant("Hi there!")]

        converted, system = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"

    def test_convert_tool_message(self):
        """Test converting a tool result message."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [Message.tool("call_123", "Tool result content")]

        converted, system = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        content_block = converted[0]["content"][0]
        assert content_block["type"] == "tool_result"
        assert content_block["tool_use_id"] == "call_123"
        assert content_block["content"] == "Tool result content"

    def test_convert_assistant_with_tool_calls(self):
        """Test converting an assistant message with tool calls."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    {
                        "id": "call_123",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            )
        ]

        converted, system = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "assistant"
        content = converted[0]["content"]
        # Find the tool_use block
        tool_blocks = [b for b in content if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["id"] == "call_123"
        # Arguments should be parsed from JSON string to dict
        assert tool_blocks[0]["input"] == {"location": "NYC"}

    def test_convert_tool_calls_with_dict_args(self):
        """Test converting tool calls with dict arguments (not JSON string)."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [
            Message(
                role="assistant",
                content="Using a tool",
                tool_calls=[
                    {
                        "id": "call_456",
                        "function": {
                            "name": "search",
                            "arguments": {"query": "python"},
                        },
                    }
                ],
            )
        ]

        converted, system = provider._convert_messages(messages)
        content = converted[0]["content"]
        tool_blocks = [b for b in content if b["type"] == "tool_use"]
        assert tool_blocks[0]["input"] == {"query": "python"}


class TestAnthropicProviderConvertConfig:
    """Test cases for config conversion."""

    def test_convert_default_config(self):
        """Test converting default config sets max_tokens default."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        config = ModelConfig()

        params = provider._convert_config_to_params(config)

        # max_tokens is required by Anthropic, defaults to 4096
        assert params["max_tokens"] == 4096
        # Default temperature/top_p should not be included
        assert "temperature" not in params
        assert "top_p" not in params

    def test_convert_custom_config(self):
        """Test converting custom config values."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        config = ModelConfig(
            temperature=0.7,
            max_tokens=200,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END"],
        )

        params = provider._convert_config_to_params(config)

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 200
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50
        assert params["stop_sequences"] == ["END"]

    def test_convert_config_with_system(self):
        """Test that system prompt is included in params."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        config = ModelConfig()

        params = provider._convert_config_to_params(config, system="Be helpful.")

        assert params["system"] == "Be helpful."

    def test_convert_config_with_tools(self):
        """Test converting config with tools."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        config = ModelConfig(tools=tools, tool_choice="auto")

        params = provider._convert_config_to_params(config)

        assert "tools" in params
        # Check Anthropic format
        assert params["tools"][0]["name"] == "get_weather"
        assert "input_schema" in params["tools"][0]
        # tool_choice should be converted to Anthropic dict format
        assert params["tool_choice"] == {"type": "auto"}

    def test_convert_tools_format(self):
        """Test OpenAI-format tools are converted to Anthropic format."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                    },
                },
            }
        ]

        converted = provider._convert_tools(tools)

        assert len(converted) == 1
        assert converted[0]["name"] == "calculate"
        assert converted[0]["description"] == "Perform a calculation"
        assert "input_schema" in converted[0]
        assert converted[0]["input_schema"]["type"] == "object"


class TestAnthropicProviderEstimateCost:
    """Test cases for cost estimation."""

    def test_estimate_cost_known_model(self):
        """Test estimating cost for a known model."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # claude-3-5-sonnet: $3.0/M input, $15.0/M output
        expected_input = (1000 / 1_000_000) * 3.0
        expected_output = (500 / 1_000_000) * 15.0
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_opus_model(self):
        """Test estimating cost for the Opus model."""
        provider = AnthropicProvider(model="claude-3-opus-20240229")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # claude-3-opus: $15.0/M input, $75.0/M output
        expected_input = (1000 / 1_000_000) * 15.0
        expected_output = (500 / 1_000_000) * 75.0

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9

    def test_estimate_cost_unknown_model(self):
        """Test estimating cost for unknown model falls back to default."""
        provider = AnthropicProvider(model="claude-unknown-model")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # Should fall back to claude-3-5-sonnet pricing
        assert cost.total_cost > 0

    def test_estimate_cost_zero_usage(self):
        """Test estimating cost with zero usage."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        cost = provider.estimate_cost(usage)

        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0


class TestAnthropicProviderCountTokens:
    """Test cases for token counting (char-based fallback)."""

    def test_count_tokens_string(self):
        """Test counting tokens in a string falls back to char approximation."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        text = "Hello, world! This is a test."

        # With no API key, count_tokens falls back to char-based estimation
        tokens = provider.count_tokens(text)

        # Rough approximation expected
        assert isinstance(tokens, int)
        assert tokens >= 0

    def test_count_tokens_messages(self):
        """Test counting tokens in messages falls back to char approximation."""
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there"),
        ]

        tokens = provider.count_tokens(messages)

        assert isinstance(tokens, int)
        assert tokens >= 0


class TestAnthropicPricing:
    """Test cases for pricing data."""

    def test_pricing_data_exists(self):
        """Test that pricing data exists for known models."""
        assert "claude-3-5-sonnet-20241022" in ANTHROPIC_PRICING
        assert "claude-3-opus-20240229" in ANTHROPIC_PRICING
        assert "claude-3-haiku-20240307" in ANTHROPIC_PRICING
        assert "claude-3-5-haiku-20241022" in ANTHROPIC_PRICING

    def test_pricing_format(self):
        """Test that pricing data has correct format."""
        for model_name, pricing in ANTHROPIC_PRICING.items():
            assert hasattr(pricing, "input_cost_per_million")
            assert hasattr(pricing, "output_cost_per_million")
            assert pricing.input_cost_per_million >= 0
            assert pricing.output_cost_per_million >= 0

    def test_haiku_cheaper_than_sonnet(self):
        """Test relative pricing: Haiku should be cheaper than Sonnet."""
        haiku = ANTHROPIC_PRICING["claude-3-haiku-20240307"]
        sonnet = ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]
        assert haiku.input_cost_per_million < sonnet.input_cost_per_million
        assert haiku.output_cost_per_million < sonnet.output_cost_per_million

    def test_cached_pricing_available(self):
        """Test that some models have cached pricing."""
        sonnet = ANTHROPIC_PRICING["claude-3-5-sonnet-20241022"]
        assert sonnet.cached_input_cost_per_million is not None
        assert sonnet.cached_input_cost_per_million < sonnet.input_cost_per_million
