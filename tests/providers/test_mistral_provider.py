"""
Unit tests for the Mistral provider.
"""

import pytest

from kader.providers.base import (
    CostInfo,
    Message,
    ModelConfig,
    StreamChunk,
    Usage,
)
from kader.providers.mistral import MistralProvider, MISTRAL_PRICING


class TestMistralProviderInit:
    """Test cases for MistralProvider initialization."""

    def test_initialization_with_model(self):
        """Test provider initialization with model only."""
        provider = MistralProvider(model="mistral-small-latest")
        assert provider.model == "mistral-small-latest"

    def test_initialization_with_config(self):
        """Test provider initialization with default config."""
        config = ModelConfig(temperature=0.7, max_tokens=100)
        provider = MistralProvider(model="mistral-small-latest", default_config=config)
        
        assert provider.model == "mistral-small-latest"
        assert provider._default_config == config

    def test_repr(self):
        """Test string representation."""
        provider = MistralProvider(model="mistral-small-latest")
        assert repr(provider) == "MistralProvider(model='mistral-small-latest')"


class TestMistralProviderConvertMessages:
    """Test cases for message conversion."""

    def test_convert_simple_messages(self):
        """Test converting simple messages."""
        provider = MistralProvider(model="mistral-small-latest")
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
            Message.assistant("Hi there!"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 3
        assert converted[0]["role"] == "system"
        assert converted[0]["content"] == "You are a helpful assistant."
        assert converted[1]["role"] == "user"
        assert converted[1]["content"] == "Hello!"
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "Hi there!"

    def test_convert_tool_message(self):
        """Test converting tool message."""
        provider = MistralProvider(model="mistral-small-latest")
        messages = [
            Message.tool("call_123", "Tool result content"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["content"] == "Tool result content"
        assert converted[0]["tool_call_id"] == "call_123"

    def test_convert_message_with_tool_calls(self):
        """Test converting assistant message with tool calls."""
        provider = MistralProvider(model="mistral-small-latest")
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
            ),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        assert "tool_calls" in converted[0]
        assert converted[0]["tool_calls"][0]["id"] == "call_123"
        assert converted[0]["tool_calls"][0]["function"]["name"] == "get_weather"


class TestMistralProviderConvertConfig:
    """Test cases for config conversion."""

    def test_convert_default_config(self):
        """Test converting default config."""
        provider = MistralProvider(model="mistral-small-latest")
        config = ModelConfig()

        params = provider._convert_config_to_params(config)

        # Default values should not be included
        assert "temperature" not in params
        assert "max_tokens" not in params
        assert "top_p" not in params

    def test_convert_custom_config(self):
        """Test converting custom config."""
        provider = MistralProvider(model="mistral-small-latest")
        config = ModelConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            stop_sequences=["END"],
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        params = provider._convert_config_to_params(config)

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9
        assert params["stop"] == ["END"]
        assert params["random_seed"] == 42
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3

    def test_convert_config_with_tools(self):
        """Test converting config with tools."""
        provider = MistralProvider(model="mistral-small-latest")
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

        assert params["tools"] == tools
        assert params["tool_choice"] == "auto"

    def test_convert_config_with_response_format(self):
        """Test converting config with response format."""
        provider = MistralProvider(model="mistral-small-latest")
        config = ModelConfig(response_format={"type": "json_object"})

        params = provider._convert_config_to_params(config)

        assert params["response_format"] == {"type": "json_object"}


class TestMistralProviderEstimateCost:
    """Test cases for cost estimation."""

    def test_estimate_cost_known_model(self):
        """Test estimating cost for known model."""
        provider = MistralProvider(model="mistral-small-latest")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # mistral-small-latest: $0.2/M input, $0.6/M output
        expected_input = (1000 / 1_000_000) * 0.2
        expected_output = (500 / 1_000_000) * 0.6
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_large_model(self):
        """Test estimating cost for large model."""
        provider = MistralProvider(model="mistral-large-latest")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # mistral-large-latest: $2.0/M input, $6.0/M output
        expected_input = (1000 / 1_000_000) * 2.0
        expected_output = (500 / 1_000_000) * 6.0

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9

    def test_estimate_cost_unknown_model(self):
        """Test estimating cost for unknown model falls back to default."""
        provider = MistralProvider(model="unknown-model")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # Should fall back to mistral-small pricing
        assert cost.total_cost > 0

    def test_estimate_cost_zero_usage(self):
        """Test estimating cost with zero usage."""
        provider = MistralProvider(model="mistral-small-latest")
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        cost = provider.estimate_cost(usage)

        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0


class TestMistralProviderCountTokens:
    """Test cases for token counting."""

    def test_count_tokens_string(self):
        """Test counting tokens in a string."""
        provider = MistralProvider(model="mistral-small-latest")
        text = "Hello, world! This is a test."

        tokens = provider.count_tokens(text)

        # Approximate: ~4 chars per token
        assert tokens == len(text) // 4

    def test_count_tokens_messages(self):
        """Test counting tokens in messages."""
        provider = MistralProvider(model="mistral-small-latest")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there"),
        ]

        tokens = provider.count_tokens(messages)

        total_chars = len("Hello") + len("Hi there")
        assert tokens == total_chars // 4


class TestMistralPricing:
    """Test cases for pricing data."""

    def test_pricing_data_exists(self):
        """Test that pricing data exists for known models."""
        assert "mistral-large-latest" in MISTRAL_PRICING
        assert "mistral-small-latest" in MISTRAL_PRICING
        assert "codestral-latest" in MISTRAL_PRICING

    def test_pricing_format(self):
        """Test that pricing data has correct format."""
        for model_name, pricing in MISTRAL_PRICING.items():
            assert hasattr(pricing, "input_cost_per_million")
            assert hasattr(pricing, "output_cost_per_million")
            assert pricing.input_cost_per_million >= 0
            assert pricing.output_cost_per_million >= 0
