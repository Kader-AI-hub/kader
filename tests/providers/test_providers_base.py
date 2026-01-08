"""
Unit tests for the base provider functionality.
"""

import pytest

from kader.providers.base import (
    BaseLLMProvider,
    CostInfo,
    LLMResponse,
    Message,
    MessageRole,
    ModelConfig,
    StreamChunk,
    Usage,
)


class ConcreteProvider(BaseLLMProvider):
    """Concrete implementation of BaseLLMProvider for testing."""

    def invoke(self, messages, config=None):
        return LLMResponse(
            content="test response",
            model=self.model,
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            finish_reason="stop",
        )

    async def ainvoke(self, messages, config=None):
        return self.invoke(messages, config)

    def stream(self, messages, config=None):
        yield StreamChunk(content="test", delta="test", finish_reason="stop")

    async def astream(self, messages, config=None):
        yield StreamChunk(content="test", delta="test", finish_reason="stop")

    def count_tokens(self, text):
        return 10

    def estimate_cost(self, usage):
        return CostInfo(total_cost=0.0)


class TestBaseLLMProvider:
    """Test cases for BaseLLMProvider."""

    def test_initialization(self):
        """Test provider initialization."""
        config = ModelConfig(temperature=0.7)
        provider = ConcreteProvider(model="test-model", default_config=config)

        assert provider.model == "test-model"
        assert provider._default_config == config
        assert provider.total_usage == Usage()
        assert provider.total_cost == CostInfo()

    def test_reset_tracking(self):
        """Test resetting usage and cost tracking."""
        provider = ConcreteProvider(model="test-model")

        # Simulate some usage
        provider._total_usage = Usage(prompt_tokens=100, completion_tokens=200)
        provider._total_cost = CostInfo(input_cost=0.1, output_cost=0.2)

        provider.reset_tracking()

        assert provider.total_usage == Usage()
        assert provider.total_cost == CostInfo()

    def test_merge_config_with_none(self):
        """Test merging config with None."""
        default_config = ModelConfig(temperature=0.7, max_tokens=100)
        provider = ConcreteProvider(model="test-model", default_config=default_config)

        merged = provider._merge_config(None)

        assert merged == default_config

    def test_merge_config_with_override(self):
        """Test merging config with override values."""
        default_config = ModelConfig(temperature=0.7, max_tokens=100)
        override_config = ModelConfig(temperature=0.9, top_p=0.8)

        provider = ConcreteProvider(model="test-model", default_config=default_config)
        merged = provider._merge_config(override_config)

        assert merged.temperature == 0.9  # Override value
        assert merged.max_tokens == 100  # Default value
        assert merged.top_p == 0.8  # Override value

    def test_update_tracking(self):
        """Test updating tracking with response."""
        provider = ConcreteProvider(model="test-model")

        response = LLMResponse(
            content="test",
            model="test-model",
            usage=Usage(prompt_tokens=50, completion_tokens=30),
            cost=CostInfo(input_cost=0.05, output_cost=0.03),
        )

        provider._update_tracking(response)

        assert provider.total_usage.prompt_tokens == 50
        assert provider.total_usage.completion_tokens == 30
        assert provider.total_cost.input_cost == 0.05
        assert provider.total_cost.output_cost == 0.03

    def test_get_supported_models(self):
        """Test getting supported models."""
        models = ConcreteProvider.get_supported_models()
        assert models == []

    def test_validate_config_valid(self):
        """Test validating a valid config."""
        provider = ConcreteProvider(model="test-model")
        config = ModelConfig(temperature=0.7, top_p=0.8, max_tokens=100)

        assert provider.validate_config(config) is True

    def test_validate_config_invalid_temperature(self):
        """Test validating config with invalid temperature."""
        provider = ConcreteProvider(model="test-model")
        config = ModelConfig(temperature=3.0)  # Too high

        assert provider.validate_config(config) is False

    def test_validate_config_invalid_top_p(self):
        """Test validating config with invalid top_p."""
        provider = ConcreteProvider(model="test-model")
        config = ModelConfig(top_p=1.5)  # Too high

        assert provider.validate_config(config) is False

    def test_validate_config_invalid_max_tokens(self):
        """Test validating config with invalid max_tokens."""
        provider = ConcreteProvider(model="test-model")
        config = ModelConfig(max_tokens=0)  # Too low

        assert provider.validate_config(config) is False

    def test_validate_messages_valid(self):
        """Test validating valid messages."""
        provider = ConcreteProvider(model="test-model")
        messages = [
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ]

        assert provider.validate_messages(messages) is True

    def test_validate_messages_empty(self):
        """Test validating empty messages."""
        provider = ConcreteProvider(model="test-model")

        assert provider.validate_messages([]) is False

    def test_validate_messages_invalid_role(self):
        """Test validating messages with invalid role."""
        provider = ConcreteProvider(model="test-model")
        messages = [Message(role="invalid_role", content="Hello")]

        assert provider.validate_messages(messages) is False

    def test_validate_messages_empty_content(self):
        """Test validating messages with empty content and no tool calls."""
        provider = ConcreteProvider(model="test-model")
        messages = [Message(role=MessageRole.USER, content="")]

        assert provider.validate_messages(messages) is False

    def test_repr(self):
        """Test string representation."""
        provider = ConcreteProvider(model="test-model")
        assert repr(provider) == "ConcreteProvider(model='test-model')"


class TestMessage:
    """Test cases for Message class."""

    def test_message_creation(self):
        """Test creating messages with different roles."""
        system_msg = Message.system("System message")
        user_msg = Message.user("User message")
        assistant_msg = Message.assistant("Assistant message")
        tool_msg = Message.tool("tool_id", "Tool result")

        assert system_msg.role == "system"
        assert system_msg.content == "System message"

        assert user_msg.role == "user"
        assert user_msg.content == "User message"

        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == "Assistant message"

        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "tool_id"
        assert tool_msg.content == "Tool result"

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = Message(
            role=MessageRole.USER,
            content="Hello",
            name="test_user",
            tool_call_id="call_123",
            tool_calls=[{"id": "call_123", "function": {"name": "test"}}],
        )

        expected = {
            "role": "user",
            "content": "Hello",
            "name": "test_user",
            "tool_call_id": "call_123",
            "tool_calls": [{"id": "call_123", "function": {"name": "test"}}],
        }

        assert msg.to_dict() == expected

    def test_message_to_dict_optional_fields(self):
        """Test converting message to dictionary with optional fields omitted."""
        msg = Message(role=MessageRole.USER, content="Hello")

        expected = {"role": "user", "content": "Hello"}

        assert msg.to_dict() == expected


class TestUsage:
    """Test cases for Usage class."""

    def test_usage_creation(self):
        """Test creating usage with explicit values."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_usage_auto_total(self):
        """Test automatic total calculation."""
        usage = Usage(prompt_tokens=10, completion_tokens=20)

        assert usage.total_tokens == 30

    def test_usage_addition(self):
        """Test adding two usage objects."""
        usage1 = Usage(prompt_tokens=10, completion_tokens=20, cached_tokens=5)
        usage2 = Usage(prompt_tokens=15, completion_tokens=25, cached_tokens=3)

        result = usage1 + usage2

        assert result.prompt_tokens == 25
        assert result.completion_tokens == 45
        assert result.total_tokens == 70
        assert result.cached_tokens == 8


class TestCostInfo:
    """Test cases for CostInfo class."""

    def test_cost_info_creation(self):
        """Test creating cost info with explicit values."""
        cost = CostInfo(input_cost=0.1, output_cost=0.2, total_cost=0.3)

        assert cost.input_cost == 0.1
        assert cost.output_cost == 0.2
        assert cost.total_cost == 0.3

    def test_cost_info_auto_total(self):
        """Test automatic total calculation."""
        cost = CostInfo(input_cost=0.1, output_cost=0.2)

        assert abs(cost.total_cost - 0.3) < 1e-9

    def test_cost_info_addition(self):
        """Test adding two cost info objects."""
        cost1 = CostInfo(input_cost=0.1, output_cost=0.2, cached_input_cost=0.01)
        cost2 = CostInfo(input_cost=0.15, output_cost=0.25, cached_input_cost=0.02)

        result = cost1 + cost2

        assert abs(result.input_cost - 0.25) < 1e-9
        assert abs(result.output_cost - 0.45) < 1e-9
        assert abs(result.total_cost - 0.7) < 1e-9
        assert abs(result.cached_input_cost - 0.03) < 1e-9

    def test_cost_info_addition_different_currency(self):
        """Test adding cost info with different currencies."""
        cost1 = CostInfo(input_cost=0.1, output_cost=0.2, currency="USD")
        cost2 = CostInfo(input_cost=0.15, output_cost=0.25, currency="EUR")

        with pytest.raises(ValueError):
            cost1 + cost2

    def test_cost_info_format(self):
        """Test cost formatting."""
        cost = CostInfo(total_cost=1.234567)

        assert cost.format() == "$1.234567 USD"
        assert cost.format(precision=2) == "$1.23 USD"


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_model_config_creation(self):
        """Test creating model config with default values."""
        config = ModelConfig()

        assert config.temperature == 1.0
        assert config.max_tokens is None
        assert config.top_p == 1.0
        assert config.stream is False

    def test_model_config_to_dict(self):
        """Test converting config to dictionary."""
        config = ModelConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.8,
            stream=True,
            extra={"custom_param": "value"},
        )

        expected = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.8,
            "custom_param": "value",
        }

        assert config.to_dict() == expected

    def test_model_config_to_dict_with_none_values(self):
        """Test converting config to dictionary with default values."""
        config = ModelConfig()

        expected = {}

        assert config.to_dict() == expected


class TestLLMResponse:
    """Test cases for LLMResponse class."""

    def test_llm_response_creation(self):
        """Test creating LLM response."""
        response = LLMResponse(
            content="Hello world",
            model="test-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20),
            finish_reason="stop",
            tool_calls=[{"id": "call_123", "function": {"name": "test"}}],
        )

        assert response.content == "Hello world"
        assert response.model == "test-model"
        assert response.usage.prompt_tokens == 10
        assert response.finish_reason == "stop"
        assert response.has_tool_calls is True

    def test_llm_response_has_tool_calls(self):
        """Test has_tool_calls property."""
        response = LLMResponse(
            content="Hello", model="test-model", usage=Usage(), tool_calls=[]
        )

        assert response.has_tool_calls is False

        response.tool_calls = [{"id": "call_123", "function": {"name": "test"}}]
        assert response.has_tool_calls is True

    def test_llm_response_to_message(self):
        """Test converting response to message."""
        response = LLMResponse(
            content="Hello world",
            model="test-model",
            usage=Usage(),
            tool_calls=[{"id": "call_123", "function": {"name": "test"}}],
        )

        message = response.to_message()

        assert message.role == "assistant"
        assert message.content == "Hello world"
        assert message.tool_calls == [{"id": "call_123", "function": {"name": "test"}}]


class TestStreamChunk:
    """Test cases for StreamChunk class."""

    def test_stream_chunk_creation(self):
        """Test creating stream chunk."""
        chunk = StreamChunk(
            content="test",
            delta="test",
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=10),
            index=0,
        )

        assert chunk.content == "test"
        assert chunk.delta == "test"
        assert chunk.finish_reason == "stop"
        assert chunk.usage.prompt_tokens == 5
        assert chunk.index == 0
        assert chunk.is_final is True

    def test_stream_chunk_is_final(self):
        """Test is_final property."""
        chunk = StreamChunk(finish_reason=None)
        assert chunk.is_final is False

        chunk.finish_reason = "stop"
        assert chunk.is_final is True
