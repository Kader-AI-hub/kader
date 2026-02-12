"""
Unit tests for the OpenAI-Compatible provider.
"""

from kader.providers.base import (
    Message,
    ModelConfig,
    Usage,
)
from kader.providers.openai_compatible import (
    MOONSHOT_PRICING,
    OPENAI_PRICING,
    OPENCODE_PRICING,
    OPENROUTER_PRICING,
    ZAI_PRICING,
    OpenAICompatibleProvider,
    OpenAIProviderConfig,
    _detect_provider,
)


class TestOpenAICompatibleProviderInit:
    """Test cases for OpenAICompatibleProvider initialization."""

    def test_initialization_with_model(self):
        """Test provider initialization with model only."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        assert provider.model == "gpt-4o"

    def test_initialization_with_config(self):
        """Test provider initialization with default config."""
        config = ModelConfig(temperature=0.7, max_tokens=100)
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            default_config=config,
        )

        assert provider.model == "gpt-4o"
        assert provider._default_config == config

    def test_initialization_with_provider_config(self):
        """Test provider initialization with provider config."""
        provider_config = OpenAIProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com/v1",
        )
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=provider_config,
        )

        assert provider.model == "gpt-4o"
        assert provider._provider_config.api_key == "test-key"
        assert provider._provider_config.base_url == "https://api.example.com/v1"

    def test_repr(self):
        """Test string representation."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        assert repr(provider) == "OpenAICompatibleProvider(model='gpt-4o')"


class TestProviderDetection:
    """Test cases for provider detection."""

    def test_detect_openai_by_model(self):
        """Test detecting OpenAI by model name."""
        assert _detect_provider(None, "gpt-4o") == "openai"
        assert _detect_provider(None, "gpt-3.5-turbo") == "openai"

    def test_detect_moonshot_by_model(self):
        """Test detecting Moonshot by model name."""
        assert _detect_provider(None, "kimi-k2.5") == "moonshot"
        assert _detect_provider(None, "moonshot-v1-8k") == "moonshot"

    def test_detect_moonshot_by_url(self):
        """Test detecting Moonshot by base URL."""
        assert _detect_provider("https://api.moonshot.cn/v1", "model") == "moonshot"

    def test_detect_zai_by_model(self):
        """Test detecting Z.ai by model name."""
        assert _detect_provider(None, "glm-5") == "zai"
        assert _detect_provider(None, "glm-4") == "zai"

    def test_detect_zai_by_url(self):
        """Test detecting Z.ai by base URL."""
        assert _detect_provider("https://api.z.ai/api/paas/v4/", "model") == "zai"

    def test_detect_openrouter_by_url(self):
        """Test detecting OpenRouter by base URL."""
        assert _detect_provider("https://openrouter.ai/api/v1", "model") == "openrouter"

    def test_detect_openrouter_by_model(self):
        """Test detecting OpenRouter by model name (contains /)."""
        assert _detect_provider(None, "anthropic/claude-3.5-sonnet") == "openrouter"
        assert _detect_provider(None, "openai/gpt-4o") == "openrouter"
        assert _detect_provider(None, "google/gemini-1.5-pro") == "openrouter"

    def test_detect_opencode_by_url(self):
        """Test detecting OpenCode by base URL."""
        assert _detect_provider("https://opencode.ai/zen/v1", "model") == "opencode"

    def test_detect_unknown(self):
        """Test unknown provider detection."""
        assert _detect_provider(None, "unknown-model") == "unknown"


class TestOpenAICompatibleProviderConvertMessages:
    """Test cases for message conversion."""

    def test_convert_simple_messages(self):
        """Test converting simple messages."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
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
        provider = OpenAICompatibleProvider(model="gpt-4o")
        messages = [
            Message.tool("call_123", "Tool result content"),
        ]

        converted = provider._convert_messages(messages)

        assert len(converted) == 1
        assert converted[0]["role"] == "tool"
        assert converted[0]["content"] == "Tool result content"
        assert converted[0]["tool_call_id"] == "call_123"


class TestOpenAICompatibleProviderConvertConfig:
    """Test cases for config conversion."""

    def test_convert_default_config(self):
        """Test converting default config."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        config = ModelConfig()

        params = provider._convert_config_to_params(config)

        # Default values should not be included
        assert "temperature" not in params
        assert "max_tokens" not in params
        assert "top_p" not in params

    def test_convert_custom_config(self):
        """Test converting custom config."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        config = ModelConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            top_k=50,
            stop_sequences=["END"],
            seed=42,
            frequency_penalty=0.5,
            presence_penalty=0.3,
        )

        params = provider._convert_config_to_params(config)

        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 100
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50
        assert params["stop"] == ["END"]
        assert params["seed"] == 42
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3

    def test_convert_config_with_tools(self):
        """Test converting config with tools."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
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
        provider = OpenAICompatibleProvider(model="gpt-4o")
        config = ModelConfig(response_format={"type": "json_object"})

        params = provider._convert_config_to_params(config)

        assert params["response_format"] == {"type": "json_object"}


class TestOpenAICompatibleProviderEstimateCost:
    """Test cases for cost estimation."""

    def test_estimate_cost_openai_model(self):
        """Test estimating cost for OpenAI model."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # gpt-4o: $2.50/M input, $10.00/M output
        expected_input = (1000 / 1_000_000) * 2.50
        expected_output = (500 / 1_000_000) * 10.00
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_moonshot_model(self):
        """Test estimating cost for Moonshot model."""
        provider = OpenAICompatibleProvider(
            model="kimi-k2.5",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://api.moonshot.cn/v1",
            ),
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # kimi-k2.5: $1.50/M input, $1.50/M output
        expected_input = (1000 / 1_000_000) * 1.50
        expected_output = (500 / 1_000_000) * 1.50
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_zai_model(self):
        """Test estimating cost for Z.ai model."""
        provider = OpenAICompatibleProvider(
            model="glm-5",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://api.z.ai/api/paas/v4/",
            ),
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # glm-5: $1.00/M input, $1.00/M output
        expected_input = (1000 / 1_000_000) * 1.00
        expected_output = (500 / 1_000_000) * 1.00
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_openrouter_model(self):
        """Test estimating cost for OpenRouter model."""
        provider = OpenAICompatibleProvider(
            model="anthropic/claude-3.5-sonnet",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # anthropic/claude-3.5-sonnet: $3.00/M input, $15.00/M output
        expected_input = (1000 / 1_000_000) * 3.00
        expected_output = (500 / 1_000_000) * 15.00
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_opencode_model(self):
        """Test estimating cost for OpenCode Zen model."""
        provider = OpenAICompatibleProvider(
            model="claude-sonnet-4-5",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        usage = Usage(prompt_tokens=1000, completion_tokens=500)

        cost = provider.estimate_cost(usage)

        # claude-sonnet-4-5: $3.00/M input, $15.00/M output
        expected_input = (1000 / 1_000_000) * 3.00
        expected_output = (500 / 1_000_000) * 15.00
        expected_total = expected_input + expected_output

        assert abs(cost.input_cost - expected_input) < 1e-9
        assert abs(cost.output_cost - expected_output) < 1e-9
        assert abs(cost.total_cost - expected_total) < 1e-9

    def test_estimate_cost_zero_usage(self):
        """Test estimating cost with zero usage."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        usage = Usage(prompt_tokens=0, completion_tokens=0)

        cost = provider.estimate_cost(usage)

        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.total_cost == 0.0


class TestOpenAICompatibleProviderCountTokens:
    """Test cases for token counting."""

    def test_count_tokens_string(self):
        """Test counting tokens in a string."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        text = "Hello, world! This is a test."

        tokens = provider.count_tokens(text)

        # Approximate: ~4 chars per token
        assert tokens == len(text) // 4

    def test_count_tokens_messages(self):
        """Test counting tokens in messages."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there"),
        ]

        tokens = provider.count_tokens(messages)

        total_chars = len("Hello") + len("Hi there")
        assert tokens == total_chars // 4


class TestOpenAICompatibleProviderModelInfo:
    """Test cases for model info."""

    def test_get_model_info_openai(self):
        """Test getting model info for OpenAI model."""
        provider = OpenAICompatibleProvider(model="gpt-4o")
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "gpt-4o"
        assert model_info.provider == "openai"
        assert model_info.supports_tools is True
        assert model_info.supports_streaming is True
        assert model_info.supports_vision is True  # gpt-4o has vision

    def test_get_model_info_moonshot(self):
        """Test getting model info for Moonshot model."""
        provider = OpenAICompatibleProvider(
            model="kimi-k2.5",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://api.moonshot.cn/v1",
            ),
        )
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "kimi-k2.5"
        assert model_info.provider == "moonshot"

    def test_get_model_info_openrouter(self):
        """Test getting model info for OpenRouter model."""
        provider = OpenAICompatibleProvider(
            model="anthropic/claude-3.5-sonnet",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://openrouter.ai/api/v1",
            ),
        )
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "anthropic/claude-3.5-sonnet"
        assert model_info.provider == "openrouter"
        assert model_info.supports_tools is True
        assert model_info.supports_streaming is True

    def test_get_model_info_opencode(self):
        """Test getting model info for OpenCode Zen model."""
        provider = OpenAICompatibleProvider(
            model="claude-sonnet-4-5",
            provider_config=OpenAIProviderConfig(
                api_key="test",
                base_url="https://opencode.ai/zen/v1",
            ),
        )
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "claude-sonnet-4-5"
        assert model_info.provider == "opencode"
        assert model_info.supports_tools is True
        assert model_info.supports_streaming is True

    def test_get_model_info_context_window(self):
        """Test context window sizes."""
        # Test GPT-4 32k
        provider_32k = OpenAICompatibleProvider(model="gpt-4-32k")
        info_32k = provider_32k.get_model_info()
        assert info_32k.context_window == 32768

        # Test model with 8k in name
        provider_8k = OpenAICompatibleProvider(model="moonshot-v1-8k")
        info_8k = provider_8k.get_model_info()
        assert info_8k.context_window == 8192


class TestOpenAIProviderConfig:
    """Test cases for OpenAIProviderConfig."""

    def test_config_creation(self):
        """Test creating config with all parameters."""
        config = OpenAIProviderConfig(
            api_key="test-key",
            base_url="https://api.example.com/v1",
            timeout=30.0,
            max_retries=5,
            default_headers={"X-Custom": "header"},
        )

        assert config.api_key == "test-key"
        assert config.base_url == "https://api.example.com/v1"
        assert config.timeout == 30.0
        assert config.max_retries == 5
        assert config.default_headers == {"X-Custom": "header"}

    def test_config_defaults(self):
        """Test config with default values."""
        config = OpenAIProviderConfig(api_key="test-key")

        assert config.api_key == "test-key"
        assert config.base_url is None
        assert config.timeout == 60.0
        assert config.max_retries == 3
        assert config.default_headers is None


class TestPricingData:
    """Test cases for pricing data."""

    def test_openai_pricing_exists(self):
        """Test that OpenAI pricing data exists for known models."""
        assert "gpt-4o" in OPENAI_PRICING
        assert "gpt-4o-mini" in OPENAI_PRICING
        assert "gpt-4-turbo" in OPENAI_PRICING
        assert "gpt-4" in OPENAI_PRICING
        assert "gpt-3.5-turbo" in OPENAI_PRICING

    def test_moonshot_pricing_exists(self):
        """Test that Moonshot pricing data exists for known models."""
        assert "kimi-k2.5" in MOONSHOT_PRICING
        assert "moonshot-v1-8k" in MOONSHOT_PRICING
        assert "moonshot-v1-32k" in MOONSHOT_PRICING
        assert "moonshot-v1-128k" in MOONSHOT_PRICING

    def test_zai_pricing_exists(self):
        """Test that Z.ai pricing data exists for known models."""
        assert "glm-5" in ZAI_PRICING
        assert "glm-4" in ZAI_PRICING
        assert "glm-4-plus" in ZAI_PRICING
        assert "glm-4-air" in ZAI_PRICING
        assert "glm-4-flash" in ZAI_PRICING

    def test_openrouter_pricing_exists(self):
        """Test that OpenRouter pricing data exists for known models."""
        assert "anthropic/claude-3.5-sonnet" in OPENROUTER_PRICING
        assert "openai/gpt-4o" in OPENROUTER_PRICING
        assert "google/gemini-1.5-pro" in OPENROUTER_PRICING
        assert "meta-llama/llama-3.3-70b-instruct" in OPENROUTER_PRICING
        assert "deepseek/deepseek-chat" in OPENROUTER_PRICING

    def test_opencode_pricing_exists(self):
        """Test that OpenCode Zen pricing data exists for known models."""
        assert "claude-sonnet-4-5" in OPENCODE_PRICING
        assert "claude-opus-4-6" in OPENCODE_PRICING
        assert "gpt-5.2" in OPENCODE_PRICING
        assert "gemini-3-pro" in OPENCODE_PRICING
        assert "kimi-k2.5" in OPENCODE_PRICING
        assert "glm-4.7" in OPENCODE_PRICING

    def test_pricing_format(self):
        """Test that pricing data has correct format."""
        for model_name, pricing in OPENAI_PRICING.items():
            assert hasattr(pricing, "input_cost_per_million")
            assert hasattr(pricing, "output_cost_per_million")
            assert pricing.input_cost_per_million >= 0
            assert pricing.output_cost_per_million >= 0
