"""
Unit tests for the Google provider functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from kader.providers.base import Message, ModelConfig, Usage
from kader.providers.google import GEMINI_PRICING, GoogleProvider


class TestGoogleProvider:
    """Test cases for GoogleProvider."""

    @patch("kader.providers.google.genai.Client")
    def test_initialization(self, mock_client):
        """Test GoogleProvider initialization."""
        config = ModelConfig(temperature=0.7)
        provider = GoogleProvider(
            model="gemini-2.5-flash", api_key="test-api-key", default_config=config
        )

        assert provider.model == "gemini-2.5-flash"
        assert provider._api_key == "test-api-key"
        assert provider._default_config == config

    @patch.dict("os.environ", {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=False)
    @patch("kader.providers.google.genai.Client")
    def test_initialization_default_api_key(self, mock_client):
        """Test GoogleProvider initialization with default API key."""
        provider = GoogleProvider(model="gemini-2.5-flash")

        assert provider.model == "gemini-2.5-flash"
        assert provider._api_key is None

    @patch("kader.providers.google.genai.Client")
    def test_convert_messages_user(self, mock_client):
        """Test converting user Message objects to Google format."""
        provider = GoogleProvider(model="gemini-2.5-flash")
        messages = [Message.user("Hello")]

        contents, system_instruction = provider._convert_messages(messages)

        assert len(contents) == 1
        assert contents[0].role == "user"
        assert system_instruction is None

    @patch("kader.providers.google.genai.Client")
    def test_convert_messages_with_system(self, mock_client):
        """Test converting messages with system instruction."""
        provider = GoogleProvider(model="gemini-2.5-flash")
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello"),
        ]

        contents, system_instruction = provider._convert_messages(messages)

        assert len(contents) == 1  # Only user message in contents
        assert system_instruction == "You are a helpful assistant."

    @patch("kader.providers.google.genai.Client")
    def test_convert_messages_assistant(self, mock_client):
        """Test converting assistant Message objects to Google format."""
        provider = GoogleProvider(model="gemini-2.5-flash")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there!"),
        ]

        contents, system_instruction = provider._convert_messages(messages)

        assert len(contents) == 2
        assert contents[0].role == "user"
        assert contents[1].role == "model"  # Google uses "model" role for assistant

    @patch("kader.providers.google.genai.Client")
    def test_convert_config_to_generate_config(self, mock_client):
        """Test converting ModelConfig to GenerateContentConfig."""
        config = ModelConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.8,
            top_k=40,
            stop_sequences=["stop", "end"],
        )

        provider = GoogleProvider(model="gemini-2.5-flash")
        generate_config = provider._convert_config_to_generate_config(
            config, system_instruction="Be helpful"
        )

        assert generate_config.temperature == 0.7
        assert generate_config.max_output_tokens == 100
        assert generate_config.top_p == 0.8
        assert generate_config.top_k == 40
        assert generate_config.stop_sequences == ["stop", "end"]
        assert generate_config.system_instruction == "Be helpful"

    @patch("kader.providers.google.genai.Client")
    def test_convert_config_to_generate_config_defaults(self, mock_client):
        """Test converting ModelConfig with defaults."""
        config = ModelConfig()  # All defaults

        provider = GoogleProvider(model="gemini-2.5-flash")
        generate_config = provider._convert_config_to_generate_config(config)

        # Default temperature 1.0 should result in None
        assert generate_config.temperature is None
        assert generate_config.max_output_tokens is None
        assert generate_config.top_p is None

    @patch("kader.providers.google.genai.Client")
    def test_parse_response(self, mock_client):
        """Test parsing Google response to LLMResponse."""
        # Create a mock response object
        mock_part = Mock()
        mock_part.text = "Hello from Gemini"
        mock_part.function_call = None

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.cached_content_token_count = 0

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        provider = GoogleProvider(model="gemini-2.5-flash")
        llm_response = provider._parse_response(mock_response, "gemini-2.5-flash")

        assert llm_response.content == "Hello from Gemini"
        assert llm_response.model == "gemini-2.5-flash"
        assert llm_response.usage.prompt_tokens == 10
        assert llm_response.usage.completion_tokens == 20
        assert llm_response.finish_reason == "stop"
        assert llm_response.tool_calls is None

    @patch("kader.providers.google.genai.Client")
    def test_parse_response_with_tool_calls(self, mock_client):
        """Test parsing Google response with function calls."""
        # Create mock function call
        mock_function_call = Mock()
        mock_function_call.name = "get_weather"
        mock_function_call.args = {"location": "Boston"}

        mock_part = Mock()
        mock_part.text = ""
        mock_part.function_call = mock_function_call

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "FUNCTION_CALL"

        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 5
        mock_usage.cached_content_token_count = 0

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        provider = GoogleProvider(model="gemini-2.5-flash")
        llm_response = provider._parse_response(mock_response, "gemini-2.5-flash")

        assert llm_response.tool_calls is not None
        assert len(llm_response.tool_calls) == 1
        assert llm_response.tool_calls[0]["function"]["name"] == "get_weather"
        assert llm_response.tool_calls[0]["function"]["arguments"] == {
            "location": "Boston"
        }

    @patch("kader.providers.google.genai.Client")
    def test_parse_stream_chunk(self, mock_client):
        """Test parsing streaming chunk to StreamChunk."""
        mock_part = Mock()
        mock_part.text = "Hello"
        mock_part.function_call = None

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = None

        mock_chunk = Mock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = None

        provider = GoogleProvider(model="gemini-2.5-flash")
        stream_chunk = provider._parse_stream_chunk(
            mock_chunk, "Previous content ", "gemini-2.5-flash"
        )

        assert stream_chunk.content == "Previous content Hello"
        assert stream_chunk.delta == "Hello"
        assert stream_chunk.finish_reason is None
        assert stream_chunk.usage is None

    @patch("kader.providers.google.genai.Client")
    def test_parse_stream_chunk_final(self, mock_client):
        """Test parsing final streaming chunk."""
        mock_part = Mock()
        mock_part.text = ""
        mock_part.function_call = None

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_chunk = Mock()
        mock_chunk.candidates = [mock_candidate]
        mock_chunk.usage_metadata = mock_usage

        provider = GoogleProvider(model="gemini-2.5-flash")
        stream_chunk = provider._parse_stream_chunk(
            mock_chunk, "Final content", "gemini-2.5-flash"
        )

        assert stream_chunk.content == "Final content"
        assert stream_chunk.delta == ""
        assert stream_chunk.finish_reason == "stop"
        assert stream_chunk.usage == Usage(prompt_tokens=10, completion_tokens=20)

    @patch("kader.providers.google.genai.Client")
    def test_invoke(self, mock_client_class):
        """Test synchronous invoke method."""
        # Mock the client instance
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the response
        mock_part = Mock()
        mock_part.text = "Hello from Gemini"
        mock_part.function_call = None

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.cached_content_token_count = 0

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        mock_client_instance.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.5-flash")
        messages = [Message.user("Hello")]
        config = ModelConfig(temperature=0.7)

        response = provider.invoke(messages, config)

        # Verify the client was called
        mock_client_instance.models.generate_content.assert_called_once()

        assert response.content == "Hello from Gemini"
        assert response.model == "gemini-2.5-flash"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20

    @pytest.mark.asyncio
    @patch("kader.providers.google.genai.Client")
    async def test_ainvoke(self, mock_client_class):
        """Test asynchronous invoke method."""
        # Mock the client instance
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the async client
        mock_aio = AsyncMock()
        mock_client_instance.aio = mock_aio

        # Mock the response
        mock_part = Mock()
        mock_part.text = "Hello from Gemini"
        mock_part.function_call = None

        mock_content = Mock()
        mock_content.parts = [mock_part]

        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_candidate.finish_reason = "STOP"

        mock_usage = Mock()
        mock_usage.prompt_token_count = 10
        mock_usage.candidates_token_count = 20
        mock_usage.cached_content_token_count = 0

        mock_response = Mock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        mock_aio.models.generate_content.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.5-flash")
        messages = [Message.user("Hello")]
        config = ModelConfig(temperature=0.7)

        response = await provider.ainvoke(messages, config)

        # Verify the async client was called
        mock_aio.models.generate_content.assert_called_once()

        assert response.content == "Hello from Gemini"
        assert response.model == "gemini-2.5-flash"

    @patch("kader.providers.google.genai.Client")
    def test_count_tokens_string(self, mock_client_class):
        """Test counting tokens in a string."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        mock_response = Mock()
        mock_response.total_tokens = 10
        mock_client_instance.models.count_tokens.return_value = mock_response

        provider = GoogleProvider(model="gemini-2.5-flash")
        count = provider.count_tokens("This is a test string")

        assert count == 10

    @patch("kader.providers.google.genai.Client")
    def test_count_tokens_fallback(self, mock_client_class):
        """Test counting tokens with fallback estimation."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Simulate API failure
        mock_client_instance.models.count_tokens.side_effect = Exception("API Error")

        provider = GoogleProvider(model="gemini-2.5-flash")
        # 36 chars // 4 = 9 tokens
        count = provider.count_tokens("This is a test string for counting")

        assert count == 8  # 35 chars // 4 = 8

    @patch("kader.providers.google.genai.Client")
    def test_estimate_cost(self, mock_client):
        """Test cost estimation."""
        provider = GoogleProvider(model="gemini-2.5-flash")
        usage = Usage(prompt_tokens=1000000, completion_tokens=500000)

        cost = provider.estimate_cost(usage)

        # gemini-2.5-flash: $0.15 per 1M input, $0.60 per 1M output
        # Expected: (1M / 1M) * 0.15 + (0.5M / 1M) * 0.60 = 0.15 + 0.30 = 0.45
        assert cost.input_cost == pytest.approx(0.15)
        assert cost.output_cost == pytest.approx(0.30)
        assert cost.total_cost == pytest.approx(0.45)
        assert cost.currency == "USD"

    @patch("kader.providers.google.genai.Client")
    def test_estimate_cost_unknown_model(self, mock_client):
        """Test cost estimation for unknown model (uses default pricing)."""
        provider = GoogleProvider(model="gemini-future-model")
        usage = Usage(prompt_tokens=1000000, completion_tokens=1000000)

        cost = provider.estimate_cost(usage)

        # Should use default gemini-2.5-flash pricing
        assert cost.total_cost > 0
        assert cost.currency == "USD"

    @patch("kader.providers.google.genai.Client")
    def test_get_model_info(self, mock_client_class):
        """Test getting model information."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the get response
        mock_model_info = Mock()
        mock_model_info.input_token_limit = 1048576
        mock_model_info.output_token_limit = 8192
        mock_model_info.display_name = "Gemini 2.5 Flash"
        mock_model_info.description = "Fast and efficient model"

        mock_client_instance.models.get.return_value = mock_model_info

        provider = GoogleProvider(model="gemini-2.5-flash")
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "gemini-2.5-flash"
        assert model_info.provider == "google"
        assert model_info.context_window == 1048576
        assert model_info.max_output_tokens == 8192
        assert model_info.supports_tools is True
        assert model_info.supports_streaming is True

    @patch("kader.providers.google.genai.Client")
    def test_get_model_info_exception(self, mock_client_class):
        """Test getting model information when exception occurs."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock an exception
        mock_client_instance.models.get.side_effect = Exception("API Error")

        provider = GoogleProvider(model="gemini-2.5-flash")
        model_info = provider.get_model_info()

        assert model_info is None

    @patch("kader.providers.google.genai.Client")
    def test_get_supported_models(self, mock_client_class):
        """Test getting supported models dynamically."""
        # Mock the client class
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the list response
        mock_model_1 = Mock()
        mock_model_1.name = "models/gemini-2.5-flash"
        mock_model_1.supported_generation_methods = ["generateContent"]

        mock_model_2 = Mock()
        mock_model_2.name = "models/gemini-2.5-pro"
        mock_model_2.supported_generation_methods = ["generateContent"]

        mock_model_3 = Mock()
        mock_model_3.name = "models/embedding-001"
        mock_model_3.supported_generation_methods = ["embedContent"]

        mock_client_instance.models.list.return_value = [
            mock_model_1,
            mock_model_2,
            mock_model_3,
        ]

        models = GoogleProvider.get_supported_models()

        # Should include gemini models but not embedding model
        assert "gemini-2.5-flash" in models
        assert "gemini-2.5-pro" in models
        assert "embedding-001" not in models

    @patch("kader.providers.google.genai.Client")
    def test_get_supported_models_exception(self, mock_client_class):
        """Test getting supported models when exception occurs."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        mock_client_instance.models.list.side_effect = Exception("API Error")

        models = GoogleProvider.get_supported_models()

        assert models == []

    @patch("kader.providers.google.genai.Client")
    def test_list_models(self, mock_client_class):
        """Test listing models via instance method."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the list response
        mock_model = Mock()
        mock_model.name = "models/gemini-2.5-flash"
        mock_model.supported_generation_methods = ["generateContent"]

        mock_client_instance.models.list.return_value = [mock_model]

        provider = GoogleProvider(model="gemini-2.5-flash", api_key="test-key")
        models = provider.list_models()

        assert "gemini-2.5-flash" in models

    def test_gemini_pricing_data(self):
        """Test that pricing data is available for common models."""
        assert "gemini-2.5-flash" in GEMINI_PRICING
        assert "gemini-2.5-pro" in GEMINI_PRICING
        assert "gemini-2.0-flash" in GEMINI_PRICING

        flash_pricing = GEMINI_PRICING["gemini-2.5-flash"]
        assert flash_pricing.input_cost_per_million == 0.15
        assert flash_pricing.output_cost_per_million == 0.60
