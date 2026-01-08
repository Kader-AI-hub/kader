"""
Unit tests for the Ollama provider functionality.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from kader.providers.base import Message, ModelConfig, Usage
from kader.providers.ollama import OllamaProvider


class TestOllamaProvider:
    """Test cases for OllamaProvider."""

    def test_initialization(self):
        """Test OllamaProvider initialization."""
        config = ModelConfig(temperature=0.7)
        provider = OllamaProvider(
            model="llama3.2", host="http://localhost:11434", default_config=config
        )

        assert provider.model == "llama3.2"
        assert provider._host == "http://localhost:11434"
        assert provider._default_config == config

    def test_initialization_default_host(self):
        """Test OllamaProvider initialization with default host."""
        provider = OllamaProvider(model="llama3.2")

        assert provider.model == "llama3.2"
        assert provider._host is None

    @patch("kader.providers.ollama.Client")
    def test_convert_messages(self, mock_client):
        """Test converting Message objects to Ollama format."""
        provider = OllamaProvider(model="llama3.2")
        messages = [
            Message.user("Hello"),
            Message.assistant("Hi there"),
            Message.system("System message"),
        ]

        result = provider._convert_messages(messages)

        expected = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": "System message"},
        ]

        assert result == expected

    @patch("kader.providers.ollama.Client")
    def test_convert_config_to_options(self, mock_client):
        """Test converting ModelConfig to Ollama Options."""
        config = ModelConfig(
            temperature=0.7,
            max_tokens=100,
            top_p=0.8,
            top_k=40,
            frequency_penalty=0.5,
            presence_penalty=0.3,
            stop_sequences=["stop", "end"],
            seed=42,
        )

        provider = OllamaProvider(model="llama3.2")
        options = provider._convert_config_to_options(config)

        # Check that options were created with the correct values
        assert options.temperature == 0.7
        assert options.num_predict == 100
        assert options.top_p == 0.8
        assert options.top_k == 40
        assert options.frequency_penalty == 0.5
        assert options.presence_penalty == 0.3
        assert options.stop == ["stop", "end"]
        assert options.seed == 42

    @patch("kader.providers.ollama.Client")
    def test_convert_config_to_options_defaults(self, mock_client):
        """Test converting ModelConfig to Ollama Options with default values."""
        config = ModelConfig()  # All defaults

        provider = OllamaProvider(model="llama3.2")
        options = provider._convert_config_to_options(config)

        # Default values should be None to let Ollama use its defaults
        assert options.temperature is None
        assert options.num_predict is None
        assert options.top_p is None
        assert options.top_k is None
        assert options.frequency_penalty is None
        assert options.presence_penalty is None
        assert options.stop is None
        assert options.seed is None

    @patch("kader.providers.ollama.Client")
    def test_parse_response(self, mock_client):
        """Test parsing Ollama response to LLMResponse."""
        # Create a mock response object
        mock_response = Mock()
        mock_response.model = "llama3.2"
        mock_response.message = Mock()
        mock_response.message.content = "Hello from Ollama"
        mock_response.message.tool_calls = None
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20
        mock_response.done_reason = "stop"
        mock_response.created_at = 1234567890

        provider = OllamaProvider(model="llama3.2")
        llm_response = provider._parse_response(mock_response)

        assert llm_response.content == "Hello from Ollama"
        assert llm_response.model == "llama3.2"
        assert llm_response.usage.prompt_tokens == 10
        assert llm_response.usage.completion_tokens == 20
        assert llm_response.finish_reason == "stop"
        assert llm_response.created == 1234567890
        assert llm_response.tool_calls is None

    @patch("kader.providers.ollama.Client")
    def test_parse_response_with_tool_calls(self, mock_client):
        """Test parsing Ollama response with tool calls."""
        # Create mock tool call objects
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = '{"param": "value"}'

        mock_response = Mock()
        mock_response.model = "llama3.2"
        mock_response.message = Mock()
        mock_response.message.content = "Hello from Ollama"
        mock_response.message.tool_calls = [mock_tool_call]
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20
        mock_response.done_reason = "stop"

        provider = OllamaProvider(model="llama3.2")
        llm_response = provider._parse_response(mock_response)

        assert llm_response.tool_calls == [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "test_function",
                    "arguments": '{"param": "value"}',
                },
            }
        ]

    @patch("kader.providers.ollama.Client")
    def test_parse_stream_chunk(self, mock_client):
        """Test parsing streaming chunk to StreamChunk."""
        mock_chunk = Mock()
        mock_chunk.message = Mock()
        mock_chunk.message.content = "Hello"
        mock_chunk.done = False
        mock_chunk.done_reason = None

        provider = OllamaProvider(model="llama3.2")
        stream_chunk = provider._parse_stream_chunk(mock_chunk, "Previous content ")

        assert stream_chunk.content == "Previous content Hello"
        assert stream_chunk.delta == "Hello"
        assert stream_chunk.finish_reason is None
        assert stream_chunk.usage is None

    @patch("kader.providers.ollama.Client")
    def test_parse_stream_chunk_final(self, mock_client):
        """Test parsing final streaming chunk."""
        mock_chunk = Mock()
        mock_chunk.message = Mock()
        mock_chunk.message.content = ""
        mock_chunk.done = True
        mock_chunk.done_reason = "stop"
        mock_chunk.prompt_eval_count = 10
        mock_chunk.eval_count = 20

        provider = OllamaProvider(model="llama3.2")
        stream_chunk = provider._parse_stream_chunk(mock_chunk, "Final content")

        assert stream_chunk.content == "Final content"
        assert stream_chunk.delta == ""
        assert stream_chunk.finish_reason == "stop"
        assert stream_chunk.usage == Usage(prompt_tokens=10, completion_tokens=20)

    @patch("kader.providers.ollama.Client")
    def test_invoke(self, mock_client_class):
        """Test synchronous invoke method."""
        # Mock the client instance
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the response
        mock_response = Mock()
        mock_response.model = "llama3.2"
        mock_response.message = Mock()
        mock_response.message.content = "Hello from Ollama"
        mock_response.message.tool_calls = None
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20
        mock_response.done_reason = "stop"

        mock_client_instance.chat.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        messages = [Message.user("Hello")]
        config = ModelConfig(temperature=0.7)

        response = provider.invoke(messages, config)

        # Verify the client was called correctly
        mock_client_instance.chat.assert_called_once_with(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello"}],
            options=provider._convert_config_to_options(config),
            tools=None,
            format=None,
            stream=False,
        )

        assert response.content == "Hello from Ollama"
        assert response.model == "llama3.2"
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 20

    @pytest.mark.asyncio
    @patch("kader.providers.ollama.AsyncClient")
    async def test_ainvoke(self, mock_async_client_class):
        """Test asynchronous invoke method."""
        # Mock the async client instance
        mock_async_client_instance = AsyncMock()
        mock_async_client_class.return_value = mock_async_client_instance

        # Mock the response
        mock_response = Mock()
        mock_response.model = "llama3.2"
        mock_response.message = Mock()
        mock_response.message.content = "Hello from Ollama"
        mock_response.message.tool_calls = None
        mock_response.prompt_eval_count = 10
        mock_response.eval_count = 20
        mock_response.done_reason = "stop"

        mock_async_client_instance.chat.return_value = mock_response

        provider = OllamaProvider(model="llama3.2")
        messages = [Message.user("Hello")]
        config = ModelConfig(temperature=0.7)

        response = await provider.ainvoke(messages, config)

        # Verify the async client was called correctly
        mock_async_client_instance.chat.assert_called_once_with(
            model="llama3.2",
            messages=[{"role": "user", "content": "Hello"}],
            options=provider._convert_config_to_options(config),
            tools=None,
            format=None,
            stream=False,
        )

        assert response.content == "Hello from Ollama"
        assert response.model == "llama3.2"

    @patch("kader.providers.ollama.Client")
    def test_count_tokens_string(self, mock_client):
        """Test counting tokens in a string."""
        provider = OllamaProvider(model="llama3.2")

        # String length is 35 chars. 35 // 4 = 8 tokens
        count = provider.count_tokens("This is a test string with 20 chars")
        assert count == 8

    @patch("kader.providers.ollama.Client")
    def test_count_tokens_messages(self, mock_client):
        """Test counting tokens in messages."""
        provider = OllamaProvider(model="llama3.2")
        messages = [
            Message.user("Hello world"),  # 11 chars
            Message.assistant("Hi there"),  # 8 chars
        ]

        # Total chars: 11 + 8 = 19, so 19/4 = 4 tokens
        # Total chars: 11 + 8 = 19, so 19 // 4 = 4 tokens
        count = provider.count_tokens(messages)

        assert count == 4

    @patch("kader.providers.ollama.Client")
    def test_estimate_cost(self, mock_client):
        """Test cost estimation."""
        provider = OllamaProvider(model="llama3.2")
        usage = Usage(prompt_tokens=100, completion_tokens=50)

        cost = provider.estimate_cost(usage)

        # Ollama is free/local, so cost should be 0
        assert cost.total_cost == 0.0
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
        assert cost.currency == "USD"

    @patch("kader.providers.ollama.Client")
    def test_get_model_info(self, mock_client_class):
        """Test getting model information."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock the show response
        mock_client_instance.show.return_value = {
            "model_info": {"context_length": 4096, "max_output_tokens": 1024},
            "details": {
                "family": "llama",
                "parameter_size": "7B",
                "quantization_level": "Q4_0",
            },
        }

        provider = OllamaProvider(model="llama3.2")
        model_info = provider.get_model_info()

        assert model_info is not None
        assert model_info.name == "llama3.2"
        assert model_info.provider == "ollama"
        assert model_info.context_window == 4096
        assert model_info.max_output_tokens == 1024
        assert model_info.supports_tools is True
        assert model_info.supports_streaming is True
        assert model_info.capabilities["family"] == "llama"

    @patch("kader.providers.ollama.Client")
    def test_get_model_info_exception(self, mock_client_class):
        """Test getting model information when exception occurs."""
        mock_client_instance = Mock()
        mock_client_class.return_value = mock_client_instance

        # Mock an exception
        mock_client_instance.show.side_effect = Exception("API Error")

        provider = OllamaProvider(model="llama3.2")
        model_info = provider.get_model_info()

        assert model_info is None

    @patch("kader.providers.ollama.Client")
    def test_get_supported_models(self, mock_client_class):
        """Test getting supported models."""
        # Mock the client class to handle host argument
        mock_client_instance = Mock()
        mock_client_class.side_effect = lambda **kwargs: mock_client_instance

        # Mock the list response
        mock_list_response = Mock()
        mock_list_response.models = [
            Mock(model="llama3.2"),
            Mock(model="mistral"),
            Mock(model="phi3"),
        ]
        mock_client_instance.list.return_value = mock_list_response
        # Mock the show response for each model
        mock_client_instance.show.return_value = Mock(
            capabilities=["completion", "tools"]
        )

        models = OllamaProvider.get_supported_models()

        assert models == ["llama3.2", "mistral", "phi3"]

    @patch("kader.providers.ollama.Client")
    def test_get_supported_models_exception(self, mock_client_class):
        """Test getting supported models when exception occurs."""
        mock_client_instance = Mock()
        mock_client_class.side_effect = lambda **kwargs: mock_client_instance

        mock_client_instance.list.side_effect = Exception("API Error")

        models = OllamaProvider.get_supported_models()

        assert models == []

    @patch("kader.providers.ollama.Client")
    def test_list_models(self, mock_client_class):
        """Test listing models."""
        mock_client_instance = Mock()
        mock_client_class.side_effect = lambda **kwargs: mock_client_instance

        # Mock the list response
        mock_list_response = Mock()
        mock_list_response.models = [Mock(model="llama3.2")]
        mock_client_instance.list.return_value = mock_list_response
        # Mock the show response for each model
        mock_client_instance.show.return_value = Mock(
            capabilities=["completion", "tools"]
        )

        provider = OllamaProvider(model="llama3.2")
        models = provider.list_models()

        assert models == ["llama3.2"]
