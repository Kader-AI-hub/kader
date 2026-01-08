"""
Unit tests for the mock provider functionality.
"""

from kader.providers.base import Message, ModelConfig, Usage
from kader.providers.mock import MockLLM


class TestMockLLM:
    """Test cases for MockLLM provider."""

    def test_initialization(self):
        """Test MockLLM initialization."""
        config = ModelConfig(temperature=0.7)
        provider = MockLLM(model="test-model", default_config=config)

        assert provider.model == "test-model"
        assert provider._default_config == config

    def test_invoke(self):
        """Test synchronous invoke method."""
        provider = MockLLM(model="test-model")
        messages = [Message.user("Hello")]

        response = provider.invoke(messages)

        assert response.content == "Mock response to: Hello"
        assert response.model == "test-model"
        assert response.usage == Usage(prompt_tokens=10, completion_tokens=10)
        assert response.finish_reason == "stop"

    def test_invoke_with_empty_messages(self):
        """Test invoke with empty messages."""
        provider = MockLLM(model="test-model")

        response = provider.invoke([])

        assert response.content == "Mock response to: "
        assert response.model == "test-model"

    def test_ainvoke(self):
        """Test asynchronous invoke method."""
        import asyncio

        async def test_async():
            provider = MockLLM(model="test-model")
            messages = [Message.user("Hello")]

            response = await provider.ainvoke(messages)

            assert response.content == "Mock response to: Hello"
            assert response.model == "test-model"
            assert response.usage == Usage(prompt_tokens=10, completion_tokens=10)
            assert response.finish_reason == "stop"

        asyncio.run(test_async())

    def test_stream(self):
        """Test synchronous stream method."""
        provider = MockLLM(model="test-model")
        messages = [Message.user("Hello world")]

        chunks = list(provider.stream(messages))

        # Should have chunks for each word plus final chunk
        # "Mock", "response", "to:", "Hello", "world" -> 5 words + 1 final chunk = 6
        assert len(chunks) == 6

        # Check the first few chunks
        assert chunks[0].content == "Mock "
        assert chunks[0].delta == "Mock "

        assert chunks[1].content == "Mock response "
        assert chunks[1].delta == "response "

        # Check the final chunk
        final_chunk = chunks[-1]
        assert final_chunk.finish_reason == "stop"
        assert final_chunk.usage == Usage(prompt_tokens=10, completion_tokens=10)

    def test_stream_with_empty_messages(self):
        """Test stream with empty messages."""
        provider = MockLLM(model="test-model")

        chunks = list(provider.stream([]))

        # Should have content chunks and final chunk
        # "Mock", "response", "to:" -> 3 words + 1 final = 4
        assert len(chunks) == 4
        assert chunks[0].content == "Mock "
        assert chunks[-1].finish_reason == "stop"

    def test_astream(self):
        """Test asynchronous stream method."""
        import asyncio

        async def test_async():
            provider = MockLLM(model="test-model")
            messages = [Message.user("Hello")]

            chunks = []
            async for chunk in provider.astream(messages):
                chunks.append(chunk)

            # Should have chunks for each word plus final chunk
            assert len(chunks) >= 2  # At least one content chunk and final chunk

            # Check the final chunk
            final_chunk = chunks[-1]
            assert final_chunk.finish_reason == "stop"
            assert final_chunk.usage == Usage(prompt_tokens=10, completion_tokens=10)

        asyncio.run(test_async())

    def test_count_tokens_string(self):
        """Test counting tokens in a string."""
        provider = MockLLM(model="test-model")

        count = provider.count_tokens("Hello world test")

        assert count == 3  # 3 words

    def test_count_tokens_messages(self):
        """Test counting tokens in messages."""
        provider = MockLLM(model="test-model")
        messages = [Message.user("Hello world"), Message.assistant("Hi there")]

        count = provider.count_tokens(messages)

        assert count == 4  # "Hello world" (2) + "Hi there" (2) = 4

    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string."""
        provider = MockLLM(model="test-model")

        count = provider.count_tokens("")

        assert count == 0

    def test_estimate_cost(self):
        """Test cost estimation."""
        provider = MockLLM(model="test-model")
        usage = Usage(prompt_tokens=100, completion_tokens=50)

        cost = provider.estimate_cost(usage)

        assert cost.total_cost == 0.0  # Mock provider is free
        assert cost.input_cost == 0.0
        assert cost.output_cost == 0.0
