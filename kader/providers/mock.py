"""
Mock LLM Provider for testing and development.
"""

from typing import AsyncIterator, Iterator, List

from .base import (
    BaseLLMProvider,
    CostInfo,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
    Usage,
)


class MockLLM(BaseLLMProvider):
    """
    A mock LLM provider that echoes inputs or returns predefined responses.
    Useful for testing without incurring costs or latency.
    """

    def invoke(
        self,
        messages: List[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """Synchronous mock invocation."""
        last_msg = messages[-1] if messages else Message.user("")
        content = f"Mock response to: {last_msg.content}"

        usage = Usage(prompt_tokens=10, completion_tokens=10)

        return LLMResponse(
            content=content, model=self.model, usage=usage, finish_reason="stop"
        )

    async def ainvoke(
        self,
        messages: List[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """Asynchronous mock invocation."""
        import asyncio

        return await asyncio.to_thread(self.invoke, messages, config)

    def stream(
        self,
        messages: List[Message],
        config: ModelConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """Synchronous mock streaming."""
        last_msg = messages[-1] if messages else Message.user("")
        content = f"Mock response to: {last_msg.content}"
        words = content.split()

        accumulated = ""
        for i, word in enumerate(words):
            word_with_space = word + " "
            accumulated += word_with_space
            yield StreamChunk(
                content=accumulated, delta=word_with_space, index=i, finish_reason=None
            )

        yield StreamChunk(
            content=content,
            delta="",
            index=len(words),
            finish_reason="stop",
            usage=Usage(prompt_tokens=10, completion_tokens=10),
        )

    async def astream(
        self,
        messages: List[Message],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous mock streaming."""
        for chunk in self.stream(messages, config):
            yield chunk

    def count_tokens(self, text: str | List[Message]) -> int:
        """Mock token counting (1 word = 1 token)."""
        if isinstance(text, str):
            return len(text.split())

        count = 0
        for msg in text:
            count += len(msg.content.split())
        return count

    def estimate_cost(self, usage: Usage) -> CostInfo:
        """Mock cost estimation (free)."""
        return CostInfo(total_cost=0.0)
