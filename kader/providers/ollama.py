"""
Ollama LLM Provider implementation.

Provides synchronous and asynchronous access to Ollama models.
"""

from typing import AsyncIterator, Iterator

from ollama import AsyncClient, Client
from ollama._types import Options

from .base import (
    BaseLLMProvider,
    CostInfo,
    LLMResponse,
    Message,
    ModelConfig,
    ModelInfo,
    StreamChunk,
    Usage,
)


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM Provider.

    Provides access to locally-running Ollama models with full support
    for synchronous and asynchronous operations, including streaming.

    Example:
        provider = OllamaProvider(model="llama3.2")
        response = provider.invoke([Message.user("Hello!")])
        print(response.content)
    """

    def __init__(
        self,
        model: str,
        host: str | None = None,
        default_config: ModelConfig | None = None,
    ) -> None:
        """
        Initialize the Ollama provider.

        Args:
            model: The Ollama model identifier (e.g., "llama3.2", "gpt-oss:120b-cloud")
            host: Optional Ollama server host (default: http://localhost:11434)
            default_config: Default configuration for all requests
        """
        super().__init__(model=model, default_config=default_config)
        self._host = host
        self._client = Client(host=host) if host else Client()
        self._async_client = AsyncClient(host=host) if host else AsyncClient()

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to Ollama format."""
        return [msg.to_dict() for msg in messages]

    def _convert_config_to_options(self, config: ModelConfig) -> Options:
        """Convert ModelConfig to Ollama Options."""
        return Options(
            temperature=config.temperature if config.temperature != 1.0 else None,
            num_predict=config.max_tokens,
            top_p=config.top_p if config.top_p != 1.0 else None,
            top_k=config.top_k,
            frequency_penalty=config.frequency_penalty
            if config.frequency_penalty != 0.0
            else None,
            presence_penalty=config.presence_penalty
            if config.presence_penalty != 0.0
            else None,
            stop=config.stop_sequences,
            seed=config.seed,
        )

    def _parse_response(self, response) -> LLMResponse:
        """Parse Ollama ChatResponse to LLMResponse."""
        # Extract usage information
        usage = Usage(
            prompt_tokens=getattr(response, "prompt_eval_count", 0) or 0,
            completion_tokens=getattr(response, "eval_count", 0) or 0,
        )

        # Extract content from message
        content = ""
        tool_calls = None
        if hasattr(response, "message"):
            content = response.message.content or ""
            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                tool_calls = [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for i, tc in enumerate(response.message.tool_calls)
                ]

        # Determine finish reason
        finish_reason = "stop"
        if getattr(response, "done_reason", None):
            done_reason = response.done_reason
            if done_reason == "stop":
                finish_reason = "stop"
            elif done_reason == "length":
                finish_reason = "length"

        return LLMResponse(
            content=content,
            model=getattr(response, "model", self._model),
            usage=usage,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            raw_response=response,
            created=getattr(response, "created_at", None),
        )

    def _parse_stream_chunk(self, chunk, accumulated_content: str) -> StreamChunk:
        """Parse streaming chunk to StreamChunk."""
        delta = ""
        if hasattr(chunk, "message") and chunk.message.content:
            delta = chunk.message.content

        usage = None
        if getattr(chunk, "done", False):
            usage = Usage(
                prompt_tokens=getattr(chunk, "prompt_eval_count", 0) or 0,
                completion_tokens=getattr(chunk, "eval_count", 0) or 0,
            )

        finish_reason = None
        if getattr(chunk, "done", False):
            finish_reason = "stop"
            done_reason = getattr(chunk, "done_reason", None)
            if done_reason == "length":
                finish_reason = "length"

        return StreamChunk(
            content=accumulated_content + delta,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
        )

    # -------------------------------------------------------------------------
    # Synchronous Methods
    # -------------------------------------------------------------------------

    def invoke(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """
        Synchronously invoke the Ollama model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        options = self._convert_config_to_options(merged_config)

        # Handle response format properly for Ollama
        format_param = None
        if merged_config.response_format:
            resp_format_type = merged_config.response_format.get("type")
            if resp_format_type == "json_object":
                format_param = "json"
            elif resp_format_type == "text":
                format_param = ""  # Default, no special format
            # For 'object' type and other types, don't set format (Ollama doesn't support all OpenAI formats)

        response = self._client.chat(
            model=self._model,
            messages=self._convert_messages(messages),
            options=options,
            tools=merged_config.tools,
            format=format_param,
            stream=False,
        )

        llm_response = self._parse_response(response)
        self._update_tracking(llm_response)
        return llm_response

    def stream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Synchronously stream the Ollama model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        options = self._convert_config_to_options(merged_config)

        # Handle response format properly for Ollama
        format_param = None
        if merged_config.response_format:
            resp_format_type = merged_config.response_format.get("type")
            if resp_format_type == "json_object":
                format_param = "json"
            elif resp_format_type == "text":
                format_param = ""  # Default, no special format
            # For 'object' type and other types, don't set format (Ollama doesn't support all OpenAI formats)

        response_stream = self._client.chat(
            model=self._model,
            messages=self._convert_messages(messages),
            options=options,
            tools=merged_config.tools,
            format=format_param,
            stream=True,
        )

        accumulated_content = ""
        for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(chunk, accumulated_content)
            accumulated_content = stream_chunk.content
            yield stream_chunk

            # Update tracking on final chunk
            if stream_chunk.is_final and stream_chunk.usage:
                final_response = LLMResponse(
                    content=accumulated_content,
                    model=self._model,
                    usage=stream_chunk.usage,
                    finish_reason=stream_chunk.finish_reason,
                )
                self._update_tracking(final_response)

    # -------------------------------------------------------------------------
    # Asynchronous Methods
    # -------------------------------------------------------------------------

    async def ainvoke(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """
        Asynchronously invoke the Ollama model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        options = self._convert_config_to_options(merged_config)

        # Handle response format properly for Ollama
        format_param = None
        if merged_config.response_format:
            resp_format_type = merged_config.response_format.get("type")
            if resp_format_type == "json_object":
                format_param = "json"
            elif resp_format_type == "text":
                format_param = ""  # Default, no special format
            # For 'object' type and other types, don't set format (Ollama doesn't support all OpenAI formats)

        response = await self._async_client.chat(
            model=self._model,
            messages=self._convert_messages(messages),
            options=options,
            tools=merged_config.tools,
            format=format_param,
            stream=False,
        )

        llm_response = self._parse_response(response)
        self._update_tracking(llm_response)
        return llm_response

    async def astream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Asynchronously stream the Ollama model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        options = self._convert_config_to_options(merged_config)

        # Handle response format properly for Ollama
        format_param = None
        if merged_config.response_format:
            resp_format_type = merged_config.response_format.get("type")
            if resp_format_type == "json_object":
                format_param = "json"
            elif resp_format_type == "text":
                format_param = ""  # Default, no special format
            # For 'object' type and other types, don't set format (Ollama doesn't support all OpenAI formats)

        response_stream = await self._async_client.chat(
            model=self._model,
            messages=self._convert_messages(messages),
            options=options,
            tools=merged_config.tools,
            format=format_param,
            stream=True,
        )

        accumulated_content = ""
        async for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(chunk, accumulated_content)
            accumulated_content = stream_chunk.content
            yield stream_chunk

            # Update tracking on final chunk
            if stream_chunk.is_final and stream_chunk.usage:
                final_response = LLMResponse(
                    content=accumulated_content,
                    model=self._model,
                    usage=stream_chunk.usage,
                    finish_reason=stream_chunk.finish_reason,
                )
                self._update_tracking(final_response)

    # -------------------------------------------------------------------------
    # Token & Cost Methods
    # -------------------------------------------------------------------------

    def count_tokens(
        self,
        text: str | list[Message],
    ) -> int:
        """
        Estimate token count for text or messages.

        Note: Ollama doesn't provide a direct tokenization API,
        so this is an approximation based on character count.

        Args:
            text: A string or list of messages to count tokens for

        Returns:
            Estimated number of tokens (approx 4 chars per token)
        """
        if isinstance(text, str):
            # Rough estimate: ~4 characters per token
            return len(text) // 4
        else:
            total_chars = sum(len(msg.content) for msg in text)
            return total_chars // 4

    def estimate_cost(
        self,
        usage: Usage,
    ) -> CostInfo:
        """
        Estimate cost for usage.

        Note: Ollama runs locally, so there's no API cost.

        Args:
            usage: Token usage information

        Returns:
            CostInfo with zero cost (Ollama is free/local)
        """
        return CostInfo(
            input_cost=0.0,
            output_cost=0.0,
            total_cost=0.0,
            currency="USD",
        )

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo | None:
        """Get information about the current model."""
        try:
            info = self._client.show(self._model)
            return ModelInfo(
                name=self._model,
                provider="ollama",
                context_window=info.get("model_info", {}).get("context_length", 4096),
                max_output_tokens=info.get("model_info", {}).get("max_output_tokens"),
                supports_tools=True,
                supports_streaming=True,
                capabilities={
                    "family": info.get("details", {}).get("family"),
                    "parameter_size": info.get("details", {}).get("parameter_size"),
                    "quantization": info.get("details", {}).get("quantization_level"),
                },
            )
        except Exception:
            return None

    @classmethod
    def get_supported_models(cls, host: str | None = None) -> list[str]:
        """
        Get list of models available on the Ollama server.

        Args:
            host: Optional Ollama server host

        Returns:
            List of available model names
        """
        try:
            client = Client(host=host) if host else Client()
            response = client.list()
            models = [model.model for model in response.models]
            models_config = {}
            for model in models:
                models_config[model] = client.show(model)
            return [
                model
                for model, config in models_config.items()
                if config.capabilities
                in [["completion", "tools", "thinking"], ["completion", "tools"]]
            ]
        except Exception:
            return []

    def list_models(self) -> list[str]:
        """List all available models on the Ollama server."""
        return self.get_supported_models(self._host)
