"""
Mistral LLM Provider implementation.

Provides synchronous and asynchronous access to Mistral AI models.
"""

import os
from typing import AsyncIterator, Iterator

from mistralai import Mistral

# Import config to ensure ~/.kader/.env is loaded
import kader.config  # noqa: F401

from .base import (
    BaseLLMProvider,
    CostInfo,
    LLMResponse,
    Message,
    ModelConfig,
    ModelInfo,
    ModelPricing,
    StreamChunk,
    Usage,
)

# Pricing data for Mistral models (per 1M tokens, in USD)
# Source: https://mistral.ai/technology/#pricing
MISTRAL_PRICING: dict[str, ModelPricing] = {
    "mistral-large-latest": ModelPricing(
        input_cost_per_million=2.0,
        output_cost_per_million=6.0,
    ),
    "mistral-large-2411": ModelPricing(
        input_cost_per_million=2.0,
        output_cost_per_million=6.0,
    ),
    "mistral-small-latest": ModelPricing(
        input_cost_per_million=0.2,
        output_cost_per_million=0.6,
    ),
    "mistral-small-2503": ModelPricing(
        input_cost_per_million=0.2,
        output_cost_per_million=0.6,
    ),
    "codestral-latest": ModelPricing(
        input_cost_per_million=0.3,
        output_cost_per_million=0.9,
    ),
    "codestral-2501": ModelPricing(
        input_cost_per_million=0.3,
        output_cost_per_million=0.9,
    ),
    "ministral-8b-latest": ModelPricing(
        input_cost_per_million=0.1,
        output_cost_per_million=0.1,
    ),
    "ministral-3b-latest": ModelPricing(
        input_cost_per_million=0.04,
        output_cost_per_million=0.04,
    ),
    "pixtral-large-latest": ModelPricing(
        input_cost_per_million=2.0,
        output_cost_per_million=6.0,
    ),
    "open-mistral-nemo": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.15,
    ),
    "open-codestral-mamba": ModelPricing(
        input_cost_per_million=0.25,
        output_cost_per_million=0.25,
    ),
}


class MistralProvider(BaseLLMProvider):
    """
    Mistral LLM Provider.

    Provides access to Mistral AI models with full support
    for synchronous and asynchronous operations, including streaming.

    The API key is loaded from (in order of priority):
    1. The `api_key` parameter passed to the constructor
    2. The MISTRAL_API_KEY environment variable (loaded from ~/.kader/.env)

    Example:
        provider = MistralProvider(model="mistral-small-latest")
        response = provider.invoke([Message.user("Hello!")])
        print(response.content)
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        default_config: ModelConfig | None = None,
    ) -> None:
        """
        Initialize the Mistral provider.

        Args:
            model: The Mistral model identifier (e.g., "mistral-small-latest")
            api_key: Optional API key. If not provided, uses MISTRAL_API_KEY
                     from ~/.kader/.env environment variable.
            default_config: Default configuration for all requests
        """
        super().__init__(model=model, default_config=default_config)

        # Resolve API key: parameter > MISTRAL_API_KEY
        if api_key is None:
            api_key = os.environ.get("MISTRAL_API_KEY")
            # Filter out empty strings from the .env default
            if api_key == "":
                api_key = None

        self._api_key = api_key
        self._client = Mistral(api_key=api_key) if api_key else Mistral()

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to Mistral format."""
        mistral_messages = []

        for msg in messages:
            mistral_msg: dict = {
                "role": msg.role,
                "content": msg.content,
            }

            # Handle tool call ID for tool messages
            if msg.role == "tool" and msg.tool_call_id:
                mistral_msg["tool_call_id"] = msg.tool_call_id

            # Handle tool calls for assistant messages
            if msg.tool_calls:
                mistral_msg["tool_calls"] = [
                    {
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                            if isinstance(tc["function"]["arguments"], str)
                            else str(tc["function"]["arguments"]),
                        },
                    }
                    for i, tc in enumerate(msg.tool_calls)
                ]

            # Handle name field if present
            if msg.name:
                mistral_msg["name"] = msg.name

            mistral_messages.append(mistral_msg)

        return mistral_messages

    def _convert_config_to_params(self, config: ModelConfig) -> dict:
        """Convert ModelConfig to Mistral API parameters."""
        params: dict = {}

        if config.temperature != 1.0:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p != 1.0:
            params["top_p"] = config.top_p
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["random_seed"] = config.seed
        if config.frequency_penalty != 0.0:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty != 0.0:
            params["presence_penalty"] = config.presence_penalty

        # Handle tools
        if config.tools:
            params["tools"] = config.tools
        if config.tool_choice is not None:
            params["tool_choice"] = config.tool_choice

        # Handle response format
        if config.response_format:
            params["response_format"] = config.response_format

        return params

    def _parse_response(self, response) -> LLMResponse:
        """Parse Mistral ChatCompletionResponse to LLMResponse."""
        # Extract content and tool calls from the first choice
        content = ""
        tool_calls = None

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message

            if message.content:
                content = message.content

            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

        # Extract usage information
        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

        # Determine finish reason
        finish_reason = "stop"
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if choice.finish_reason:
                reason = str(choice.finish_reason).lower()
                if "stop" in reason:
                    finish_reason = "stop"
                elif "length" in reason:
                    finish_reason = "length"
                elif "tool" in reason:
                    finish_reason = "tool_calls"
                elif "model_length" in reason:
                    finish_reason = "length"

        # Calculate cost
        cost = self.estimate_cost(usage)

        return LLMResponse(
            content=content,
            model=response.model or self._model,
            usage=usage,
            finish_reason=finish_reason,
            cost=cost,
            tool_calls=tool_calls,
            raw_response=response,
            id=response.id,
            created=response.created,
        )

    def _parse_stream_chunk(self, chunk, accumulated_content: str) -> StreamChunk:
        """Parse streaming chunk to StreamChunk."""
        delta = ""
        tool_calls = None
        finish_reason = None
        usage = None

        if chunk.data and chunk.data.choices and len(chunk.data.choices) > 0:
            choice = chunk.data.choices[0]

            # Extract delta content
            if choice.delta and choice.delta.content:
                delta = choice.delta.content

            # Extract tool calls from delta
            if choice.delta and choice.delta.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name if tc.function else "",
                            "arguments": tc.function.arguments if tc.function else "",
                        },
                    }
                    for tc in choice.delta.tool_calls
                ]

            # Extract finish reason
            if choice.finish_reason:
                reason = str(choice.finish_reason).lower()
                if "stop" in reason:
                    finish_reason = "stop"
                elif "length" in reason:
                    finish_reason = "length"
                elif "tool" in reason:
                    finish_reason = "tool_calls"

        # Extract usage from final chunk
        if chunk.data and chunk.data.usage:
            usage = Usage(
                prompt_tokens=chunk.data.usage.prompt_tokens or 0,
                completion_tokens=chunk.data.usage.completion_tokens or 0,
            )

        return StreamChunk(
            content=accumulated_content + delta,
            delta=delta,
            finish_reason=finish_reason,
            usage=usage,
            tool_calls=tool_calls,
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
        Synchronously invoke the Mistral model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response = self._client.chat.complete(
            model=self._model,
            messages=self._convert_messages(messages),
            **params,
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
        Synchronously stream the Mistral model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response_stream = self._client.chat.stream(
            model=self._model,
            messages=self._convert_messages(messages),
            **params,
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
                    cost=self.estimate_cost(stream_chunk.usage),
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
        Asynchronously invoke the Mistral model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response = await self._client.chat.complete_async(
            model=self._model,
            messages=self._convert_messages(messages),
            **params,
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
        Asynchronously stream the Mistral model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response_stream = await self._client.chat.stream_async(
            model=self._model,
            messages=self._convert_messages(messages),
            **params,
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
                    cost=self.estimate_cost(stream_chunk.usage),
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

        Note: Mistral doesn't provide a public tokenization API,
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
        Estimate the cost for the given token usage.

        Args:
            usage: Token usage information

        Returns:
            CostInfo with cost breakdown
        """
        # Try to find exact pricing, then fall back to pattern matching
        pricing = MISTRAL_PRICING.get(self._model)

        if not pricing:
            # Try to match by prefix (e.g., "mistral-large-2411" -> "mistral-large-latest")
            for model_pattern, model_pricing in MISTRAL_PRICING.items():
                base_name = (
                    model_pattern.replace("-latest", "")
                    .replace("-2411", "")
                    .replace("-2503", "")
                    .replace("-2501", "")
                )
                if self._model.startswith(base_name):
                    pricing = model_pricing
                    break

        if not pricing:
            # Default to mistral-small pricing if unknown model
            pricing = MISTRAL_PRICING.get(
                "mistral-small-latest",
                ModelPricing(
                    input_cost_per_million=0.2,
                    output_cost_per_million=0.6,
                ),
            )

        return pricing.calculate_cost(usage)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo | None:
        """Get information about the current model."""
        try:
            models_response = self._client.models.list()
            if models_response and models_response.data:
                for model in models_response.data:
                    if model.id == self._model:
                        return ModelInfo(
                            name=self._model,
                            provider="mistral",
                            context_window=getattr(model, "max_context_length", 32768)
                            or 32768,
                            max_output_tokens=getattr(model, "max_tokens", None),
                            pricing=MISTRAL_PRICING.get(self._model),
                            supports_tools=True,
                            supports_streaming=True,
                            supports_json_mode=True,
                            supports_vision="pixtral" in self._model.lower(),
                            capabilities={
                                "created": getattr(model, "created", None),
                                "owned_by": getattr(model, "owned_by", None),
                            },
                        )
            return None
        except Exception:
            return None

    @classmethod
    def get_supported_models(cls, api_key: str | None = None) -> list[str]:
        """
        Get list of models available from Mistral.

        Args:
            api_key: Optional API key

        Returns:
            List of available model identifiers
        """
        try:
            # Resolve API key
            if api_key is None:
                api_key = os.environ.get("MISTRAL_API_KEY")
                if api_key == "":
                    api_key = None

            client = Mistral(api_key=api_key) if api_key else Mistral()
            models_response = client.models.list()

            if models_response and models_response.data:
                return [model.id for model in models_response.data if model.id]

            return []
        except Exception:
            return []

    def list_models(self) -> list[str]:
        """List all available Mistral models."""
        return self.get_supported_models(self._api_key)
