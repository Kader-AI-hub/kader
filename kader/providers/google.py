"""
Google LLM Provider implementation.

Provides synchronous and asynchronous access to Google Gemini models
via the Google Gen AI SDK.
"""

import os
from typing import AsyncIterator, Iterator

from google import genai
from google.genai import types

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

# Pricing data for Gemini models (per 1M tokens, in USD)
GEMINI_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        cached_input_cost_per_million=0.0375,
    ),
    "gemini-2.5-flash-preview-05-20": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        cached_input_cost_per_million=0.0375,
    ),
    "gemini-2.5-pro": ModelPricing(
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
        cached_input_cost_per_million=0.3125,
    ),
    "gemini-2.5-pro-preview-05-06": ModelPricing(
        input_cost_per_million=1.25,
        output_cost_per_million=10.00,
        cached_input_cost_per_million=0.3125,
    ),
    "gemini-2.0-flash": ModelPricing(
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
        cached_input_cost_per_million=0.025,
    ),
    "gemini-2.0-flash-lite": ModelPricing(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        cached_input_cost_per_million=0.01875,
    ),
    "gemini-1.5-flash": ModelPricing(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        cached_input_cost_per_million=0.01875,
    ),
    "gemini-1.5-pro": ModelPricing(
        input_cost_per_million=1.25,
        output_cost_per_million=5.00,
        cached_input_cost_per_million=0.3125,
    ),
}


class GoogleProvider(BaseLLMProvider):
    """
    Google LLM Provider.

    Provides access to Google Gemini models with full support
    for synchronous and asynchronous operations, including streaming.

    The API key is loaded from (in order of priority):
    1. The `api_key` parameter passed to the constructor
    2. The GEMINI_API_KEY environment variable (loaded from ~/.kader/.env)
    3. The GOOGLE_API_KEY environment variable

    Example:
        provider = GoogleProvider(model="gemini-2.5-flash")
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
        Initialize the Google provider.

        Args:
            model: The Gemini model identifier (e.g., "gemini-2.5-flash")
            api_key: Optional API key. If not provided, uses GEMINI_API_KEY
                     from ~/.kader/.env or GOOGLE_API_KEY environment variable.
            default_config: Default configuration for all requests
        """
        super().__init__(model=model, default_config=default_config)

        # Resolve API key: parameter > GEMINI_API_KEY > GOOGLE_API_KEY
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
            # Filter out empty strings from the .env default
            if api_key == "":
                api_key = None

        self._api_key = api_key
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[types.Content], str | None]:
        """
        Convert Message objects to Google GenAI Content format.

        Returns:
            Tuple of (contents list, system_instruction if present)
        """
        contents: list[types.Content] = []
        system_instruction: str | None = None

        for msg in messages:
            if msg.role == "system":
                # System messages are handled separately in Google's API
                system_instruction = msg.content
            elif msg.role == "user":
                contents.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=msg.content)],
                    )
                )
            elif msg.role == "assistant":
                parts: list[types.Part] = []
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        parts.append(
                            types.Part.from_function_call(
                                name=tc["function"]["name"],
                                args=tc["function"]["arguments"]
                                if isinstance(tc["function"]["arguments"], dict)
                                else {},
                            )
                        )
                contents.append(types.Content(role="model", parts=parts))
            elif msg.role == "tool":
                contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=msg.name or "tool",
                                response={"result": msg.content},
                            )
                        ],
                    )
                )

        return contents, system_instruction

    def _convert_config_to_generate_config(
        self, config: ModelConfig, system_instruction: str | None = None
    ) -> types.GenerateContentConfig:
        """Convert ModelConfig to Google GenerateContentConfig."""
        generate_config = types.GenerateContentConfig(
            temperature=config.temperature if config.temperature != 1.0 else None,
            max_output_tokens=config.max_tokens,
            top_p=config.top_p if config.top_p != 1.0 else None,
            top_k=config.top_k,
            stop_sequences=config.stop_sequences,
            system_instruction=system_instruction,
        )

        # Handle tools
        if config.tools:
            generate_config.tools = config.tools

        # Handle response format
        if config.response_format:
            resp_format_type = config.response_format.get("type")
            if resp_format_type == "json_object":
                generate_config.response_mime_type = "application/json"

        return generate_config

    def _parse_response(self, response, model: str) -> LLMResponse:
        """Parse Google GenAI response to LLMResponse."""
        # Extract content
        content = ""
        tool_calls = None

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text_parts = []
                function_calls = []

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        function_calls.append(
                            {
                                "id": f"call_{len(function_calls)}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {},
                                },
                            }
                        )

                content = "".join(text_parts)
                if function_calls:
                    tool_calls = function_calls

        # Extract usage
        usage = Usage()
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = Usage(
                prompt_tokens=getattr(response.usage_metadata, "prompt_token_count", 0)
                or 0,
                completion_tokens=getattr(
                    response.usage_metadata, "candidates_token_count", 0
                )
                or 0,
                cached_tokens=getattr(
                    response.usage_metadata, "cached_content_token_count", 0
                )
                or 0,
            )

        # Determine finish reason
        finish_reason = "stop"
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                reason = str(candidate.finish_reason).lower()
                if "stop" in reason:
                    finish_reason = "stop"
                elif "length" in reason or "max_tokens" in reason:
                    finish_reason = "length"
                elif "tool" in reason or "function" in reason:
                    finish_reason = "tool_calls"
                elif "safety" in reason or "filter" in reason:
                    finish_reason = "content_filter"

        # Calculate cost
        cost = self.estimate_cost(usage)

        return LLMResponse(
            content=content,
            model=model,
            usage=usage,
            finish_reason=finish_reason,
            cost=cost,
            tool_calls=tool_calls,
            raw_response=response,
        )

    def _parse_stream_chunk(
        self, chunk, accumulated_content: str, model: str
    ) -> StreamChunk:
        """Parse streaming chunk to StreamChunk."""
        delta = ""
        tool_calls = None

        if chunk.candidates and len(chunk.candidates) > 0:
            candidate = chunk.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        delta = part.text
                    if hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        tool_calls = [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": dict(fc.args) if fc.args else {},
                                },
                            }
                        ]

        # Extract usage from final chunk
        usage = None
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage = Usage(
                prompt_tokens=getattr(chunk.usage_metadata, "prompt_token_count", 0)
                or 0,
                completion_tokens=getattr(
                    chunk.usage_metadata, "candidates_token_count", 0
                )
                or 0,
            )

        # Determine finish reason
        finish_reason = None
        if chunk.candidates and len(chunk.candidates) > 0:
            candidate = chunk.candidates[0]
            if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                reason = str(candidate.finish_reason).lower()
                if "stop" in reason:
                    finish_reason = "stop"
                elif "length" in reason:
                    finish_reason = "length"

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
        Synchronously invoke the Google Gemini model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        contents, system_instruction = self._convert_messages(messages)
        generate_config = self._convert_config_to_generate_config(
            merged_config, system_instruction
        )

        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=generate_config,
        )

        llm_response = self._parse_response(response, self._model)
        self._update_tracking(llm_response)
        return llm_response

    def stream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Synchronously stream the Google Gemini model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        contents, system_instruction = self._convert_messages(messages)
        generate_config = self._convert_config_to_generate_config(
            merged_config, system_instruction
        )

        response_stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=generate_config,
        )

        accumulated_content = ""
        for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(
                chunk, accumulated_content, self._model
            )
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
        Asynchronously invoke the Google Gemini model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        contents, system_instruction = self._convert_messages(messages)
        generate_config = self._convert_config_to_generate_config(
            merged_config, system_instruction
        )

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=generate_config,
        )

        llm_response = self._parse_response(response, self._model)
        self._update_tracking(llm_response)
        return llm_response

    async def astream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Asynchronously stream the Google Gemini model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        contents, system_instruction = self._convert_messages(messages)
        generate_config = self._convert_config_to_generate_config(
            merged_config, system_instruction
        )

        response_stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=generate_config,
        )

        accumulated_content = ""
        async for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(
                chunk, accumulated_content, self._model
            )
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
        Count the number of tokens in the given text or messages.

        Args:
            text: A string or list of messages to count tokens for

        Returns:
            Number of tokens
        """
        try:
            if isinstance(text, str):
                response = self._client.models.count_tokens(
                    model=self._model,
                    contents=text,
                )
            else:
                contents, _ = self._convert_messages(text)
                response = self._client.models.count_tokens(
                    model=self._model,
                    contents=contents,
                )
            return getattr(response, "total_tokens", 0) or 0
        except Exception:
            # Fallback to character-based estimation
            if isinstance(text, str):
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
        # Try to find exact pricing, then fall back to base model name
        pricing = GEMINI_PRICING.get(self._model)

        if not pricing:
            # Try to match by prefix (e.g., "gemini-2.5-flash-preview" -> "gemini-2.5-flash")
            for model_prefix, model_pricing in GEMINI_PRICING.items():
                if self._model.startswith(model_prefix):
                    pricing = model_pricing
                    break

        if not pricing:
            # Default to gemini-2.5-flash pricing if unknown model
            pricing = GEMINI_PRICING.get(
                "gemini-2.5-flash",
                ModelPricing(
                    input_cost_per_million=0.15,
                    output_cost_per_million=0.60,
                ),
            )

        return pricing.calculate_cost(usage)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo | None:
        """Get information about the current model."""
        try:
            model_info = self._client.models.get(model=self._model)

            return ModelInfo(
                name=self._model,
                provider="google",
                context_window=getattr(model_info, "input_token_limit", 0) or 128000,
                max_output_tokens=getattr(model_info, "output_token_limit", None),
                pricing=GEMINI_PRICING.get(self._model),
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_vision=True,
                capabilities={
                    "display_name": getattr(model_info, "display_name", None),
                    "description": getattr(model_info, "description", None),
                },
            )
        except Exception:
            return None

    @classmethod
    def get_supported_models(cls, api_key: str | None = None) -> list[str]:
        """
        Get list of models available from Google.

        Args:
            api_key: Optional API key

        Returns:
            List of available model names that support generation
        """
        try:
            client = genai.Client(api_key=api_key) if api_key else genai.Client()
            models = []

            for model in client.models.list():
                model_name = getattr(model, "name", "")
                # Filter to only include gemini models that support generateContent
                if model_name and "gemini" in model_name.lower():
                    supported_methods = getattr(
                        model, "supported_generation_methods", []
                    )
                    if supported_methods is None:
                        supported_methods = []
                    # Include models that support content generation
                    if (
                        any("generateContent" in method for method in supported_methods)
                        or not supported_methods
                    ):
                        # Extract just the model ID from full path
                        # e.g., "models/gemini-2.5-flash" -> "gemini-2.5-flash"
                        if "/" in model_name:
                            model_name = model_name.split("/")[-1]
                        models.append(model_name)

            return models
        except Exception:
            return []

    def list_models(self) -> list[str]:
        """List all available Gemini models."""
        return self.get_supported_models(self._api_key)
