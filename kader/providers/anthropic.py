"""
Anthropic LLM Provider implementation.

Provides synchronous and asynchronous access to Anthropic Claude models.
"""

import json
import os
from typing import AsyncIterator, Iterator

import anthropic as anthropic_sdk

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

# Pricing data for Anthropic Claude models (per 1M tokens, in USD)
# Source: https://www.anthropic.com/pricing
ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 4 series
    "claude-opus-4-5": ModelPricing(
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        cached_input_cost_per_million=1.875,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    # Claude 3.7 series
    "claude-sonnet-3-7": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    "claude-3-7-sonnet-20250219": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    "claude-3-5-sonnet-20240620": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        input_cost_per_million=0.80,
        output_cost_per_million=4.0,
        cached_input_cost_per_million=0.08,
    ),
    # Claude 3 series
    "claude-3-opus-20240229": ModelPricing(
        input_cost_per_million=15.0,
        output_cost_per_million=75.0,
        cached_input_cost_per_million=1.875,
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        input_cost_per_million=3.0,
        output_cost_per_million=15.0,
        cached_input_cost_per_million=0.30,
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
        cached_input_cost_per_million=0.03,
    ),
}

# Official Anthropic API base URL
# Hardcoded to prevent the SDK from reading ANTHROPIC_BASE_URL from the
# environment, which may point to a local proxy such as Ollama.
ANTHROPIC_BASE_URL = "https://api.anthropic.com/"


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM Provider.

    Provides access to Anthropic Claude models with full support
    for synchronous and asynchronous operations, including streaming
    and tool calling.

    The API key is loaded from (in order of priority):
    1. The `api_key` parameter passed to the constructor
    2. The ANTHROPIC_API_KEY environment variable (loaded from ~/.kader/.env)

    Example:
        provider = AnthropicProvider(model="claude-3-5-sonnet-20241022")
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
        Initialize the Anthropic provider.

        Args:
            model: The Anthropic model identifier (e.g., "claude-3-5-sonnet-20241022")
            api_key: Optional API key. If not provided, uses ANTHROPIC_API_KEY
                     from ~/.kader/.env environment variable.
            default_config: Default configuration for all requests
        """
        super().__init__(model=model, default_config=default_config)

        # Resolve API key: parameter > ANTHROPIC_API_KEY
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            # Filter out empty strings from the .env default
            if api_key == "":
                api_key = None

        self._api_key = api_key
        # Always pass base_url explicitly to prevent the SDK from reading
        # ANTHROPIC_BASE_URL from the environment (which may point to a local
        # proxy such as Ollama).
        self._client = anthropic_sdk.Anthropic(
            api_key=api_key, base_url=ANTHROPIC_BASE_URL
        )
        self._async_client = anthropic_sdk.AsyncAnthropic(
            api_key=api_key, base_url=ANTHROPIC_BASE_URL
        )

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[list[dict], str | None]:
        """
        Convert Message objects to Anthropic message format.

        Anthropic treats system messages separately (as a top-level param),
        and tool results go as user messages with `tool_result` content blocks.

        Returns:
            Tuple of (anthropic_messages list, system_prompt or None)
        """
        anthropic_messages: list[dict] = []
        system_prompt: str | None = None

        for msg in messages:
            if msg.role == "system":
                # System messages are passed as a separate parameter in Anthropic API
                system_prompt = msg.content
            elif msg.role == "user":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": msg.content,
                    }
                )
            elif msg.role == "assistant":
                content: list[dict] = []

                if msg.content:
                    content.append({"type": "text", "text": msg.content})

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        args = tc["function"].get("arguments", {})
                        # Anthropic expects a dict for input, not a JSON string
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except (json.JSONDecodeError, ValueError):
                                args = {}
                        content.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", f"toolu_{len(content)}"),
                                "name": tc["function"]["name"],
                                "input": args,
                            }
                        )

                anthropic_messages.append(
                    {
                        "role": "assistant",
                        "content": content if content else msg.content,
                    }
                )
            elif msg.role == "tool":
                # Tool results become user messages with tool_result content blocks
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content,
                            }
                        ],
                    }
                )

        return anthropic_messages, system_prompt

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert OpenAI-format tools to Anthropic tool format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func_def = tool["function"]
                anthropic_tools.append(
                    {
                        "name": func_def.get("name", ""),
                        "description": func_def.get("description", ""),
                        "input_schema": func_def.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )
        return anthropic_tools

    def _convert_config_to_params(
        self, config: ModelConfig, system: str | None = None
    ) -> dict:
        """Convert ModelConfig to Anthropic API parameters."""
        params: dict = {}

        # max_tokens is required by Anthropic; default to 4096
        params["max_tokens"] = config.max_tokens or 4096

        if config.temperature != 1.0:
            params["temperature"] = config.temperature
        if config.top_p != 1.0:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.stop_sequences:
            params["stop_sequences"] = config.stop_sequences
        if system:
            params["system"] = system

        # Handle tools
        if config.tools:
            params["tools"] = self._convert_tools(config.tools)
        if config.tool_choice is not None:
            # Anthropic tool_choice: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": ...}
            if isinstance(config.tool_choice, str):
                params["tool_choice"] = {"type": config.tool_choice}
            else:
                params["tool_choice"] = config.tool_choice

        return params

    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic Message response to LLMResponse."""
        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": block.input
                            if isinstance(block.input, str)
                            else json.dumps(block.input),
                        },
                    }
                )

        # Extract usage
        usage = Usage(
            prompt_tokens=getattr(response.usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(response.usage, "output_tokens", 0) or 0,
            cached_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        )

        # Determine finish reason
        finish_reason: str | None = "stop"
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "end_turn":
            finish_reason = "stop"
        elif stop_reason == "max_tokens":
            finish_reason = "length"
        elif stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif stop_reason == "stop_sequence":
            finish_reason = "stop"

        cost = self.estimate_cost(usage)

        return LLMResponse(
            content=content,
            model=getattr(response, "model", self._model),
            usage=usage,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            cost=cost,
            tool_calls=tool_calls,
            raw_response=response,
            id=getattr(response, "id", None),
        )

    def _parse_stream_chunk(
        self,
        event,
        accumulated_content: str,
        accumulated_tool_calls: list[dict],
    ) -> StreamChunk | None:
        """
        Parse a streaming event to StreamChunk.

        Returns None for events that don't produce a chunk (e.g., meta events).
        """
        delta = ""
        tool_calls = None
        finish_reason = None
        usage = None

        event_type = getattr(event, "type", None)

        if event_type == "content_block_delta":
            delta_obj = getattr(event, "delta", None)
            if delta_obj:
                delta_type = getattr(delta_obj, "type", None)
                if delta_type == "text_delta":
                    delta = getattr(delta_obj, "text", "")
                elif delta_type == "input_json_delta":
                    # Accumulate tool call input JSON - handled at message_stop
                    pass

        elif event_type == "message_delta":
            delta_obj = getattr(event, "delta", None)
            if delta_obj:
                stop_reason = getattr(delta_obj, "stop_reason", None)
                if stop_reason == "end_turn":
                    finish_reason = "stop"
                elif stop_reason == "max_tokens":
                    finish_reason = "length"
                elif stop_reason == "tool_use":
                    finish_reason = "tool_calls"
                elif stop_reason == "stop_sequence":
                    finish_reason = "stop"
            # Extract usage from message_delta
            usage_obj = getattr(event, "usage", None)
            if usage_obj:
                usage = Usage(
                    prompt_tokens=0,  # Only output usage is available in delta
                    completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
                )

        elif event_type == "message_stop":
            finish_reason = "stop"

        if accumulated_tool_calls:
            tool_calls = accumulated_tool_calls

        return StreamChunk(
            content=accumulated_content + delta,
            delta=delta,
            finish_reason=finish_reason,  # type: ignore[arg-type]
            usage=usage,
            tool_calls=tool_calls if tool_calls else None,
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
        Synchronously invoke the Anthropic Claude model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        anthropic_messages, system_prompt = self._convert_messages(messages)
        params = self._convert_config_to_params(merged_config, system=system_prompt)

        response = self._client.messages.create(
            model=self._model,
            messages=anthropic_messages,
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
        Synchronously stream the Anthropic Claude model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        anthropic_messages, system_prompt = self._convert_messages(messages)
        params = self._convert_config_to_params(merged_config, system=system_prompt)

        accumulated_content = ""
        accumulated_tool_calls: list[dict] = []
        current_tool_call: dict | None = None
        current_tool_input_json = ""

        with self._client.messages.stream(
            model=self._model,
            messages=anthropic_messages,
            **params,
        ) as stream:
            for event in stream:
                event_type = getattr(event, "type", None)

                # Handle tool call block start
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        current_tool_call = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": "",
                            },
                        }
                        current_tool_input_json = ""

                # Accumulate tool input JSON
                elif event_type == "content_block_delta":
                    delta_obj = getattr(event, "delta", None)
                    if delta_obj:
                        delta_type = getattr(delta_obj, "type", None)
                        if delta_type == "input_json_delta" and current_tool_call:
                            current_tool_input_json += getattr(
                                delta_obj, "partial_json", ""
                            )

                # Finalize tool call block
                elif event_type == "content_block_stop":
                    if current_tool_call is not None:
                        current_tool_call["function"]["arguments"] = (
                            current_tool_input_json
                        )
                        accumulated_tool_calls.append(current_tool_call)
                        current_tool_call = None
                        current_tool_input_json = ""

                chunk = self._parse_stream_chunk(
                    event, accumulated_content, accumulated_tool_calls
                )
                if chunk is None:
                    continue

                accumulated_content = chunk.content
                yield chunk

                # Update tracking on final chunk
                if chunk.is_final and chunk.usage:
                    final_response = LLMResponse(
                        content=accumulated_content,
                        model=self._model,
                        usage=chunk.usage,
                        finish_reason=chunk.finish_reason,
                        cost=self.estimate_cost(chunk.usage),
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
        Asynchronously invoke the Anthropic Claude model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        anthropic_messages, system_prompt = self._convert_messages(messages)
        params = self._convert_config_to_params(merged_config, system=system_prompt)

        response = await self._async_client.messages.create(
            model=self._model,
            messages=anthropic_messages,
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
        Asynchronously stream the Anthropic Claude model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        anthropic_messages, system_prompt = self._convert_messages(messages)
        params = self._convert_config_to_params(merged_config, system=system_prompt)

        accumulated_content = ""
        accumulated_tool_calls: list[dict] = []
        current_tool_call: dict | None = None
        current_tool_input_json = ""

        async with self._async_client.messages.stream(
            model=self._model,
            messages=anthropic_messages,
            **params,
        ) as stream:
            async for event in stream:
                event_type = getattr(event, "type", None)

                # Handle tool call block start
                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block and getattr(block, "type", None) == "tool_use":
                        current_tool_call = {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": "",
                            },
                        }
                        current_tool_input_json = ""

                # Accumulate tool input JSON
                elif event_type == "content_block_delta":
                    delta_obj = getattr(event, "delta", None)
                    if delta_obj:
                        delta_type = getattr(delta_obj, "type", None)
                        if delta_type == "input_json_delta" and current_tool_call:
                            current_tool_input_json += getattr(
                                delta_obj, "partial_json", ""
                            )

                # Finalize tool call block
                elif event_type == "content_block_stop":
                    if current_tool_call is not None:
                        current_tool_call["function"]["arguments"] = (
                            current_tool_input_json
                        )
                        accumulated_tool_calls.append(current_tool_call)
                        current_tool_call = None
                        current_tool_input_json = ""

                chunk = self._parse_stream_chunk(
                    event, accumulated_content, accumulated_tool_calls
                )
                if chunk is None:
                    continue

                accumulated_content = chunk.content
                yield chunk

                # Update tracking on final chunk
                if chunk.is_final and chunk.usage:
                    final_response = LLMResponse(
                        content=accumulated_content,
                        model=self._model,
                        usage=chunk.usage,
                        finish_reason=chunk.finish_reason,
                        cost=self.estimate_cost(chunk.usage),
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

        Uses the Anthropic token counting API (client.beta.messages.count_tokens)
        when possible, with a character-based fallback.

        Args:
            text: A string or list of messages to count tokens for

        Returns:
            Number of tokens
        """
        try:
            if isinstance(text, str):
                response = self._client.beta.messages.count_tokens(
                    model=self._model,
                    messages=[{"role": "user", "content": text}],
                    betas=["token-counting-2024-11-01"],
                )
            else:
                anthropic_messages, system = self._convert_messages(text)
                kwargs: dict = {
                    "model": self._model,
                    "messages": anthropic_messages,
                    "betas": ["token-counting-2024-11-01"],
                }
                if system:
                    kwargs["system"] = system
                response = self._client.beta.messages.count_tokens(**kwargs)
            return getattr(response, "input_tokens", 0) or 0
        except Exception:
            # Fallback to character-based approximation
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
        pricing = ANTHROPIC_PRICING.get(self._model)

        if not pricing:
            # Try prefix matching (e.g., "claude-3-5-sonnet" prefix)
            for model_key, model_pricing in ANTHROPIC_PRICING.items():
                if self._model.startswith(model_key) or model_key.startswith(
                    self._model
                ):
                    pricing = model_pricing
                    break

        if not pricing:
            # Default to Claude 3.5 Sonnet pricing if unknown model
            pricing = ANTHROPIC_PRICING.get(
                "claude-3-5-sonnet-20241022",
                ModelPricing(
                    input_cost_per_million=3.0,
                    output_cost_per_million=15.0,
                ),
            )

        return pricing.calculate_cost(usage)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo | None:
        """Get information about the current model."""
        try:
            model_info = self._client.models.retrieve(self._model)
            return ModelInfo(
                name=self._model,
                provider="anthropic",
                context_window=200000,  # Claude models support up to 200K context
                pricing=ANTHROPIC_PRICING.get(self._model),
                supports_tools=True,
                supports_streaming=True,
                supports_json_mode=False,  # Anthropic uses tool_use for structured output
                supports_vision=True,
                capabilities={
                    "display_name": getattr(model_info, "display_name", None),
                    "created_at": getattr(model_info, "created_at", None),
                    "type": getattr(model_info, "type", None),
                },
            )
        except Exception:
            return None

    @classmethod
    def get_supported_models(cls, api_key: str | None = None) -> list[str]:
        """
        Get list of models available from Anthropic using the models API.

        Args:
            api_key: Optional API key

        Returns:
            List of available model identifiers
        """
        # Resolve API key from environment if not provided
        if api_key is None:
            import kader.config  # noqa: F401

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:  # catches both None and empty string
                return []

        try:
            # Always pass base_url explicitly to avoid SDK env auto-discovery
            client = anthropic_sdk.Anthropic(
                api_key=api_key, base_url=ANTHROPIC_BASE_URL
            )

            models = []
            for model in client.models.list():
                model_id = getattr(model, "id", None)
                if model_id:
                    models.append(model_id)

            return models
        except Exception:
            return []

    def list_models(self) -> list[str]:
        """List all available Anthropic models."""
        return self.get_supported_models(self._api_key)
