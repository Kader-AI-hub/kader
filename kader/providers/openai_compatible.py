"""
OpenAI-Compatible LLM Provider implementation.

Provides synchronous and asynchronous access to OpenAI-compatible LLM providers
including OpenAI, Moonshot AI (kimi-k2.5), Z.ai (GLM-5), OpenRouter, OpenCode Zen,
and other providers that implement the OpenAI API specification.
"""

import os
from dataclasses import dataclass
from typing import AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI

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


@dataclass
class OpenAIProviderConfig:
    """Configuration for OpenAI-compatible LLM providers.

    This dataclass allows configuring different OpenAI-compatible providers
    (e.g., Moonshot AI, Z.ai) with their specific API endpoints.

    Example:
        # Z.AI configuration
        config = OpenAIProviderConfig(
            api_key="your-api-key",
            base_url="https://api.z.ai/api/paas/v4/",
        )

        # Moonshot AI configuration
        config = OpenAIProviderConfig(
            api_key="your-api-key",
            base_url="https://api.moonshot.cn/v1",
        )
    """

    api_key: str
    base_url: str | None = None
    timeout: float | None = 60.0
    max_retries: int = 3
    default_headers: dict[str, str] | None = None


# Pricing data for OpenAI models (per 1M tokens, in USD)
# Source: https://openai.com/pricing
OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o models
    "gpt-4o": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-2024-11-20": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-4o-mini": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    "gpt-4o-mini-2024-07-18": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    # GPT-4 Turbo models
    "gpt-4-turbo": ModelPricing(
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
    ),
    "gpt-4-turbo-2024-04-09": ModelPricing(
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
    ),
    "gpt-4": ModelPricing(
        input_cost_per_million=30.00,
        output_cost_per_million=60.00,
    ),
    "gpt-4-0613": ModelPricing(
        input_cost_per_million=30.00,
        output_cost_per_million=60.00,
    ),
    "gpt-4-32k": ModelPricing(
        input_cost_per_million=60.00,
        output_cost_per_million=120.00,
    ),
    # GPT-3.5 Turbo models
    "gpt-3.5-turbo": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
    ),
    "gpt-3.5-turbo-0125": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
    ),
    "gpt-3.5-turbo-1106": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=2.00,
    ),
}

# Pricing data for Moonshot AI models (per 1M tokens, in USD)
# Source: https://platform.moonshot.cn/docs/pricing
MOONSHOT_PRICING: dict[str, ModelPricing] = {
    "moonshot-v1-8k": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
    ),
    "moonshot-v1-32k": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    "moonshot-v1-128k": ModelPricing(
        input_cost_per_million=2.00,
        output_cost_per_million=2.00,
    ),
    "kimi-k2.5": ModelPricing(
        input_cost_per_million=1.50,
        output_cost_per_million=1.50,
    ),
}

# Pricing data for Z.ai models (GLM series) (per 1M tokens, in USD)
# Source: https://open.bigmodel.cn/pricing
ZAI_PRICING: dict[str, ModelPricing] = {
    "glm-5": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    "glm-4": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
    ),
    "glm-4-plus": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    "glm-4-air": ModelPricing(
        input_cost_per_million=0.10,
        output_cost_per_million=0.10,
    ),
    "glm-4-flash": ModelPricing(
        input_cost_per_million=0.01,
        output_cost_per_million=0.01,
    ),
}

# Pricing data for OpenRouter models (per 1M tokens, in USD)
# Source: https://openrouter.ai/docs#pricing
# Note: OpenRouter adds a small fee on top of base provider pricing
OPENROUTER_PRICING: dict[str, ModelPricing] = {
    # Anthropic models via OpenRouter
    "anthropic/claude-3.5-sonnet": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "anthropic/claude-3.5-haiku": ModelPricing(
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
    ),
    "anthropic/claude-3-opus": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    "anthropic/claude-3-sonnet": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "anthropic/claude-3-haiku": ModelPricing(
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
    ),
    # OpenAI models via OpenRouter
    "openai/gpt-4o": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "openai/gpt-4o-mini": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    "openai/gpt-4-turbo": ModelPricing(
        input_cost_per_million=10.00,
        output_cost_per_million=30.00,
    ),
    "openai/gpt-4": ModelPricing(
        input_cost_per_million=30.00,
        output_cost_per_million=60.00,
    ),
    "openai/gpt-3.5-turbo": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=1.50,
    ),
    # Google models via OpenRouter
    "google/gemini-2.0-flash-001": ModelPricing(
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
    ),
    "google/gemini-2.0-flash-lite-001": ModelPricing(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
    "google/gemini-2.0-pro-exp-02-05:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    "google/gemini-1.5-pro": ModelPricing(
        input_cost_per_million=1.25,
        output_cost_per_million=5.00,
    ),
    "google/gemini-1.5-flash": ModelPricing(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
    # Meta models via OpenRouter
    "meta-llama/llama-3.3-70b-instruct": ModelPricing(
        input_cost_per_million=0.12,
        output_cost_per_million=0.30,
    ),
    "meta-llama/llama-3.2-3b-instruct:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    "meta-llama/llama-3.1-405b-instruct": ModelPricing(
        input_cost_per_million=0.80,
        output_cost_per_million=0.80,
    ),
    "meta-llama/llama-3.1-70b-instruct": ModelPricing(
        input_cost_per_million=0.18,
        output_cost_per_million=0.18,
    ),
    # Mistral models via OpenRouter
    "mistralai/mistral-large": ModelPricing(
        input_cost_per_million=2.00,
        output_cost_per_million=6.00,
    ),
    "mistralai/mistral-small": ModelPricing(
        input_cost_per_million=0.20,
        output_cost_per_million=0.60,
    ),
    # DeepSeek models via OpenRouter
    "deepseek/deepseek-chat": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=2.00,
    ),
    "deepseek/deepseek-r1": ModelPricing(
        input_cost_per_million=0.55,
        output_cost_per_million=2.19,
    ),
    "deepseek/deepseek-r1:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    # Qwen models via OpenRouter
    "qwen/qwen-2.5-72b-instruct": ModelPricing(
        input_cost_per_million=0.35,
        output_cost_per_million=0.40,
    ),
    "qwen/qwen-2.5-7b-instruct": ModelPricing(
        input_cost_per_million=0.10,
        output_cost_per_million=0.10,
    ),
    # Cohere models via OpenRouter
    "cohere/command-r-plus": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "cohere/command-r": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    # Nous models via OpenRouter
    "nousresearch/hermes-3-llama-3.1-405b:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    "nousresearch/deephermes-3-mistral-24b-preview:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    # Microsoft models via OpenRouter
    "microsoft/phi-4": ModelPricing(
        input_cost_per_million=0.07,
        output_cost_per_million=0.14,
    ),
    # Hugging Face models via OpenRouter
    "huggingfaceh4/zephyr-7b-beta:free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
}

# Pricing data for OpenCode Zen models (per 1M tokens, in USD)
# Source: https://opencode.ai/zen/pricing
OPENCODE_PRICING: dict[str, ModelPricing] = {
    # Claude models
    "claude-opus-4-6": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    "claude-opus-4-5": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    "claude-opus-4-1": ModelPricing(
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
    ),
    "claude-sonnet-4": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-sonnet-4-5": ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "claude-3-5-haiku": ModelPricing(
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
    ),
    "claude-haiku-4-5": ModelPricing(
        input_cost_per_million=0.25,
        output_cost_per_million=1.25,
    ),
    # Gemini models
    "gemini-3-pro": ModelPricing(
        input_cost_per_million=1.25,
        output_cost_per_million=5.00,
    ),
    "gemini-3-flash": ModelPricing(
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
    ),
    # GPT models
    "gpt-5.2": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-5.2-codex": ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    ),
    "gpt-5.1": ModelPricing(
        input_cost_per_million=2.00,
        output_cost_per_million=8.00,
    ),
    "gpt-5.1-codex-max": ModelPricing(
        input_cost_per_million=2.00,
        output_cost_per_million=8.00,
    ),
    "gpt-5.1-codex": ModelPricing(
        input_cost_per_million=1.50,
        output_cost_per_million=6.00,
    ),
    "gpt-5.1-codex-mini": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=2.00,
    ),
    "gpt-5": ModelPricing(
        input_cost_per_million=1.50,
        output_cost_per_million=6.00,
    ),
    "gpt-5-codex": ModelPricing(
        input_cost_per_million=1.50,
        output_cost_per_million=6.00,
    ),
    "gpt-5-nano": ModelPricing(
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
    ),
    # GLM models
    "glm-4.7": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    "glm-4.6": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
    ),
    "glm-4.7-free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    # MiniMax models
    "minimax-m2.1": ModelPricing(
        input_cost_per_million=0.15,
        output_cost_per_million=0.15,
    ),
    "minimax-m2.1-free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    # Kimi models
    "kimi-k2.5": ModelPricing(
        input_cost_per_million=1.50,
        output_cost_per_million=1.50,
    ),
    "kimi-k2.5-free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    "kimi-k2": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    "kimi-k2-thinking": ModelPricing(
        input_cost_per_million=1.00,
        output_cost_per_million=1.00,
    ),
    # Other models
    "trinity-large-preview-free": ModelPricing(
        input_cost_per_million=0.00,
        output_cost_per_million=0.00,
    ),
    "big-pickle": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
    ),
    "alpha-g5": ModelPricing(
        input_cost_per_million=0.50,
        output_cost_per_million=0.50,
    ),
}

# Combine all pricing data
PROVIDER_PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "moonshot": MOONSHOT_PRICING,
    "zai": ZAI_PRICING,
    "openrouter": OPENROUTER_PRICING,
    "opencode": OPENCODE_PRICING,
}


def _detect_provider(base_url: str | None, model: str) -> str:
    """Detect the provider based on base_url or model name.

    Args:
        base_url: The API base URL
        model: The model identifier

    Returns:
        Provider identifier ("openai", "moonshot", "zai", "openrouter", "opencode", or "unknown")
    """
    if base_url:
        base_url_lower = base_url.lower()
        if "moonshot" in base_url_lower or "moonshot.cn" in base_url_lower:
            return "moonshot"
        elif "z.ai" in base_url_lower or "bigmodel" in base_url_lower:
            return "zai"
        elif "openrouter" in base_url_lower:
            return "openrouter"
        elif "opencode" in base_url_lower:
            return "opencode"

    # Try to detect from model name
    model_lower = model.lower()
    if model_lower.startswith("kimi") or model_lower.startswith("moonshot"):
        return "moonshot"
    elif model_lower.startswith("glm"):
        return "zai"
    elif "/" in model_lower:
        # OpenRouter uses provider/model format (e.g., "anthropic/claude-3.5-sonnet")
        return "openrouter"
    elif model_lower.startswith("gpt"):
        return "openai"

    return "unknown"


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI-Compatible LLM Provider.

    Provides access to OpenAI-compatible LLM providers including:
    - OpenAI (GPT-4, GPT-4o, GPT-3.5 Turbo)
    - Moonshot AI (kimi-k2.5, moonshot-v1 series)
    - Z.ai (GLM-5, GLM-4 series)
    - OpenRouter (access to 200+ models from various providers)
    - OpenCode Zen (Claude, Gemini, GPT, GLM, Kimi, and more)
    - Any other provider implementing the OpenAI API specification

    The API key is loaded from (in order of priority):
    1. The `api_key` parameter in the provider_config
    2. The OPENAI_API_KEY environment variable (loaded from ~/.kader/.env)
    3. Provider-specific environment variables (e.g., MOONSHOT_API_KEY)

    Example:
        # OpenAI
        provider = OpenAICompatibleProvider(
            model="gpt-4o",
            provider_config=OpenAIProviderConfig(
                api_key="your-api-key"
            )
        )

        # Moonshot AI
        provider = OpenAICompatibleProvider(
            model="kimi-k2.5",
            provider_config=OpenAIProviderConfig(
                api_key="your-api-key",
                base_url="https://api.moonshot.cn/v1",
            )
        )

        # Z.ai (GLM-5)
        provider = OpenAICompatibleProvider(
            model="glm-5",
            provider_config=OpenAIProviderConfig(
                api_key="your-api-key",
                base_url="https://api.z.ai/api/paas/v4/",
            )
        )

        # OpenRouter
        provider = OpenAICompatibleProvider(
            model="anthropic/claude-3.5-sonnet",
            provider_config=OpenAIProviderConfig(
                api_key="your-api-key",
                base_url="https://openrouter.ai/api/v1",
            )
        )

        # OpenCode Zen
        provider = OpenAICompatibleProvider(
            model="claude-sonnet-4-5",
            provider_config=OpenAIProviderConfig(
                api_key="your-api-key",
                base_url="https://opencode.ai/zen/v1",
            )
        )

        # Common usage
        response = provider.invoke([Message.user("Hello!")])
        print(response.content)
    """

    def __init__(
        self,
        model: str,
        provider_config: OpenAIProviderConfig | None = None,
        default_config: ModelConfig | None = None,
    ) -> None:
        """Initialize the OpenAI-compatible provider.

        Args:
            model: The model identifier (e.g., "gpt-4o", "kimi-k2.5", "glm-5")
            provider_config: Configuration for the provider including API key,
                base_url, timeout, and max_retries.
            default_config: Default configuration for all requests
        """
        super().__init__(model=model, default_config=default_config)

        # Resolve provider configuration
        if provider_config is None:
            # Try to load from environment variables
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")
            timeout = os.environ.get("OPENAI_TIMEOUT")
            max_retries = os.environ.get("OPENAI_MAX_RETRIES")

            # Filter out empty strings
            api_key = api_key if api_key else None
            base_url = base_url if base_url else None

            provider_config = OpenAIProviderConfig(
                api_key=api_key or "",
                base_url=base_url,
                timeout=float(timeout) if timeout else 60.0,
                max_retries=int(max_retries) if max_retries else 3,
            )

        self._provider_config = provider_config
        self._detected_provider = _detect_provider(provider_config.base_url, model)

        # Initialize OpenAI clients
        client_kwargs: dict = {
            "api_key": provider_config.api_key,
            "max_retries": provider_config.max_retries,
        }

        if provider_config.base_url:
            client_kwargs["base_url"] = provider_config.base_url
        if provider_config.timeout:
            client_kwargs["timeout"] = provider_config.timeout
        if provider_config.default_headers:
            client_kwargs["default_headers"] = provider_config.default_headers

        self._client = OpenAI(**client_kwargs)
        self._async_client = AsyncOpenAI(**client_kwargs)

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to OpenAI format."""
        return [msg.to_dict() for msg in messages]

    def _convert_config_to_params(self, config: ModelConfig) -> dict:
        """Convert ModelConfig to OpenAI API parameters."""
        params: dict = {}

        if config.temperature != 1.0:
            params["temperature"] = config.temperature
        if config.max_tokens is not None:
            params["max_tokens"] = config.max_tokens
        if config.top_p != 1.0:
            params["top_p"] = config.top_p
        if config.top_k is not None:
            params["top_k"] = config.top_k
        if config.frequency_penalty != 0.0:
            params["frequency_penalty"] = config.frequency_penalty
        if config.presence_penalty != 0.0:
            params["presence_penalty"] = config.presence_penalty
        if config.stop_sequences:
            params["stop"] = config.stop_sequences
        if config.seed is not None:
            params["seed"] = config.seed

        # Handle tools
        if config.tools:
            params["tools"] = config.tools
        if config.tool_choice is not None:
            params["tool_choice"] = config.tool_choice

        # Handle response format
        if config.response_format:
            params["response_format"] = config.response_format

        # Merge extra parameters
        params.update(config.extra)

        return params

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI ChatCompletion to LLMResponse."""
        # Extract content and tool calls from the first choice
        content = ""
        tool_calls = None
        finish_reason = None

        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message

            if message.content:
                content = message.content

            if message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Extract usage information
        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

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

        if chunk.choices and len(chunk.choices) > 0:
            choice = chunk.choices[0]

            # Extract delta content
            if choice.delta and choice.delta.content:
                delta = choice.delta.content

            # Extract tool calls from delta
            if choice.delta and choice.delta.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type or "function",
                        "function": {
                            "name": tc.function.name if tc.function else "",
                            "arguments": tc.function.arguments if tc.function else "",
                        },
                    }
                    for tc in choice.delta.tool_calls
                ]

            # Extract finish reason
            if choice.finish_reason:
                finish_reason = choice.finish_reason

        # Extract usage from final chunk (some providers include it)
        if chunk.usage:
            usage = Usage(
                prompt_tokens=chunk.usage.prompt_tokens or 0,
                completion_tokens=chunk.usage.completion_tokens or 0,
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
        """Synchronously invoke the OpenAI-compatible model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=False,
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
        """Synchronously stream the OpenAI-compatible model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response_stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=True,
            **params,
        )

        accumulated_content = ""
        final_usage: Usage | None = None

        for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(chunk, accumulated_content)
            accumulated_content = stream_chunk.content

            # Capture usage from final chunk if available
            if stream_chunk.is_final and stream_chunk.usage:
                final_usage = stream_chunk.usage

            yield stream_chunk

        # Update tracking after stream completes
        # If no usage was provided, estimate it
        if final_usage is None:
            final_usage = Usage(
                prompt_tokens=0,  # Can't determine without tokenizer
                completion_tokens=len(accumulated_content) // 4,
            )

        final_response = LLMResponse(
            content=accumulated_content,
            model=self._model,
            usage=final_usage,
            finish_reason=stream_chunk.finish_reason,
            cost=self.estimate_cost(final_usage),
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
        """Asynchronously invoke the OpenAI-compatible model.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=False,
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
        """Asynchronously stream the OpenAI-compatible model response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        merged_config = self._merge_config(config)
        params = self._convert_config_to_params(merged_config)

        response_stream = await self._async_client.chat.completions.create(
            model=self._model,
            messages=self._convert_messages(messages),
            stream=True,
            **params,
        )

        accumulated_content = ""
        final_usage: Usage | None = None

        async for chunk in response_stream:
            stream_chunk = self._parse_stream_chunk(chunk, accumulated_content)
            accumulated_content = stream_chunk.content

            # Capture usage from final chunk if available
            if stream_chunk.is_final and stream_chunk.usage:
                final_usage = stream_chunk.usage

            yield stream_chunk

        # Update tracking after stream completes
        if final_usage is None:
            final_usage = Usage(
                prompt_tokens=0,
                completion_tokens=len(accumulated_content) // 4,
            )

        final_response = LLMResponse(
            content=accumulated_content,
            model=self._model,
            usage=final_usage,
            finish_reason=stream_chunk.finish_reason,
            cost=self.estimate_cost(final_usage),
        )
        self._update_tracking(final_response)

    # -------------------------------------------------------------------------
    # Token & Cost Methods
    # -------------------------------------------------------------------------

    def count_tokens(
        self,
        text: str | list[Message],
    ) -> int:
        """Count the number of tokens in the given text or messages.

        Note: This uses a rough estimate. For exact token counts,
        you would need the provider's tokenizer (e.g., tiktoken for OpenAI).

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
        """Estimate the cost for the given token usage.

        Args:
            usage: Token usage information

        Returns:
            CostInfo with cost breakdown
        """
        # Get pricing for detected provider
        provider_pricing = PROVIDER_PRICING.get(self._detected_provider, {})
        pricing = provider_pricing.get(self._model)

        if not pricing:
            # Try to find by model pattern matching
            for model_id, model_pricing in provider_pricing.items():
                # Remove version/date suffixes for matching
                base_name = model_id.split("-")[:2]
                if self._model.startswith("-".join(base_name)):
                    pricing = model_pricing
                    break

        if not pricing:
            # Try other providers as fallback
            for provider_name, provider_prices in PROVIDER_PRICING.items():
                if self._model in provider_prices:
                    pricing = provider_prices[self._model]
                    self._detected_provider = provider_name
                    break

        if pricing:
            return pricing.calculate_cost(usage)

        # Return zero cost if pricing unknown
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
        """Get information about the current model.

        Returns:
            ModelInfo with model capabilities and pricing if available
        """
        # Get pricing from detected provider
        provider_pricing = PROVIDER_PRICING.get(self._detected_provider, {})
        pricing = provider_pricing.get(self._model)

        # Default capabilities based on model name
        supports_vision = any(
            x in self._model.lower() for x in ["gpt-4o", "vision", "pixtral"]
        )
        supports_tools = True  # Most OpenAI-compatible providers support tools
        supports_json_mode = True
        supports_streaming = True

        # Context window sizes (approximate defaults)
        context_window = 128000  # Default for newer models
        if "gpt-4" in self._model.lower() and "32k" in self._model.lower():
            context_window = 32768
        elif "gpt-4" in self._model.lower():
            context_window = 8192
        elif "8k" in self._model.lower():
            context_window = 8192
        elif "32k" in self._model.lower():
            context_window = 32768
        elif "128k" in self._model.lower():
            context_window = 128000

        return ModelInfo(
            name=self._model,
            provider=self._detected_provider,
            context_window=context_window,
            pricing=pricing,
            supports_vision=supports_vision,
            supports_tools=supports_tools,
            supports_json_mode=supports_json_mode,
            supports_streaming=supports_streaming,
        )

    @classmethod
    def get_supported_models(
        cls, provider_config: OpenAIProviderConfig | None = None
    ) -> list[str]:
        """Get list of models available from the provider.

        Args:
            provider_config: Optional provider configuration.
                If not provided, attempts to use OPENAI_API_KEY environment variable.

        Returns:
            List of available model identifiers
        """
        try:
            if provider_config is None:
                # Try to load from environment
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_BASE_URL")
                api_key = api_key if api_key else None
                base_url = base_url if base_url else None

                if not api_key:
                    return []

                provider_config = OpenAIProviderConfig(
                    api_key=api_key, base_url=base_url
                )

            client = OpenAI(
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
            )
            models_response = client.models.list()

            return [model.id for model in models_response.data if model.id]
        except Exception:
            return []

    def list_models(self) -> list[str]:
        """List all available models from the provider."""
        return self.get_supported_models(self._provider_config)
