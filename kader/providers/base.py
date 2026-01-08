"""
Base class for LLM Providers.

A versatile, provider-agnostic base class for LLM interactions supporting
OpenAI, Google, Anthropic, Mistral, and other providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    Literal,
    TypeAlias,
)

# Type Aliases
Role: TypeAlias = Literal["system", "user", "assistant", "tool"]
FinishReason: TypeAlias = Literal[
    "stop", "length", "tool_calls", "content_filter", "error", None
]


class MessageRole(str, Enum):
    """Enumeration of message roles for type safety."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a chat message in a conversation."""

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format for API calls."""
        data: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }
        if self.name:
            data["name"] = self.name
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = self.tool_calls
        return data

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        """Create a tool message."""
        return cls(role="tool", tool_call_id=tool_call_id, content=content)


@dataclass
class Usage:
    """Tracks token usage for an LLM request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Additional usage details (provider-specific)
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    def __post_init__(self) -> None:
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        """Add two Usage instances together."""
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class CostInfo:
    """Cost breakdown for an LLM request."""

    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"

    # Additional cost details
    cached_input_cost: float = 0.0

    def __post_init__(self) -> None:
        """Calculate total cost if not provided."""
        if self.total_cost == 0.0:
            self.total_cost = self.input_cost + self.output_cost

    def __add__(self, other: "CostInfo") -> "CostInfo":
        """Add two CostInfo instances together."""
        if self.currency != other.currency:
            raise ValueError(
                f"Cannot add costs with different currencies: {self.currency} vs {other.currency}"
            )
        return CostInfo(
            input_cost=self.input_cost + other.input_cost,
            output_cost=self.output_cost + other.output_cost,
            total_cost=self.total_cost + other.total_cost,
            currency=self.currency,
            cached_input_cost=self.cached_input_cost + other.cached_input_cost,
        )

    def format(self, precision: int = 6) -> str:
        """Format cost as a readable string."""
        return f"${self.total_cost:.{precision}f} {self.currency}"


@dataclass
class ModelConfig:
    """Configuration for model inference parameters."""

    # Core parameters
    temperature: float = 1.0
    max_tokens: int | None = None
    top_p: float = 1.0

    # Sampling parameters
    top_k: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Stop sequences
    stop_sequences: list[str] | None = None

    # Streaming
    stream: bool = False

    # Tool/Function calling
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None

    # Response format
    response_format: dict[str, Any] | None = None

    # Seed for reproducibility
    seed: int | None = None

    # Additional provider-specific parameters
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary, excluding None values."""
        data: dict[str, Any] = {}

        if self.temperature != 1.0:
            data["temperature"] = self.temperature
        if self.max_tokens is not None:
            data["max_tokens"] = self.max_tokens
        if self.top_p != 1.0:
            data["top_p"] = self.top_p
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.frequency_penalty != 0.0:
            data["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty != 0.0:
            data["presence_penalty"] = self.presence_penalty
        if self.stop_sequences:
            data["stop"] = self.stop_sequences
        if self.tools:
            data["tools"] = self.tools
        if self.tool_choice is not None:
            data["tool_choice"] = self.tool_choice
        if self.response_format is not None:
            data["response_format"] = self.response_format
        if self.seed is not None:
            data["seed"] = self.seed

        # Merge extra parameters
        data.update(self.extra)

        return data


@dataclass
class LLMResponse:
    """Complete response from an LLM provider."""

    content: str
    model: str
    usage: Usage
    finish_reason: FinishReason = None

    # Cost information (optional, calculated if pricing is available)
    cost: CostInfo | None = None

    # Tool calls (if any)
    tool_calls: list[dict[str, Any]] | None = None

    # Raw response from provider (for debugging/extension)
    raw_response: Any = None

    # Additional metadata
    id: str | None = None
    created: int | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0

    def to_message(self) -> Message:
        """Convert response to an assistant message."""
        return Message(
            role="assistant",
            content=self.content,
            tool_calls=self.tool_calls,
        )


@dataclass
class StreamChunk:
    """A chunk from a streaming LLM response."""

    content: str = ""
    delta: str = ""
    finish_reason: FinishReason = None

    # Partial usage (available at end of stream for some providers)
    usage: Usage | None = None

    # Tool call deltas
    tool_calls: list[dict[str, Any]] | None = None

    # Index of this chunk in the stream
    index: int = 0

    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk."""
        return self.finish_reason is not None


@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_million: float  # Cost per million input tokens
    output_cost_per_million: float  # Cost per million output tokens
    cached_input_cost_per_million: float | None = (
        None  # Cached input cost (if supported)
    )

    def calculate_cost(self, usage: Usage) -> CostInfo:
        """Calculate cost from usage."""
        input_cost = (usage.prompt_tokens / 1_000_000) * self.input_cost_per_million
        output_cost = (
            usage.completion_tokens / 1_000_000
        ) * self.output_cost_per_million

        cached_cost = 0.0
        if self.cached_input_cost_per_million and usage.cached_tokens > 0:
            cached_cost = (
                usage.cached_tokens / 1_000_000
            ) * self.cached_input_cost_per_million

        return CostInfo(
            input_cost=input_cost,
            output_cost=output_cost,
            cached_input_cost=cached_cost,
        )


@dataclass
class ModelInfo:
    """Information about an LLM model."""

    name: str
    provider: str
    context_window: int
    max_output_tokens: int | None = None
    pricing: ModelPricing | None = None
    supports_vision: bool = False
    supports_tools: bool = False
    supports_json_mode: bool = False
    supports_streaming: bool = True

    # Additional capabilities
    capabilities: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Provides a unified interface for interacting with various LLM providers
    including OpenAI, Google, Anthropic, Mistral, and others.

    Subclasses must implement:
    - invoke: Synchronous single completion
    - ainvoke: Asynchronous single completion
    - stream: Synchronous streaming completion
    - astream: Asynchronous streaming completion
    - count_tokens: Count tokens in text/messages
    - estimate_cost: Estimate cost from usage
    """

    def __init__(
        self,
        model: str,
        default_config: ModelConfig | None = None,
    ) -> None:
        """
        Initialize the LLM provider.

        Args:
            model: The model identifier to use
            default_config: Default configuration for all requests
        """
        self._model = model
        self._default_config = default_config or ModelConfig()
        self._total_usage = Usage()
        self._total_cost = CostInfo()

    @property
    def model(self) -> str:
        """Get the current model identifier."""
        return self._model

    @property
    def total_usage(self) -> Usage:
        """Get total token usage across all requests."""
        return self._total_usage

    @property
    def total_cost(self) -> CostInfo:
        """Get total cost across all requests."""
        return self._total_cost

    def reset_tracking(self) -> None:
        """Reset usage and cost tracking."""
        self._total_usage = Usage()
        self._total_cost = CostInfo()

    def _merge_config(self, config: ModelConfig | None) -> ModelConfig:
        """Merge provided config with defaults."""
        if config is None:
            return self._default_config

        # Create a new config with merged values
        return ModelConfig(
            temperature=config.temperature
            if config.temperature != 1.0
            else self._default_config.temperature,
            max_tokens=config.max_tokens or self._default_config.max_tokens,
            top_p=config.top_p if config.top_p != 1.0 else self._default_config.top_p,
            top_k=config.top_k or self._default_config.top_k,
            frequency_penalty=config.frequency_penalty
            if config.frequency_penalty != 0.0
            else self._default_config.frequency_penalty,
            presence_penalty=config.presence_penalty
            if config.presence_penalty != 0.0
            else self._default_config.presence_penalty,
            stop_sequences=config.stop_sequences or self._default_config.stop_sequences,
            stream=config.stream,
            tools=config.tools or self._default_config.tools,
            tool_choice=config.tool_choice or self._default_config.tool_choice,
            response_format=config.response_format
            or self._default_config.response_format,
            seed=config.seed or self._default_config.seed,
            extra={**self._default_config.extra, **config.extra},
        )

    def _update_tracking(self, response: LLMResponse) -> None:
        """Update usage and cost tracking from a response."""
        self._total_usage = self._total_usage + response.usage
        self._total_usage.__post_init__()  # Recalculate total_tokens

        if response.cost:
            self._total_cost = self._total_cost + response.cost
            self._total_cost.__post_init__()  # Recalculate total_cost

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def invoke(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """
        Synchronously invoke the LLM with the given messages.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        ...

    @abstractmethod
    async def ainvoke(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> LLMResponse:
        """
        Asynchronously invoke the LLM with the given messages.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Returns:
            LLMResponse with the model's response
        """
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> Iterator[StreamChunk]:
        """
        Synchronously stream the LLM response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        ...

    @abstractmethod
    async def astream(
        self,
        messages: list[Message],
        config: ModelConfig | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        Asynchronously stream the LLM response.

        Args:
            messages: List of messages in the conversation
            config: Optional configuration overrides

        Yields:
            StreamChunk objects as they arrive
        """
        ...

    @abstractmethod
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
        ...

    @abstractmethod
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
        ...

    # -------------------------------------------------------------------------
    # Concrete Methods - Can be overridden if needed
    # -------------------------------------------------------------------------

    def get_model_info(self) -> ModelInfo | None:
        """
        Get information about the current model.

        Returns:
            ModelInfo if available, None otherwise
        """
        return None

    @classmethod
    def get_supported_models(cls) -> list[str]:
        """
        Get list of models supported by this provider.

        Returns:
            List of model identifiers
        """
        return []

    def validate_config(self, config: ModelConfig) -> bool:
        """
        Validate the given configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        if config.temperature < 0 or config.temperature > 2:
            return False
        if config.top_p < 0 or config.top_p > 1:
            return False
        if config.max_tokens is not None and config.max_tokens < 1:
            return False
        return True

    def validate_messages(self, messages: list[Message]) -> bool:
        """
        Validate the given messages.

        Args:
            messages: Messages to validate

        Returns:
            True if valid, False otherwise
        """
        if not messages:
            return False

        valid_roles = {"system", "user", "assistant", "tool"}
        for msg in messages:
            if msg.role not in valid_roles:
                return False
            if not msg.content and not msg.tool_calls:
                return False

        return True

    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model='{self._model}')"
