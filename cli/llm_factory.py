"""LLM Provider Factory for Kader CLI.

Factory pattern implementation for creating LLM provider instances
with automatic provider detection based on model name format.
"""

from typing import Optional

from kader.providers import GoogleProvider, MistralProvider, OllamaProvider
from kader.providers.base import BaseLLMProvider, ModelConfig


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    Supports multiple providers with automatic detection based on model name format.
    Model names can be specified as:
    - "provider:model" (e.g., "google:gemini-2.5-flash", "ollama:kimi-k2.5:cloud")
    - "model" (defaults to Ollama for backward compatibility)

    Example:
        factory = LLMProviderFactory()
        provider = factory.create_provider("google:gemini-2.5-flash")

        # Or with default provider (Ollama)
        provider = factory.create_provider("kimi-k2.5:cloud")
    """

    # Registered provider classes
    PROVIDERS: dict[str, type[BaseLLMProvider]] = {
        "ollama": OllamaProvider,
        "google": GoogleProvider,
        "mistral": MistralProvider,
    }

    # Default provider when no prefix is specified
    DEFAULT_PROVIDER = "ollama"

    @classmethod
    def parse_model_name(cls, model_string: str) -> tuple[str, str]:
        """
        Parse model string to extract provider and model name.

        Args:
            model_string: Model string in format "provider:model" or just "model"

        Returns:
            Tuple of (provider_name, model_name)
        """
        # Check if the string starts with a known provider prefix
        for provider_name in cls.PROVIDERS.keys():
            prefix = f"{provider_name}:"
            if model_string.lower().startswith(prefix):
                return provider_name, model_string[len(prefix) :]

        # No known provider prefix found, use default
        return cls.DEFAULT_PROVIDER, model_string

    @classmethod
    def create_provider(
        cls,
        model_string: str,
        config: Optional[ModelConfig] = None,
    ) -> BaseLLMProvider:
        """
        Create an LLM provider instance.

        Args:
            model_string: Model identifier (e.g., "google:gemini-2.5-flash" or "kimi-k2.5:cloud")
            config: Optional model configuration

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_name, model_name = cls.parse_model_name(model_string)

        provider_class = cls.PROVIDERS.get(provider_name)
        if not provider_class:
            supported = ", ".join(cls.PROVIDERS.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Supported: {supported}"
            )

        return provider_class(model=model_name, default_config=config)

    @classmethod
    def get_all_models(cls) -> dict[str, list[str]]:
        """
        Get all available models from all registered providers.

        Returns:
            Dictionary mapping provider names to their available models
            (with provider prefix included in model names)
        """
        models: dict[str, list[str]] = {}

        # Get Ollama models
        try:
            ollama_models = OllamaProvider.get_supported_models()
            models["ollama"] = [f"ollama:{m}" for m in ollama_models]
        except Exception:
            models["ollama"] = []

        # Get Google models
        try:
            google_models = GoogleProvider.get_supported_models()
            models["google"] = [f"google:{m}" for m in google_models]
        except Exception:
            models["google"] = []

        # Get Mistral models
        try:
            mistral_models = MistralProvider.get_supported_models()
            models["mistral"] = [f"mistral:{m}" for m in mistral_models]
        except Exception:
            models["mistral"] = []

        return models

    @classmethod
    def get_flat_model_list(cls) -> list[str]:
        """
        Get a flattened list of all available models with provider prefixes.

        Returns:
            List of model strings in "provider:model" format
        """
        all_models = cls.get_all_models()
        flat_list: list[str] = []
        for models in all_models.values():
            flat_list.extend(models)
        return flat_list

    @classmethod
    def is_provider_available(cls, provider_name: str) -> bool:
        """
        Check if a provider is available and configured.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is available and has models, False otherwise
        """
        provider_name = provider_name.lower()
        if provider_name not in cls.PROVIDERS:
            return False

        # Try to get models to verify provider is working
        try:
            provider_class = cls.PROVIDERS[provider_name]
            models = provider_class.get_supported_models()
            return len(models) > 0
        except Exception:
            return False

    @classmethod
    def get_provider_name(cls, model_string: str) -> str:
        """
        Get the provider name for a given model string.

        Args:
            model_string: Model string in format "provider:model" or just "model"

        Returns:
            Provider name (e.g., "ollama", "google")
        """
        provider_name, _ = cls.parse_model_name(model_string)
        return provider_name
