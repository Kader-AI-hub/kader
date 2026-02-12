"""
Unit tests for the LLM Provider Factory.
"""

import os

import pytest

from cli.llm_factory import LLMProviderFactory
from kader.providers import (
    GoogleProvider,
    MistralProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)


class TestLLMProviderFactoryParseModelName:
    """Test cases for parse_model_name."""

    def test_parse_ollama_model(self):
        """Test parsing Ollama model string."""
        provider, model = LLMProviderFactory.parse_model_name("ollama:llama3.2")
        assert provider == "ollama"
        assert model == "llama3.2"

    def test_parse_google_model(self):
        """Test parsing Google model string."""
        provider, model = LLMProviderFactory.parse_model_name("google:gemini-2.5-flash")
        assert provider == "google"
        assert model == "gemini-2.5-flash"

    def test_parse_mistral_model(self):
        """Test parsing Mistral model string."""
        provider, model = LLMProviderFactory.parse_model_name(
            "mistral:mistral-large-latest"
        )
        assert provider == "mistral"
        assert model == "mistral-large-latest"

    def test_parse_openai_model(self):
        """Test parsing OpenAI model string."""
        provider, model = LLMProviderFactory.parse_model_name("openai:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"

    def test_parse_moonshot_model(self):
        """Test parsing Moonshot model string."""
        provider, model = LLMProviderFactory.parse_model_name("moonshot:kimi-k2.5")
        assert provider == "moonshot"
        assert model == "kimi-k2.5"

    def test_parse_zai_model(self):
        """Test parsing Z.ai model string."""
        provider, model = LLMProviderFactory.parse_model_name("zai:glm-5")
        assert provider == "zai"
        assert model == "glm-5"

    def test_parse_openrouter_model(self):
        """Test parsing OpenRouter model string."""
        provider, model = LLMProviderFactory.parse_model_name(
            "openrouter:anthropic/claude-3.5-sonnet"
        )
        assert provider == "openrouter"
        assert model == "anthropic/claude-3.5-sonnet"

    def test_parse_opencode_model(self):
        """Test parsing OpenCode Zen model string."""
        provider, model = LLMProviderFactory.parse_model_name(
            "opencode:claude-sonnet-4-5"
        )
        assert provider == "opencode"
        assert model == "claude-sonnet-4-5"

    def test_parse_default_provider(self):
        """Test parsing without provider prefix (defaults to ollama)."""
        provider, model = LLMProviderFactory.parse_model_name("llama3.2")
        assert provider == "ollama"
        assert model == "llama3.2"

    def test_parse_case_insensitive(self):
        """Test that provider matching is case insensitive."""
        provider, model = LLMProviderFactory.parse_model_name("OPENAI:gpt-4o")
        assert provider == "openai"
        assert model == "gpt-4o"


class TestLLMProviderFactoryCreateProvider:
    """Test cases for create_provider."""

    def test_create_ollama_provider(self):
        """Test creating Ollama provider."""
        provider = LLMProviderFactory.create_provider("ollama:llama3.2")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "llama3.2"

    def test_create_google_provider(self):
        """Test creating Google provider."""
        provider = LLMProviderFactory.create_provider("google:gemini-2.5-flash")
        assert isinstance(provider, GoogleProvider)
        assert provider.model == "gemini-2.5-flash"

    def test_create_mistral_provider(self):
        """Test creating Mistral provider."""
        provider = LLMProviderFactory.create_provider("mistral:mistral-large-latest")
        assert isinstance(provider, MistralProvider)
        assert provider.model == "mistral-large-latest"

    def test_create_openai_compatible_provider_no_api_key(self):
        """Test creating OpenAI-compatible provider without API key raises error."""
        # Ensure API key is not set
        env_key = "OPENAI_API_KEY"
        original_key = os.environ.get(env_key)
        if env_key in os.environ:
            del os.environ[env_key]

        try:
            with pytest.raises(ValueError, match="API key not found"):
                LLMProviderFactory.create_provider("openai:gpt-4o")
        finally:
            # Restore original key
            if original_key:
                os.environ[env_key] = original_key

    def test_create_without_prefix_defaults_to_ollama(self):
        """Test creating provider without prefix defaults to Ollama."""
        provider = LLMProviderFactory.create_provider("custom-model")
        assert isinstance(provider, OllamaProvider)
        assert provider.model == "custom-model"


class TestLLMProviderFactoryGetProviderName:
    """Test cases for get_provider_name."""

    def test_get_provider_name_openai(self):
        """Test getting provider name for OpenAI."""
        assert LLMProviderFactory.get_provider_name("openai:gpt-4o") == "openai"

    def test_get_provider_name_moonshot(self):
        """Test getting provider name for Moonshot."""
        assert LLMProviderFactory.get_provider_name("moonshot:kimi-k2.5") == "moonshot"

    def test_get_provider_name_default(self):
        """Test getting provider name for default (no prefix)."""
        assert LLMProviderFactory.get_provider_name("llama3.2") == "ollama"


class TestLLMProviderFactoryProviderConfigs:
    """Test cases for PROVIDER_CONFIGS."""

    def test_openai_config(self):
        """Test OpenAI provider configuration."""
        config = LLMProviderFactory.PROVIDER_CONFIGS["openai"]
        assert config["env_key"] == "OPENAI_API_KEY"
        assert config["base_url"] is None

    def test_moonshot_config(self):
        """Test Moonshot provider configuration."""
        config = LLMProviderFactory.PROVIDER_CONFIGS["moonshot"]
        assert config["env_key"] == "MOONSHOT_API_KEY"
        assert config["base_url"] == "https://api.moonshot.cn/v1"

    def test_zai_config(self):
        """Test Z.ai provider configuration."""
        config = LLMProviderFactory.PROVIDER_CONFIGS["zai"]
        assert config["env_key"] == "ZAI_API_KEY"
        assert config["base_url"] == "https://api.z.ai/api/paas/v4/"

    def test_openrouter_config(self):
        """Test OpenRouter provider configuration."""
        config = LLMProviderFactory.PROVIDER_CONFIGS["openrouter"]
        assert config["env_key"] == "OPENROUTER_API_KEY"
        assert config["base_url"] == "https://openrouter.ai/api/v1"

    def test_opencode_config(self):
        """Test OpenCode Zen provider configuration."""
        config = LLMProviderFactory.PROVIDER_CONFIGS["opencode"]
        assert config["env_key"] == "OPENCODE_API_KEY"
        assert config["base_url"] == "https://opencode.ai/zen/v1"


class TestLLMProviderFactorySupportedProviders:
    """Test cases for supported providers."""

    def test_all_providers_registered(self):
        """Test that all providers are registered."""
        expected_providers = [
            "ollama",
            "google",
            "mistral",
            "openai",
            "moonshot",
            "zai",
            "openrouter",
            "opencode",
        ]
        for provider in expected_providers:
            assert provider in LLMProviderFactory.PROVIDERS

    def test_openai_compatible_providers_use_same_class(self):
        """Test that OpenAI-compatible providers use OpenAICompatibleProvider."""
        openai_compatible = ["openai", "moonshot", "zai", "openrouter", "opencode"]
        for provider in openai_compatible:
            assert LLMProviderFactory.PROVIDERS[provider] == OpenAICompatibleProvider

    def test_non_openai_providers_use_specific_classes(self):
        """Test that non-OpenAI providers use their specific classes."""
        assert LLMProviderFactory.PROVIDERS["ollama"] == OllamaProvider
        assert LLMProviderFactory.PROVIDERS["google"] == GoogleProvider
        assert LLMProviderFactory.PROVIDERS["mistral"] == MistralProvider
