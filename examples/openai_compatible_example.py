"""
OpenAI-Compatible Provider Example

Demonstrates how to use the Kader OpenAI-Compatible provider for:
- OpenAI models (GPT-4, GPT-4o, GPT-3.5 Turbo)
- Moonshot AI models (kimi-k2.5)
- Z.ai models (GLM-5)
- OpenRouter (access to 200+ models from various providers)
- Other OpenAI-compatible providers
- Basic LLM invocation
- Streaming responses
- Asynchronous operations
- Configuration options
- Token counting
- Cost estimation

API Key Setup:
    Set your API key in ~/.kader/.env:
    OPENAI_API_KEY='your-openai-api-key-here'
    MOONSHOT_API_KEY='your-moonshot-api-key-here'
    ZAI_API_KEY='your-zai-api-key-here'
    OPENROUTER_API_KEY='your-openrouter-api-key-here'

    Get your API keys from:
    - OpenAI: https://platform.openai.com/api-keys
    - Moonshot AI: https://platform.moonshot.cn
    - Z.ai: https://open.bigmodel.cn
    - OpenRouter: https://openrouter.ai/keys
"""

import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.providers.base import Message, ModelConfig
from kader.providers.openai_compatible import (
    OpenAICompatibleProvider,
    OpenAIProviderConfig,
)


def demo_openai_basic():
    """Demonstrate basic OpenAI invocation."""
    print("\n=== OpenAI Basic Invocation Demo ===")

    # Initialize the provider with OpenAI
    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    messages = [
        Message.system("You are a helpful assistant that responds concisely."),
        Message.user("What are the benefits of using GPT models?"),
    ]

    try:
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Finish reason: {response.finish_reason}")

        if response.cost:
            print(f"Cost: {response.cost.format()}")

    except Exception as e:
        print(f"Error during OpenAI invocation: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_moonshot_ai():
    """Demonstrate Moonshot AI (kimi-k2.5) invocation."""
    print("\n=== Moonshot AI (kimi-k2.5) Demo ===")

    # Initialize the provider with Moonshot AI configuration
    provider = OpenAICompatibleProvider(
        model="kimi-k2.5",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("MOONSHOT_API_KEY", ""),
            base_url="https://api.moonshot.cn/v1",
        ),
    )

    messages = [
        Message.system("You are Kimi, a helpful AI assistant."),
        Message.user("Explain the key features of kimi-k2.5 model."),
    ]

    try:
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")

        if response.cost:
            print(f"Cost: {response.cost.format()}")

    except Exception as e:
        print(f"Error during Moonshot AI invocation: {e}")
        print("Make sure MOONSHOT_API_KEY is set.")


def demo_zai_glm5():
    """Demonstrate Z.ai (GLM-5) invocation."""
    print("\n=== Z.ai (GLM-5) Demo ===")

    # Initialize the provider with Z.ai configuration
    provider = OpenAICompatibleProvider(
        model="glm-5",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("ZAI_API_KEY", ""),
            base_url="https://api.z.ai/api/paas/v4/",
        ),
    )

    messages = [
        Message.system("You are GLM-5, an AI assistant developed by Z.ai."),
        Message.user("What can you tell me about the GLM-5 model?"),
    ]

    try:
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")

        if response.cost:
            print(f"Cost: {response.cost.format()}")

    except Exception as e:
        print(f"Error during Z.ai invocation: {e}")
        print("Make sure ZAI_API_KEY is set.")


def demo_openrouter():
    """Demonstrate OpenRouter invocation with Claude 3.5 Sonnet."""
    print("\n=== OpenRouter Demo ===")

    # Initialize the provider with OpenRouter configuration
    provider = OpenAICompatibleProvider(
        model="deepseek/deepseek-r1-0528:free",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            base_url="https://openrouter.ai/api/v1",
        ),
    )

    messages = [
        Message.system("You are a helpful AI assistant."),
        Message.user("What are the benefits of using OpenRouter?"),
    ]

    try:
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")

        if response.cost:
            print(f"Cost: {response.cost.format()}")

    except Exception as e:
        print(f"Error during OpenRouter invocation: {e}")
        print("Make sure OPENROUTER_API_KEY is set.")


def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n=== OpenAI Streaming Demo ===")

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    messages = [Message.user("Write a short poem about artificial intelligence.")]

    try:
        print("Streaming response:")
        full_content = ""
        for chunk in provider.stream(messages):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                full_content = chunk.content

        print(f"\n\nFinal content length: {len(full_content)} characters")

    except Exception as e:
        print(f"Error during streaming: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_async_invocation():
    """Demonstrate asynchronous invocation."""
    print("\n=== Async Invocation Demo ===")

    async def async_demo():
        provider = OpenAICompatibleProvider(
            model="gpt-4o-mini",
            provider_config=OpenAIProviderConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            ),
        )

        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is the difference between GPT-4 and GPT-4o?"),
        ]

        try:
            response = await provider.ainvoke(messages)

            print(f"Model: {response.model}")
            print(f"Content: {response.content}")
            print(f"Tokens: {response.usage.total_tokens}")

        except Exception as e:
            print(f"Error during async invocation: {e}")
            print("Make sure OPENAI_API_KEY is set.")

    asyncio.run(async_demo())


def demo_configuration():
    """Demonstrate using different configurations."""
    print("\n=== Configuration Demo ===")

    # Create a provider with default configuration
    default_config = ModelConfig(
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
    )

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
        default_config=default_config,
    )

    messages = [Message.user("Tell me a creative fact about space.")]

    try:
        response = provider.invoke(messages)
        print(f"Using default config - Content: {response.content[:100]}...")

        # Override configuration for this specific call
        creative_config = ModelConfig(
            temperature=1.2,
            max_tokens=200,
        )

        messages = [Message.user("Generate an original haiku about technology.")]
        response = provider.invoke(messages, config=creative_config)
        print(f"\nUsing creative config - Content: {response.content}")

    except Exception as e:
        print(f"Error during configuration demo: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_conversation_history():
    """Demonstrate maintaining conversation context."""
    print("\n=== Conversation History Demo ===")

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    # Simulate a multi-turn conversation
    conversation = [
        Message.system("You are a helpful coding assistant."),
        Message.user("What is Python used for?"),
        Message.assistant(
            "Python is a versatile programming language used for web development, "
            "data science, AI/ML, automation, and more."
        ),
        Message.user("Can you give me a simple Python example?"),
    ]

    try:
        response = provider.invoke(conversation)
        print(f"Response to follow-up: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error during conversation demo: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_list_models():
    """Demonstrate listing available models dynamically."""
    print("\n=== List Models Demo ===")

    try:
        # Get supported models with OpenAI configuration
        models = OpenAICompatibleProvider.get_supported_models(
            provider_config=OpenAIProviderConfig(
                api_key=os.environ.get("OPENAI_API_KEY", ""),
            )
        )

        print(f"Found {len(models)} models:")
        for model in models[:10]:  # Show first 10
            print(f"  - {model}")

        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

    except Exception as e:
        print(f"Error listing models: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_model_info():
    """Demonstrate getting model information."""
    print("\n=== Model Info Demo ===")

    provider = OpenAICompatibleProvider(
        model="gpt-4o",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    try:
        model_info = provider.get_model_info()

        if model_info:
            print(f"Model Name: {model_info.name}")
            print(f"Provider: {model_info.provider}")
            print(f"Context Window: {model_info.context_window:,} tokens")
            print(f"Supports Tools: {model_info.supports_tools}")
            print(f"Supports Streaming: {model_info.supports_streaming}")
            print(f"Supports Vision: {model_info.supports_vision}")
        else:
            print("Could not retrieve model info.")

    except Exception as e:
        print(f"Error getting model info: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_token_counting():
    """Demonstrate token counting."""
    print("\n=== Token Counting Demo ===")

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    try:
        # Count tokens in a string
        text = "Hello, how are you today? I'm looking forward to our conversation."
        token_count = provider.count_tokens(text)
        print(f"Text: '{text}'")
        print(f"Estimated token count: {token_count}")

        # Count tokens in messages
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is the meaning of life?"),
        ]
        msg_token_count = provider.count_tokens(messages)
        print(f"\nMessages token count: {msg_token_count}")

    except Exception as e:
        print(f"Error counting tokens: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def demo_cost_estimation():
    """Demonstrate cost estimation."""
    print("\n=== Cost Estimation Demo ===")

    provider = OpenAICompatibleProvider(
        model="gpt-4o-mini",
        provider_config=OpenAIProviderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        ),
    )

    messages = [Message.user("Write a brief summary of machine learning.")]

    try:
        response = provider.invoke(messages)

        print(f"Response: {response.content[:100]}...")
        print("\nUsage:")
        print(f"  Prompt tokens: {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens}")

        if response.cost:
            print("\nCost Breakdown:")
            print(f"  Input cost: ${response.cost.input_cost:.6f}")
            print(f"  Output cost: ${response.cost.output_cost:.6f}")
            print(f"  Total cost: {response.cost.format()}")

    except Exception as e:
        print(f"Error during cost estimation demo: {e}")
        print("Make sure OPENAI_API_KEY is set.")


def main():
    """Run all OpenAI-Compatible provider demos."""
    print("Kader OpenAI-Compatible Provider Examples")
    print("=" * 40)

    print("\nAPI Key Setup:")
    print("  Add your API keys to ~/.kader/.env:")
    print("  OPENAI_API_KEY='your-openai-api-key-here'")
    print("  MOONSHOT_API_KEY='your-moonshot-api-key-here'")
    print("  ZAI_API_KEY='your-zai-api-key-here'")
    print("  OPENROUTER_API_KEY='your-openrouter-api-key-here'")
    print("\n  Get your API keys from:")
    print("  - OpenAI: https://platform.openai.com/api-keys")
    print("  - Moonshot AI: https://platform.moonshot.cn")
    print("  - Z.ai: https://open.bigmodel.cn")
    print("  - OpenRouter: https://openrouter.ai/keys")

    demo_openai_basic()
    demo_streaming()
    demo_async_invocation()
    demo_configuration()
    demo_conversation_history()
    demo_list_models()
    demo_model_info()
    demo_token_counting()
    demo_cost_estimation()

    # Optional: Uncomment to test specific providers (requires API keys)
    demo_moonshot_ai()
    demo_zai_glm5()
    demo_openrouter()

    print("\n[OK] All OpenAI-Compatible demos completed!")


if __name__ == "__main__":
    main()
