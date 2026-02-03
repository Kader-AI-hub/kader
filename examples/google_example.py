"""
Google Provider Example

Demonstrates how to use the Kader Google provider for:
- Basic LLM invocation
- Streaming responses
- Asynchronous operations
- Configuration options
- Tool/function calling
- Dynamic model listing

API Key Setup:
    Set your GEMINI_API_KEY in ~/.kader/.env:
    GEMINI_API_KEY='your-api-key-here'
"""

import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.providers.base import Message, ModelConfig
from kader.providers.google import GoogleProvider


def demo_basic_invocation():
    """Demonstrate basic synchronous invocation."""
    print("\n=== Basic Google Invocation Demo ===")

    # Initialize the provider with a model
    # Note: Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable
    provider = GoogleProvider(model="gemini-2.5-flash")

    # Create a simple conversation
    messages = [
        Message.system("You are a helpful assistant that responds concisely."),
        Message.user("What are the benefits of using Google Gemini models?"),
    ]

    try:
        # Invoke the model synchronously
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"Finish reason: {response.finish_reason}")

        # Show cost tracking
        if response.cost:
            print(f"Cost: {response.cost.format()}")

        # Show total usage tracking
        print(f"Total usage tracked: {provider.total_usage.total_tokens} tokens")

    except Exception as e:
        print(f"Error during invocation: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n=== Google Streaming Demo ===")

    provider = GoogleProvider(model="gemini-2.5-flash")

    messages = [Message.user("Write a short poem about artificial intelligence.")]

    try:
        print("Streaming response:")
        full_content = ""
        for chunk in provider.stream(messages):
            if chunk.delta:
                print(chunk.delta, end="", flush=True)
                full_content = chunk.content

        print(f"\n\nFinal content length: {len(full_content)} characters")
        print(f"Total usage: {provider.total_usage.total_tokens} tokens")

    except Exception as e:
        print(f"Error during streaming: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_async_invocation():
    """Demonstrate asynchronous invocation."""
    print("\n=== Google Async Invocation Demo ===")

    async def async_demo():
        provider = GoogleProvider(model="gemini-2.5-flash")

        messages = [
            Message.system("You are a helpful assistant."),
            Message.user(
                "What is the difference between Gemini 2.5 Flash and Pro models?"
            ),
        ]

        try:
            # Asynchronously invoke the model
            response = await provider.ainvoke(messages)

            print(f"Model: {response.model}")
            print(f"Content: {response.content}")
            print(f"Tokens: {response.usage.total_tokens}")
            print(f"Finish reason: {response.finish_reason}")

        except Exception as e:
            print(f"Error during async invocation: {e}")
            print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")

    asyncio.run(async_demo())


def demo_async_streaming():
    """Demonstrate asynchronous streaming."""
    print("\n=== Google Async Streaming Demo ===")

    async def async_stream_demo():
        provider = GoogleProvider(model="gemini-2.5-flash")

        messages = [Message.user("Explain quantum computing in simple terms.")]

        try:
            print("Async streaming response:")
            full_content = ""
            async for chunk in provider.astream(messages):
                if chunk.delta:
                    print(chunk.delta, end="", flush=True)
                    full_content = chunk.content

            print(f"\n\nFinal content length: {len(full_content)} characters")

        except Exception as e:
            print(f"Error during async streaming: {e}")
            print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")

    asyncio.run(async_stream_demo())


def demo_configuration():
    """Demonstrate using different configurations."""
    print("\n=== Google Configuration Demo ===")

    # Create a provider with default configuration
    default_config = ModelConfig(
        temperature=0.7,  # More creative
        max_tokens=150,  # Limit response length
        top_p=0.9,
    )

    provider = GoogleProvider(model="gemini-2.5-flash", default_config=default_config)

    messages = [Message.user("Tell me a creative fact about space.")]

    try:
        # This will use the default configuration
        response = provider.invoke(messages)
        print(f"Using default config - Content: {response.content[:100]}...")
        print(f"Tokens: {response.usage.total_tokens}")

        # Override configuration for this specific call
        creative_config = ModelConfig(
            temperature=1.2,  # Even more creative/random
            max_tokens=200,
        )

        messages = [Message.user("Generate an original haiku about technology.")]
        response = provider.invoke(messages, config=creative_config)
        print(f"\nUsing creative config - Content: {response.content}")

    except Exception as e:
        print(f"Error during configuration demo: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_conversation_history():
    """Demonstrate maintaining conversation context."""
    print("\n=== Google Conversation History Demo ===")

    provider = GoogleProvider(model="gemini-2.5-flash")

    # Simulate a multi-turn conversation
    conversation = [
        Message.system("You are a helpful coding assistant."),
        Message.user("What is Python used for?"),
        Message.assistant(
            "Python is a versatile programming language used for web development, data science, AI/ML, automation, and more."
        ),
        Message.user("Can you give me a simple Python example?"),
    ]

    try:
        response = provider.invoke(conversation)
        print(f"Response to follow-up: {response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error during conversation demo: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_list_models():
    """Demonstrate listing available models dynamically."""
    print("\n=== Google List Models Demo ===")

    try:
        # Get supported models without creating a provider instance
        models = GoogleProvider.get_supported_models()

        print(f"Found {len(models)} Gemini models:")
        for model in models[:10]:  # Show first 10
            print(f"  - {model}")

        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")

    except Exception as e:
        print(f"Error listing models: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_model_info():
    """Demonstrate getting model information."""
    print("\n=== Google Model Info Demo ===")

    provider = GoogleProvider(model="gemini-2.5-flash")

    try:
        model_info = provider.get_model_info()

        if model_info:
            print(f"Model Name: {model_info.name}")
            print(f"Provider: {model_info.provider}")
            print(f"Context Window: {model_info.context_window:,} tokens")
            print(f"Max Output Tokens: {model_info.max_output_tokens}")
            print(f"Supports Tools: {model_info.supports_tools}")
            print(f"Supports Streaming: {model_info.supports_streaming}")
            print(f"Supports Vision: {model_info.supports_vision}")
        else:
            print("Could not retrieve model info.")

    except Exception as e:
        print(f"Error getting model info: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_token_counting():
    """Demonstrate token counting."""
    print("\n=== Google Token Counting Demo ===")

    provider = GoogleProvider(model="gemini-2.5-flash")

    try:
        # Count tokens in a string
        text = "Hello, how are you today? I'm looking forward to our conversation."
        token_count = provider.count_tokens(text)
        print(f"Text: '{text}'")
        print(f"Token count: {token_count}")

        # Count tokens in messages
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is the meaning of life?"),
        ]
        msg_token_count = provider.count_tokens(messages)
        print(f"\nMessages token count: {msg_token_count}")

    except Exception as e:
        print(f"Error counting tokens: {e}")
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def demo_cost_estimation():
    """Demonstrate cost estimation."""
    print("\n=== Google Cost Estimation Demo ===")

    provider = GoogleProvider(model="gemini-2.5-flash")

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
        print("Make sure GEMINI_API_KEY or GOOGLE_API_KEY is set.")


def main():
    """Run all Google provider demos."""
    print("Kader Google Provider Examples")
    print("=" * 40)

    print("\nAPI Key Setup:")
    print("  Add your GEMINI_API_KEY to ~/.kader/.env:")
    print("  GEMINI_API_KEY='your-api-key-here'")
    print("\n  Get your API key from: https://aistudio.google.com/apikey")

    demo_basic_invocation()
    demo_streaming()
    demo_async_invocation()
    demo_async_streaming()
    demo_configuration()
    demo_conversation_history()
    demo_list_models()
    demo_model_info()
    demo_token_counting()
    demo_cost_estimation()

    print("\n[OK] All Google demos completed!")


if __name__ == "__main__":
    main()
