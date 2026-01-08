"""
Ollama Provider Example

Demonstrates how to use the Kader Ollama provider for:
- Basic LLM invocation
- Streaming responses
- Asynchronous operations
- Configuration options
- Tool/function calling (if supported)
"""

import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.providers.base import Message, ModelConfig
from kader.providers.ollama import OllamaProvider


def demo_basic_invocation():
    """Demonstrate basic synchronous invocation."""
    print("\n=== Basic Ollama Invocation Demo ===")

    # Initialize the provider with a model (using a common Ollama model)
    # Note: Make sure you have Ollama installed and running with the model pulled
    provider = OllamaProvider(
        model="gpt-oss:120b-cloud"
    )  # Using the model from the docstring

    # Create a simple conversation
    messages = [
        Message.system("You are a helpful assistant that responds concisely."),
        Message.user("What are the benefits of using Ollama for local LLMs?"),
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

        # Show total usage tracking
        print(f"Total usage tracked: {provider.total_usage.total_tokens} tokens")

    except Exception as e:
        print(f"Error during invocation: {e}")
        print(
            "Make sure Ollama is running and the model is pulled with: ollama pull gpt-oss:120b-cloud"
        )


def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n=== Ollama Streaming Demo ===")

    provider = OllamaProvider(model="gpt-oss:120b-cloud")

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
        print("Make sure Ollama is running and the model is pulled.")


def demo_async_invocation():
    """Demonstrate asynchronous invocation."""
    print("\n=== Ollama Async Invocation Demo ===")

    async def async_demo():
        provider = OllamaProvider(model="gpt-oss:120b-cloud")

        messages = [
            Message.system("You are a helpful assistant."),
            Message.user(
                "What is the difference between synchronous and asynchronous API calls?"
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
            print("Make sure Ollama is running and the model is pulled.")

    asyncio.run(async_demo())


def demo_async_streaming():
    """Demonstrate asynchronous streaming."""
    print("\n=== Ollama Async Streaming Demo ===")

    async def async_stream_demo():
        provider = OllamaProvider(model="gpt-oss:120b-cloud")

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
            print("Make sure Ollama is running and the model is pulled.")

    asyncio.run(async_stream_demo())


def demo_configuration():
    """Demonstrate using different configurations."""
    print("\n=== Ollama Configuration Demo ===")

    # Create a provider with default configuration
    default_config = ModelConfig(
        temperature=0.7,  # More creative
        max_tokens=150,  # Limit response length
        top_p=0.9,
    )

    provider = OllamaProvider(model="gpt-oss:120b-cloud", default_config=default_config)

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
        print("Make sure Ollama is running and the model is pulled.")


def demo_conversation_history():
    """Demonstrate maintaining conversation context."""
    print("\n=== Ollama Conversation History Demo ===")

    provider = OllamaProvider(model="gpt-oss:120b-cloud")

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
        print("Make sure Ollama is running and the model is pulled.")


def demo_error_handling():
    """Demonstrate error handling with Ollama."""
    print("\n=== Ollama Error Handling Demo ===")

    try:
        # Try to use a model that might not exist
        provider = OllamaProvider(model="nonexistent-model-123")

        messages = [Message.user("Hello")]
        response = provider.invoke(messages)

    except Exception as e:
        print(f"Expected error with non-existent model: {e}")
        print("This demonstrates proper error handling.")


def main():
    """Run all Ollama provider demos."""
    print("Kader Ollama Provider Examples")
    print("=" * 40)

    print("\nNote: Make sure Ollama is installed and running on your system.")
    print("You can install Ollama from https://ollama.ai and pull a model with:")
    print("  ollama pull gpt-oss:120b-cloud")
    print("Or any other model of your choice, adjusting the example code accordingly.")

    demo_basic_invocation()
    demo_streaming()
    demo_async_invocation()
    demo_async_streaming()
    demo_configuration()
    demo_conversation_history()
    demo_error_handling()

    print("\n[OK] All Ollama demos completed!")


if __name__ == "__main__":
    main()
