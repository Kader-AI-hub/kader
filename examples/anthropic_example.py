"""
Anthropic Provider Example

Demonstrates how to use the Kader Anthropic provider for:
- Basic LLM invocation
- Streaming responses
- Asynchronous operations
- Configuration options
- Tool/function calling
- Dynamic model listing
- Token counting
- Cost estimation
- Conversation history

API Key Setup:
    Set your ANTHROPIC_API_KEY in ~/.kader/.env:
    ANTHROPIC_API_KEY='your-api-key-here'

    Get your API key from: https://console.anthropic.com/settings/keys
"""

import asyncio
import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.providers.anthropic import AnthropicProvider
from kader.providers.base import Message, ModelConfig


def demo_basic_invocation():
    """Demonstrate basic synchronous invocation."""
    print("\n=== Basic Anthropic Invocation Demo ===")

    # Initialize the provider with a Claude model
    # Note: Set ANTHROPIC_API_KEY in ~/.kader/.env
    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

    # Anthropic supports rich system prompts — passed as a separate parameter internally
    messages = [
        Message.system("You are a helpful assistant that responds concisely."),
        Message.user("What are the key advantages of Claude AI models?"),
    ]

    try:
        response = provider.invoke(messages)

        print(f"Model: {response.model}")
        print(f"Content: {response.content}")
        print(f"Prompt tokens: {response.usage.prompt_tokens}")
        print(f"Completion tokens: {response.usage.completion_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
        print(f"Finish reason: {response.finish_reason}")

        if response.cost:
            print(f"Cost: {response.cost.format()}")

        print(f"Total usage tracked: {provider.total_usage.total_tokens} tokens")

    except Exception as e:
        print(f"Error during invocation: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_streaming():
    """Demonstrate streaming responses."""
    print("\n=== Anthropic Streaming Demo ===")

    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

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
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_async_invocation():
    """Demonstrate asynchronous invocation."""
    print("\n=== Anthropic Async Invocation Demo ===")

    async def async_demo():
        provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

        messages = [
            Message.system("You are a helpful assistant."),
            Message.user(
                "What is the difference between Claude Sonnet and Claude Opus?"
            ),
        ]

        try:
            response = await provider.ainvoke(messages)

            print(f"Model: {response.model}")
            print(f"Content: {response.content}")
            print(f"Tokens: {response.usage.total_tokens}")
            print(f"Finish reason: {response.finish_reason}")

        except Exception as e:
            print(f"Error during async invocation: {e}")
            print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")

    asyncio.run(async_demo())


def demo_async_streaming():
    """Demonstrate asynchronous streaming."""
    print("\n=== Anthropic Async Streaming Demo ===")

    async def async_stream_demo():
        provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

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
            print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")

    asyncio.run(async_stream_demo())


def demo_configuration():
    """Demonstrate using different configurations."""
    print("\n=== Anthropic Configuration Demo ===")

    # Claude supports temperature, top_p, top_k, max_tokens, stop_sequences
    default_config = ModelConfig(
        temperature=0.7,
        max_tokens=200,
        top_p=0.9,
    )

    provider = AnthropicProvider(
        model="claude-sonnet-4-5-20250929", default_config=default_config
    )

    messages = [Message.user("Tell me a creative fact about space.")]

    try:
        response = provider.invoke(messages)
        print(f"Using default config - Content: {response.content[:120]}...")
        print(f"Tokens: {response.usage.total_tokens}")

        # Override configuration for this specific call
        creative_config = ModelConfig(
            temperature=1.0,
            max_tokens=300,
            top_k=50,  # Claude supports top_k natively
        )

        messages = [Message.user("Generate an original haiku about technology.")]
        response = provider.invoke(messages, config=creative_config)
        print(f"\nUsing creative config (top_k=50) - Content:\n{response.content}")

    except Exception as e:
        print(f"Error during configuration demo: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_conversation_history():
    """Demonstrate maintaining conversation context."""
    print("\n=== Anthropic Conversation History Demo ===")

    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

    # Simulate a multi-turn conversation
    # Note: Anthropic's 200K context window handles long conversations well
    conversation = [
        Message.system("You are a helpful coding assistant."),
        Message.user("What is Python used for?"),
        Message.assistant(
            "Python is a versatile programming language used for web development, "
            "data science, AI/ML, automation, scripting, and much more."
        ),
        Message.user("Can you give me a simple Python example for data analysis?"),
    ]

    try:
        response = provider.invoke(conversation)
        print(f"Response to follow-up:\n{response.content}")
        print(f"Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print(f"Error during conversation demo: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_tool_calling():
    """Demonstrate tool/function calling with Claude."""
    print("\n=== Anthropic Tool Calling Demo ===")

    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

    # Define tools in OpenAI format — the provider converts to Anthropic format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. 'London, UK'",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    config = ModelConfig(tools=tools, tool_choice="auto")

    messages = [
        Message.system("You are a helpful assistant with access to weather tools."),
        Message.user("What's the weather like in Paris right now?"),
    ]

    try:
        response = provider.invoke(messages, config=config)

        print(f"Finish reason: {response.finish_reason}")

        if response.has_tool_calls:
            print(f"Claude wants to call {len(response.tool_calls)} tool(s):")
            for tc in response.tool_calls:
                print(f"  Tool: {tc['function']['name']}")
                print(f"  Args: {tc['function']['arguments']}")
                print(f"  Call ID: {tc['id']}")

            # --- Simulate tool result and continue the conversation ---
            # In a real app you would actually call the tool here
            tool_result = "15°C, partly cloudy with light winds"

            # Build follow-up conversation with tool results
            followup = messages + [
                response.to_message(),  # Claude's assistant message with tool_use block
                Message.tool(response.tool_calls[0]["id"], tool_result),
            ]

            final_response = provider.invoke(followup)
            print(f"\nFinal response after tool result:\n{final_response.content}")
        else:
            print(f"Direct response (no tool call): {response.content}")

    except Exception as e:
        print(f"Error during tool calling demo: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_list_models():
    """Demonstrate dynamically listing available Anthropic models."""
    print("\n=== Anthropic List Models Demo ===")

    try:
        # Get all available models via the Anthropic models API (no hardcoding!)
        models = AnthropicProvider.get_supported_models()

        print(f"Found {len(models)} Anthropic Claude models:")
        for model in models:
            print(f"  - {model}")

    except Exception as e:
        print(f"Error listing models: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_model_info():
    """Demonstrate getting model information."""
    print("\n=== Anthropic Model Info Demo ===")

    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

    try:
        model_info = provider.get_model_info()

        if model_info:
            print(f"Model Name: {model_info.name}")
            print(f"Provider: {model_info.provider}")
            print(f"Context Window: {model_info.context_window:,} tokens")
            print(f"Supports Tools: {model_info.supports_tools}")
            print(f"Supports Streaming: {model_info.supports_streaming}")
            print(f"Supports Vision: {model_info.supports_vision}")
            if model_info.pricing:
                print(
                    f"Input cost: ${model_info.pricing.input_cost_per_million}/M tokens"
                )
                print(
                    f"Output cost: ${model_info.pricing.output_cost_per_million}/M tokens"
                )
        else:
            print("Could not retrieve model info.")

    except Exception as e:
        print(f"Error getting model info: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_token_counting():
    """Demonstrate token counting using the Anthropic token counting API."""
    print("\n=== Anthropic Token Counting Demo ===")

    # Token counting uses client.beta.messages.count_tokens() — a free API call
    provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")

    try:
        # Count tokens in a plain string
        text = "Hello, how are you today? I'm looking forward to our conversation."
        token_count = provider.count_tokens(text)
        print(f"Text: '{text}'")
        print(f"Token count: {token_count}")

        # Count tokens in a structured message list (including system prompt)
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("What is the meaning of life?"),
        ]
        msg_token_count = provider.count_tokens(messages)
        print(f"\nMessages token count (with system): {msg_token_count}")

    except Exception as e:
        print(f"Error counting tokens: {e}")
        print("Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def demo_cost_estimation():
    """Demonstrate cost estimation with Anthropic pricing."""
    print("\n=== Anthropic Cost Estimation Demo ===")

    # Compare costs across Claude model tiers
    models = {
        "claude-3-haiku-20240307": "Haiku (fastest, cheapest)",
        "claude-sonnet-4-5-20250929": "Sonnet (balanced)",
        "claude-3-opus-20240229": "Opus (most capable)",
    }

    from kader.providers.base import Usage

    usage = Usage(prompt_tokens=10_000, completion_tokens=2_000)

    print(f"Estimated cost for {usage.prompt_tokens:,} input + {usage.completion_tokens:,} output tokens:\n")

    for model_id, label in models.items():
        provider = AnthropicProvider(model=model_id)
        cost = provider.estimate_cost(usage)
        print(f"  {label}")
        print(f"    Input:  ${cost.input_cost:.4f}")
        print(f"    Output: ${cost.output_cost:.4f}")
        print(f"    Total:  {cost.format()}\n")

    # Live cost from an actual API call
    print("Live API call cost:")
    live_provider = AnthropicProvider(model="claude-sonnet-4-5-20250929")
    messages = [Message.user("Write a one-sentence summary of machine learning.")]

    try:
        response = live_provider.invoke(messages)

        print(f"  Response: {response.content[:80]}...")
        print(f"  Prompt tokens:     {response.usage.prompt_tokens}")
        print(f"  Completion tokens: {response.usage.completion_tokens}")
        if response.cost:
            print(f"  Input cost:  ${response.cost.input_cost:.6f}")
            print(f"  Output cost: ${response.cost.output_cost:.6f}")
            print(f"  Total cost:  {response.cost.format()}")

    except Exception as e:
        print(f"  Error during API call: {e}")
        print("  Make sure ANTHROPIC_API_KEY is set in ~/.kader/.env")


def main():
    """Run all Anthropic provider demos."""
    print("Kader Anthropic Provider Examples")
    print("=" * 40)

    print("\nAPI Key Setup:")
    print("  Add your ANTHROPIC_API_KEY to ~/.kader/.env:")
    print("  ANTHROPIC_API_KEY='your-api-key-here'")
    print("\n  Get your API key from: https://console.anthropic.com/settings/keys")
    print("\nAvailable Model Tiers:")
    print("  - claude-3-haiku-20240307        (fastest, lowest cost)")
    print("  - claude-3-5-haiku-20241022      (fast, affordable)")
    print("  - claude-3-5-sonnet-20241022     (balanced performance/cost)")
    print("  - claude-3-7-sonnet-20250219     (extended thinking)")
    print("  - claude-3-opus-20240229         (highest capability)")
    print("\nAll models support 200K token context windows.")

    demo_basic_invocation()
    demo_streaming()
    demo_async_invocation()
    demo_async_streaming()
    demo_configuration()
    demo_conversation_history()
    demo_tool_calling()
    demo_list_models()
    demo_model_info()
    demo_token_counting()
    demo_cost_estimation()

    print("\n[OK] All Anthropic demos completed!")


if __name__ == "__main__":
    main()
