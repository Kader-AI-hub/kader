# Providers

Kader supports multiple LLM providers through a unified interface.

## Supported Providers

| Provider | Description |
|----------|-------------|
| OllamaProvider | Local LLM inference |
| GoogleProvider | Google Gemini models |
| AnthropicProvider | Anthropic Claude models |
| MistralProvider | Mistral AI models |
| OpenAICompatibleProvider | OpenAI and compatible APIs |

## OllamaProvider

For local LLM inference. Best for privacy, speed, and offline capability.

```python
from kader.providers import OllamaProvider, Message

provider = OllamaProvider(
    model="llama3.2",
    base_url="http://localhost:11434",
    timeout=120,
)

messages = [
    Message.system("You are helpful."),
    Message.user("What is Ollama?"),
]
response = provider.invoke(messages)
print(response.content)

# Streaming
for chunk in provider.stream(messages):
    print(chunk.content, end="", flush=True)

# Async
import asyncio
response = await provider.ainvoke(messages)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name |
| `base_url` | str | "http://localhost:11434" | API endpoint |
| `timeout` | int | 120 | Request timeout |

## GoogleProvider

Google Gemini models via the Google GenAI SDK.

```python
from kader.providers import GoogleProvider, Message

provider = GoogleProvider(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=2048,
)

response = provider.invoke([Message.user("Hello from Gemini!")])
```

Requires `GEMINI_API_KEY` in environment.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | "gemini-2.0-flash" | Model name |
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 2048 | Maximum tokens |

## AnthropicProvider

Anthropic Claude models via the Anthropic SDK.

```python
from kader.providers import AnthropicProvider, Message

provider = AnthropicProvider(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
)

response = provider.invoke([Message.user("Hello from Claude!")])
```

Requires `ANTHROPIC_API_KEY` in environment.

## MistralProvider

Mistral AI models for cloud inference.

```python
from kader.providers import MistralProvider, Message

provider = MistralProvider(
    model="mistral-large-latest",
    temperature=0.7,
)

response = provider.invoke([Message.user("Hello from Mistral!")])
```

Requires `MISTRAL_API_KEY` in environment.

## OpenAICompatibleProvider

Connect to OpenAI, Groq, OpenRouter, Moonshot AI, and other OpenAI-compatible APIs.

```python
from kader.providers import OpenAICompatibleProvider, Message, OpenAIProviderConfig

# Standard OpenAI
openai_provider = OpenAICompatibleProvider(
    model="gpt-4o",
    config=OpenAIProviderConfig(
        api_key="your-openai-key",
    )
)

# Groq (fast inference)
groq_provider = OpenAICompatibleProvider(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    config=OpenAIProviderConfig(
        api_key="your-groq-key",
    )
)

# OpenRouter (200+ models)
openrouter_provider = OpenAICompatibleProvider(
    model="anthropic/claude-3.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    config=OpenAIProviderConfig(
        api_key="your-openrouter-key",
    )
)

# Moonshot AI (Kimi)
moonshot_provider = OpenAICompatibleProvider(
    model="kimi-k2.5",
    base_url="https://api.moonshot.cn/v1",
    config=OpenAIProviderConfig(
        api_key="your-moonshot-key",
    )
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | required | Model name |
| `base_url` | str | "https://api.openai.com/v1" | API endpoint |
| `config` | OpenAIProviderConfig | required | API configuration |

## Message Helper

Use the `Message` class to create messages:

```python
from kader.providers import Message

messages = [
    Message.system("You are a helpful assistant."),
    Message.user("What is Python?"),
    Message.assistant("Python is a programming language."),
    Message.user("Tell me more."),
]
```
