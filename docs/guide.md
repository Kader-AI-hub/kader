# Getting Started Guide

This guide will help you get started with Kader - from installation to running your first agent.

## Prerequisites

- Python 3.11 or higher
- [Ollama](https://ollama.ai/) (optional, for local LLMs)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

## Installation

### Using uv tool (recommended)

Install Kader globally using uv toolchain:

```bash
# Install Kader globally
uv tool install kader

# Run the CLI
kader
```

### Clone and Run Locally

Clone the repository and run directly:

```bash
# Clone the repository
git clone https://github.com/Kader-AI-hub/kader.git
cd kader

# Install dependencies
uv sync

# Run the CLI
uv run python -m cli
```

## Provider Setup

### Ollama (Local)

Install from https://ollama.ai and pull a model:

```bash
ollama pull llama3.2
ollama pull qwen2.5
```

### Cloud Providers

Create a `.env` file in `~/.kader/.env`:

```bash
# Ollama Cloud (get from https://ollama.com/settings)
OLLAMA_API_KEY='your-api-key'

# Google Gemini
GEMINI_API_KEY='your-api-key'

# Mistral
MISTRAL_API_KEY='your-api-key'

# Anthropic
ANTHROPIC_API_KEY='your-api-key'

# OpenAI
OPENAI_API_KEY='your-api-key'

# OpenAI-Compatible Providers
MOONSHOT_API_KEY='your-api-key'
GROQ_API_KEY='your-api-key'
OPENROUTER_API_KEY='your-api-key'
```

## Running the CLI

```bash
# Using uv tool (recommended)
kader

# Or using uv run
uv run python -m cli
```

## Updating Kader

### Using uv tool

Update Kader to the latest version:

```bash
# Update Kader to latest version
uv tool update kader

# Or reinstall to get the latest
uv tool uninstall kader
uv tool install kader
```

### Clone and Run Locally

If running from clone:

```bash
cd kader
git pull origin main
uv sync
```

## First Steps in CLI

Once the CLI is running:

1. Type any question to start chatting with the agent
2. Use `/help` to see available commands
3. Use `/models` to switch models per agent (main or sub)
4. Run terminal commands directly by prefixing with `!` (e.g., `!ls -la`)

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command reference |
| `/models` | Switch models per agent (main/sub) |
| `/clear` | Clear conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/skills` | List loaded skills |
| `/commands` | List special commands |
| `/cost` | Show usage costs |
| `/init` | Initialize .kader directory |
| `/exit` | Exit the CLI |
| `!cmd` | Run terminal command |

## Next Steps

- [Core Framework Documentation](core-framework/index.md) - Learn about agents, providers, and tools
- [CLI Reference](cli/index.md) - Detailed CLI commands and features
- [Configuration Guide](configuration.md) - Environment variables and YAML config
