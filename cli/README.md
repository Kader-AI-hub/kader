# Kader CLI

A modern terminal-based AI coding assistant built with Python's [Textual](https://textual.textualize.io/) framework, powered by **Ollama**.

## Features

- 📁 **Directory Tree** - Sidebar showing current working directory
- 💬 **Conversation View** - Markdown-rendered chat history
- ⏳ **Streaming Responses** - Real-time LLM response streaming
- 🎨 **Color Themes** - 4 themes (dark, ocean, forest, sunset)
- 🤖 **Ollama Integration** - Uses local Ollama models

## Prerequisites

- [Ollama](https://ollama.ai/) running locally
- Model `gpt-oss:120b-cloud` (or update `DEFAULT_MODEL` in `utils.py`)

## Quick Start

```bash
cd e:\kader
uv run -m cli
```

## Commands

| Command | Description |
|---------|-------------|
| `/models` | Show available Ollama models |
| `/theme` | Cycle color themes |
| `/help` | Show command reference |
| `/clear` | Clear conversation |
| `/new` | Start new conversation |
| `/exit` | Exit the CLI |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear conversation |
| `Ctrl+T` | Cycle theme |
| `Tab` | Navigate panels |

## Project Structure

```
cli/
├── app.py          # Main application (OllamaProvider integration)
├── app.tcss        # Styles (TCSS)
├── utils.py        # Constants and helpers
├── __init__.py     # Package exports
├── __main__.py     # Entry point
└── widgets/
    ├── conversation.py  # Chat display
    └── loading.py       # Spinner animation
```

## Changing the Model

Edit `DEFAULT_MODEL` in `utils.py`:

```python
DEFAULT_MODEL = "llama3.2"  # or any Ollama model
```

## Development

Run with live CSS reloading:

```bash
uv run textual run --dev cli.app:KaderApp
```
