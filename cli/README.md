# Kader CLI

A modern terminal-based AI coding assistant built with Python's [Textual](https://textual.textualize.io/) framework, powered by **ReActAgent** with tool execution capabilities.

## Features

- 🤖 **ReAct Agent** - Intelligent agent with reasoning and tool execution
- 🛠️ **Built-in Tools** - File system, command execution, web search
- 📁 **Directory Tree** - Auto-refreshing sidebar showing current working directory
- 💬 **Conversation View** - Markdown-rendered chat history
- 💾 **Session Persistence** - Save and load conversation sessions
- 🎨 **Color Themes** - 4 themes (dark, ocean, forest, sunset)

## Prerequisites

- [Ollama](https://ollama.ai/) running locally
- Model `gpt-oss:120b-cloud` (or update `DEFAULT_MODEL` in `utils.py`)

## Quick Start

```bash
cd e:\kader
uv run python -m cli
```

## Commands

| Command | Description |
|---------|-------------|
| `/help` | Show command reference |
| `/models` | Show available Ollama models |
| `/theme` | Cycle color themes |
| `/clear` | Clear conversation |
| `/save` | Save current session |
| `/load <id>` | Load a saved session |
| `/sessions` | List saved sessions |
| `/refresh` | Refresh file tree |
| `/exit` | Exit the CLI |

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear conversation |
| `Ctrl+T` | Cycle theme |
| `Ctrl+S` | Save session |
| `Ctrl+R` | Refresh file tree |
| `Tab` | Navigate panels |

## Input Editing

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Copy selected text |
| `Ctrl+V` | Paste from clipboard |
| `Ctrl+A` | Select all text |
| Click+Drag | Select text |

## Session Management

Sessions are saved to `~/.kader/sessions/`. Use:

- `/save` to save current conversation
- `/sessions` to list all saved sessions
- `/load <session_id>` to restore a session

## Project Structure

```
cli/
├── app.py          # Main application (ReActAgent integration)
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
DEFAULT_MODEL = "gpt-oss:120b-cloud"
```

## Development

Run with live CSS reloading:

```bash
uv run textual run --dev cli.app:KaderApp
```
