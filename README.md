# Kader

Kader is an intelligent coding agent designed to assist with software development tasks.

## Features

- AI-powered code assistance using Ollama
- Interactive command-line interface
- Tool integration for enhanced capabilities
- Cross-platform compatibility

## Configuration

When the kader module is imported for the first time, it automatically:
1. Creates a `.kader` directory in your home directory (`~/.kader` on Unix systems, `%USERPROFILE%\.kader` on Windows)
2. Creates a `.env` file with the required configuration (including `OLLAMA_API_KEY=''`)
3. Loads all environment variables from the `.env` file into the application environment

## Environment Variables

The application automatically loads environment variables from `~/.kader/.env`:
- `OLLAMA_API_KEY`: API key for Ollama service (default: empty)
- Additional variables can be added to the `.env` file and will be automatically loaded

## Setup

1. Install the required dependencies
2. The `.kader` directory and `.env` file will be created automatically on first import
3. Configure your Ollama API key in `~/.kader/.env` if needed
4. Run the application

## Usage

Import and use the kader module in your Python scripts to access its functionality.

## Examples

Check out the examples directory for comprehensive demonstrations of Kader's features:

- `ollama_example.py`: Demonstrates how to use the Ollama provider for LLM interactions
- `memory_example.py`: Shows memory management capabilities
- `tools_example.py`: Demonstrates the various tools available in Kader

To run an example:
```bash
python -m examples.ollama_example
python -m examples.memory_example
python -m examples.tools_example
```