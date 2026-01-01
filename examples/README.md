# Kader Examples

This directory contains example scripts demonstrating various features of the Kader framework.

## Available Examples

### Memory Example
```bash
python -m examples.memory_example
```
Demonstrates memory management features including:
- Agent state persistence
- Session management
- Conversation windowing
- Request-scoped state

### Ollama Provider Example
```bash
python -m examples.ollama_example
```
Demonstrates how to use the Ollama provider for:
- Basic LLM invocation
- Streaming responses (sync and async)
- Configuration options
- Conversation history
- Error handling

**Note**: To run the Ollama example, you need to have Ollama installed and running with at least one model pulled (e.g., `ollama pull gpt-oss:120b-cloud` or any other model you prefer to use).

### Tools Example
```bash
python -m examples.tools_example
```
Demonstrates how to use the Kader tools for:
- File system operations (read, write, search, grep, etc.)
- Web search and content fetching
- Command execution with OS validation
- RAG (Retrieval Augmented Generation) tools
- Tool registry and management
- Creating custom tools
- Asynchronous tool operations

## Running Examples

To run any example:
1. Make sure you have the project dependencies installed: `uv sync`
2. Run the example directly: `python -m examples.<example_name>`

## Adding New Examples

When adding new examples:
1. Create a new Python file in this directory
2. Follow the same documentation pattern as existing examples
3. Include proper error handling and informative output
4. Add documentation to this README