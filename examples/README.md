# Kader Examples

This directory contains example scripts demonstrating various features of the Kader framework.

## Available Examples

### Memory Example
```bash
uv run python -m examples.memory_example
```
Demonstrates memory management features including:
- Agent state persistence
- Session management
- Conversation windowing
- Request-scoped state

### Ollama Provider Example
```bash
uv run python -m examples.ollama_example
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
uv run python -m examples.tools_example
```
Demonstrates how to use the Kader tools for:
- File system operations (read, write, search, grep, etc.)
- Web search and content fetching
- Command execution with OS validation
- RAG (Retrieval Augmented Generation) tools
- Tool registry and management
- Creating custom tools
- Asynchronous tool operations

### ReAct Agent Example
```bash
uv run python -m examples.react_agent_example
```
Demonstrates how to use the ReAct (Reasoning and Acting) agent with:
- Interactive chat interface
- Tool integration and execution
- Memory management
- Session persistence
- Asynchronous operations

### Simple Agent Example
```bash
uv run python -m examples.simple_agent
```
Demonstrates how to create a basic agent with:
- Custom system prompt using PromptBase
- Tool integration
- YAML saving/loading
- Basic execution flow

### Planning Agent Example
```bash
uv run python -m examples.planning_agent_example
```
Demonstrates how to use the Planning agent with:
- Task planning capabilities
- Tool integration
- Memory management
- Interactive chat interface

### Python Developer Example
```bash
uv run python -m examples.python_developer
```
Demonstrates a specialized Python expert agent loaded from YAML configuration including:
- Advanced Python features (decorators, async/await, type hints)
- Code generation with clean, idiomatic Python practices
- File system operations for development tasks
- Testing with pytest and performance optimization
- YAML-based agent configuration

### To-Do Agent Example
```bash
uv run python -m examples.todo_agent
```
Demonstrates an agent specialized for managing to-do lists and tasks with:
- Task management capabilities using TodoTool
- Creation and management of todo lists with specific IDs
- Marking tasks as completed
- Reading and displaying the status of all items
- YAML-based agent configuration

## Running Examples

To run any example:
1. Make sure you have the project dependencies installed: `uv sync`
2. Run the example using uv: `uv run python -m examples.<example_name>`

## Adding New Examples

When adding new examples:
1. Create a new Python file in this directory
2. Follow the same documentation pattern as existing examples
3. Include proper error handling and informative output
4. Add documentation to this README
5. Ensure the example is self-contained and well-commented