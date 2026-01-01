# Kader Tools Documentation

Kader Tools provides a versatile, provider-agnostic framework for creating and managing agentic tools that can be used with any LLM provider (OpenAI, Google, Anthropic, Mistral, and others).

## Table of Contents

1. [Available Tools](#available-tools)
2. [Using Tools](#using-tools)
3. [Creating Custom Tools](#creating-custom-tools)
4. [Tool Registry](#tool-registry)
5. [Provider Compatibility](#provider-compatibility)

## Available Tools

### File System Tools

- **ReadFileTool**: Read the contents of a file with optional line range selection
- **ReadDirectoryTool**: List directory contents with recursive option
- **WriteFileTool**: Write content to a file, with optional directory creation
- **ReplaceLinesTool**: Replace or insert lines in a file
- **SearchInDirectoryTool**: Search for files in a directory by name or content

### Web Tools

- **WebSearchTool**: Search the web for information with configurable result limit
- **WebFetchTool**: Fetch and extract text content from a web page

### Command Execution Tool

- **CommandExecutorTool**: Execute command line operations with OS-appropriate validation

### RAG (Retrieval Augmented Generation) Tools

- **RAGSearchTool**: Search through local files using semantic embeddings
- **RAGIndex**: Build and manage semantic indexes of your local files

## Using Tools

Here's how to use the available tools in your applications:

### Basic Tool Usage

```python
from kader.tools import ReadFileTool

# Create a tool instance
read_tool = ReadFileTool()

# Execute the tool
content = read_tool.execute(path="README.md")
print(content)
```

### Using with Tool Registry

```python
from kader.tools import ToolRegistry, ReadFileTool, WriteFileTool

# Create a registry and register tools
registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WriteFileTool())

# Get tools
available_tools = registry.tools
for tool in available_tools:
    print(f"Available tool: {tool.name}")

# Get a specific tool by name
read_tool = registry.get("read_file")
if read_tool:
    result = read_tool.execute(path="README.md")
    print(result)
```

### Asynchronous Execution

All tools support both synchronous and asynchronous execution:

```python
import asyncio
from kader.tools import ReadFileTool

async def async_example():
    tool = ReadFileTool()

    # Synchronous execution
    sync_result = tool.execute(path="README.md")

    # Asynchronous execution
    async_result = await tool.aexecute(path="README.md")

    return sync_result, async_result

# Run the async function
sync_result, async_result = asyncio.run(async_example())
```

## Creating Custom Tools

Creating custom tools is straightforward with the `BaseTool` class:

### Basic Custom Tool

```python
from kader.tools.base import BaseTool, ParameterSchema, ToolCategory

class GreetingTool(BaseTool[str]):
    """
    A simple tool that generates a personalized greeting.
    """
    
    def __init__(self):
        super().__init__(
            name="greeting_tool",
            description="Generate a personalized greeting message",
            parameters=[
                ParameterSchema(
                    name="name",
                    type="string",
                    description="The name to greet",
                ),
                ParameterSchema(
                    name="greeting_type",
                    type="string",
                    description="Type of greeting (formal, casual, friendly)",
                    required=False,
                    default="casual",
                    enum=["formal", "casual", "friendly"]
                ),
            ],
            category=ToolCategory.UTILITY,
        )
    
    def execute(self, name: str, greeting_type: str = "casual") -> str:
        """
        Execute the greeting tool.
        """
        greetings = {
            "formal": f"Good day, {name}!",
            "casual": f"Hello, {name}!",
            "friendly": f"Hey there, {name}! How's it going?"
        }
        
        return greetings.get(greeting_type, greetings["casual"])
    
    async def aexecute(self, name: str, greeting_type: str = "casual") -> str:
        """
        Asynchronous execution of the tool.
        """
        import asyncio
        # Simulate async operation if needed
        await asyncio.sleep(0.01)  # Placeholder for actual async work
        return self.execute(name, greeting_type)

# Using the custom tool
tool = GreetingTool()
print(tool.execute(name="Alice", greeting_type="friendly"))
# Output: Hey there, Alice! How's it going?
```

### Tool with Complex Return Types

```python
from typing import Dict, Any
from kader.tools.base import BaseTool, ParameterSchema, ToolCategory

class MathCalculatorTool(BaseTool[Dict[str, Any]]):
    """
    A tool that performs basic mathematical operations.
    """
    
    def __init__(self):
        super().__init__(
            name="math_calculator",
            description="Perform basic mathematical operations",
            parameters=[
                ParameterSchema(
                    name="operation",
                    type="string",
                    description="The operation to perform (add, subtract, multiply, divide)",
                    enum=["add", "subtract", "multiply", "divide"]
                ),
                ParameterSchema(
                    name="a",
                    type="number",
                    description="First operand"
                ),
                ParameterSchema(
                    name="b",
                    type="number",
                    description="Second operand"
                )
            ],
            category=ToolCategory.UTILITY,
        )
    
    def execute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Execute the mathematical operation.
        """
        operations = {
            "add": lambda x, y: x + y,
            "subtract": lambda x, y: x - y,
            "multiply": lambda x, y: x * y,
            "divide": lambda x, y: x / y if y != 0 else float('inf')
        }
        
        if operation not in operations:
            return {
                "error": f"Invalid operation '{operation}'. Supported operations: {list(operations.keys())}",
                "result": None
            }
        
        try:
            result = operations[operation](a, b)
            return {
                "operation": f"{a} {operation} {b}",
                "result": result,
                "success": True
            }
        except Exception as e:
            return {
                "error": f"Error during calculation: {str(e)}",
                "result": None,
                "success": False
            }
    
    async def aexecute(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Asynchronous execution of the calculator.
        """
        import asyncio
        await asyncio.sleep(0.01)  # Simulate async operation
        return self.execute(operation, a, b)
```

### Command Execution Tool Example

The CommandExecutorTool included in this package demonstrates advanced features:

```python
from kader.tools.exec_commands import CommandExecutorTool

# Create the command executor tool
cmd_tool = CommandExecutorTool()

# Execute a command (with OS validation)
result = cmd_tool.execute(command="echo 'Hello, World!'", timeout=10)
print(result)

# On Windows, this would warn about Unix-specific commands:
result = cmd_tool.execute(command="ls -la")  # Shows validation error
print(result)
```

## Tool Registry

The `ToolRegistry` provides a centralized way to manage multiple tools and execute them by name:

### Basic Registry Usage

```python
from kader.tools import ToolRegistry, ReadFileTool, WriteFileTool, CommandExecutorTool

# Create a registry
registry = ToolRegistry()

# Register individual tools
registry.register(ReadFileTool())
registry.register(WriteFileTool())
registry.register(CommandExecutorTool())

# Or register multiple tools at once
from kader.tools.filesys import get_filesystem_tools
filesystem_tools = get_filesystem_tools()
for tool in filesystem_tools:
    registry.register(tool)

# Get tools by name
read_tool = registry.get("read_file")
if read_tool:
    result = read_tool.execute(path="README.md")
    print(result)

# Get all registered tool names
tool_names = registry.names
print("Registered tools:", tool_names)

# Get all tools
all_tools = registry.tools
print(f"Total tools: {len(all_tools)}")

# Get tool schemas for LLM provider integration
schemas = registry.to_provider_format(provider="openai")
```

### Registry with Provider Compatibility

```python
from kader.tools import ToolRegistry
from kader.tools.filesys import ReadFileTool
from kader.tools.web import WebSearchTool

# Create registry and register tools
registry = ToolRegistry()
registry.register(ReadFileTool())
registry.register(WebSearchTool())

# Get schemas formatted for different providers
openai_schemas = registry.to_provider_format("openai")
anthropic_schemas = registry.to_provider_format("anthropic")
google_schemas = registry.to_provider_format("google")
mistral_schemas = registry.to_provider_format("mistral")
ollama_schemas = registry.to_provider_format("ollama")

# Use with your LLM provider
# Example with OpenAI:
# openai_client.chat.completions.create(
#     model="gpt-4",
#     messages=[...],
#     tools=openai_schemas,
#     tool_choice="auto",
# )
```

### Working with Tool Results

```python
from kader.tools import ToolRegistry, ReadFileTool, ToolCall
from kader.tools.base import ToolResult

registry = ToolRegistry()
registry.register(ReadFileTool())

# Create a tool call (this would typically come from an LLM response)
tool_call = ToolCall(
    id="call_123",
    name="read_file",
    arguments={"path": "README.md"},
    raw_arguments='{"path": "README.md"}'
)

# Execute the tool call through the registry
result = registry.run(tool_call)

# ToolResult has status, content, and optional data
print(f"Status: {result.status}")
print(f"Content: {result.content}")

# Convert to provider-specific format
openai_format = result.to_provider_format("openai")
anthropic_format = result.to_provider_format("anthropic")

# Working with the CommandExecutorTool specifically
cmd_tool_call = ToolCall(
    id="cmd_call_456",
    name="execute_command",
    arguments={"command": "echo Hello from command executor"},
    raw_arguments='{"command": "echo Hello from command executor"}'
)

cmd_result = registry.run(cmd_tool_call)
print(f"Command execution result: {cmd_result.content}")
```

## Provider Compatibility

Kader Tools are designed to work with multiple LLM providers seamlessly:

### OpenAI Integration

```python
from kader.tools import ToolRegistry
from kader.tools.filesys import ReadFileTool

registry = ToolRegistry()
registry.add_tool(ReadFileTool())

# Get tools in OpenAI format
openai_tools = registry.get_tool_schemas(provider="openai")

# Use with OpenAI
import openai
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Read the README.md file"}
    ],
    tools=openai_tools,
    tool_choice="auto"
)

# Process tool calls from response
for choice in response.choices:
    if choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            result = registry.execute_tool_from_call(tool_call, "openai")
            print(result.content)
```

### Anthropic Integration

```python
from kader.tools import ToolRegistry
from kader.tools.filesys import ReadFileTool

registry = ToolRegistry()
registry.add_tool(ReadFileTool())

# Get tools in Anthropic format
anthropic_tools = registry.get_tool_schemas(provider="anthropic")

# Use with Anthropic
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Read the configuration file"}
    ],
    tools=anthropic_tools
)

# Process tool use blocks from response
for content_block in response.content:
    if content_block.type == "tool_use":
        result = registry.execute_tool_from_call(content_block, "anthropic")
        print(result.content)
```

### Google (Gemini) Integration

```python
from kader.tools import ToolRegistry
from kader.tools.filesys import ReadFileTool

registry = ToolRegistry()
registry.add_tool(ReadFileTool())

# Get tools in Google format
google_tools = registry.get_tool_schemas(provider="google")

# Use with Google Gemini
import google.generativeai as genai
model = genai.GenerativeModel(model_name="gemini-pro", tools=google_tools)

response = model.generate_content("Read the project's README file")

# Process function calls from response
for part in response.parts:
    if hasattr(part, 'function_call'):
        result = registry.execute_tool_from_call(part.function_call, "google")
        print(result.content)
```

## Best Practices

1. **Always validate inputs** - Use the parameter schema validation provided by BaseTool
2. **Handle errors gracefully** - Implement proper error handling in your tool's execute method
3. **Follow the category system** - Use appropriate ToolCategory values for organization
4. **Provide clear descriptions** - Write clear, concise descriptions for your tools and their parameters
5. **Consider security** - For file system operations, always validate paths to prevent directory traversal
6. **Support both sync and async** - Implement both `execute` and `aexecute` methods
7. **Use structured return types** - When possible, return structured data that can be processed by LLMs

## File System Security

All file system tools operate relative to the current working directory (CWD) for security. Paths are validated to ensure they don't escape the allowed directory:

```python
from pathlib import Path
from kader.tools.filesys import ReadFileTool

# Tools default to CWD but you can specify a different base
custom_tool = ReadFileTool(base_path=Path("/safe/directory"))

# This will raise an error if path attempts to escape the base path:
# content = custom_tool.execute(path="../../../forbidden.txt")  # ValueError!
```

For more information on creating custom tools or using the registry, check out the examples directory or the source code documentation.