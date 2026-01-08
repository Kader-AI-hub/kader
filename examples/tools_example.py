"""
Kader Tools Example

Demonstrates how to use the Kader tools for various agentic operations:
- File system operations
- Web search and fetch
- Command execution
- RAG (Retrieval Augmented Generation)
- Tool registry and management
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.tools import (
    # Command execution
    CommandExecutorTool,
    GrepTool,
    # RAG tools
    RAGIndex,
    RAGSearchTool,
    ReadDirectoryTool,
    # File system tools
    ReadFileTool,
    # Core components
    ToolRegistry,
    WebFetchTool,
    WebSearchTool,
    WriteFileTool,
    tool,
)


def demo_file_system_tools():
    """Demonstrate file system tools."""
    print("\n=== File System Tools Demo ===")

    # ReadFileTool
    print("\n--- ReadFileTool ---")
    read_tool = ReadFileTool()

    # Create a test file first
    test_file = Path("test_file.txt")
    test_content = "This is a test file for the Kader tools example.\nIt has multiple lines.\nThis is the third line."
    test_file.write_text(test_content)

    try:
        content = read_tool.execute(path="test_file.txt")
        print(f"Read file content:\n{content}")
    except Exception as e:
        print(f"Error reading file: {e}")

    # ReadDirectoryTool
    print("\n--- ReadDirectoryTool ---")
    dir_tool = ReadDirectoryTool()
    try:
        files = dir_tool.execute(path=".")
        print(f"Directory contents: {files[:100]}...")  # Show first 100 chars
    except Exception as e:
        print(f"Error reading directory: {e}")

    # WriteFileTool
    print("\n--- WriteFileTool ---")
    write_tool = WriteFileTool()
    try:
        result = write_tool.execute(
            path="output_test.txt", content="This is content written by WriteFileTool."
        )
        print(f"Write result: {result}")
    except Exception as e:
        print(f"Error writing file: {e}")

    # GrepTool
    print("\n--- GrepTool ---")
    grep_tool = GrepTool()
    try:
        results = grep_tool.execute(pattern="test", path=".")
        print(f"Grep results: {str(results)[:200]}...")  # Show first 200 chars
    except Exception as e:
        print(f"Error with grep: {e}")

    # Clean up test files
    if test_file.exists():
        test_file.unlink()
    output_file = Path("output_test.txt")
    if output_file.exists():
        output_file.unlink()


def demo_web_tools():
    """Demonstrate web tools."""
    print("\n=== Web Tools Demo ===")

    # WebSearchTool
    print("\n--- WebSearchTool ---")
    search_tool = WebSearchTool()
    try:
        # Search for information about agent frameworks
        results = search_tool.execute(query="what is an agent framework", limit=3)
        print(f"Search results type: {type(results)}")

        # Handle different possible return types
        if isinstance(results, list):
            print(f"Search results: {len(results)} results found")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                # Handle both dict and object types
                if isinstance(result, dict):
                    title = (
                        result.get("title", "No title")
                        if hasattr(result, "get")
                        else "No title"
                    )
                elif hasattr(result, "__dict__"):
                    # If it's an object with attributes, try to get title
                    title = (
                        getattr(
                            result,
                            "title",
                            getattr(result, "get", lambda: "No title")(),
                        )
                        if hasattr(result, "title")
                        else "No title"
                    )
                else:
                    # For other types, just convert to string
                    title = str(result)[:50]  # Limit length
                print(f"  {i + 1}. {str(title)[:50]}...")
        else:
            print(
                f"Search returned unexpected type: {type(results)}, value: {str(results)[:200]}"
            )
    except Exception as e:
        print(f"Error with web search: {e}")
        import traceback

        traceback.print_exc()

    # WebFetchTool
    print("\n--- WebFetchTool ---")
    fetch_tool = WebFetchTool()
    try:
        # Fetch content from a simple test page - using a more reliable URL
        result = fetch_tool.execute(url="https://httpbin.org/robots.txt")
        print(f"Fetch result type: {type(result)}")
        print(f"Fetch result preview: {str(result)[:100]}...")
    except Exception as e:
        print(f"Error with web fetch: {e}")
        # Try a different URL as fallback
        try:
            result = fetch_tool.execute(url="https://example.com")
            print(f"Fetch result with fallback URL: {str(result)[:100]}...")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")


def demo_command_execution():
    """Demonstrate command execution tools."""
    print("\n=== Command Execution Demo ===")

    cmd_tool = CommandExecutorTool()

    try:
        # Execute a simple command based on the OS
        import platform

        if platform.system().lower() == "windows":
            command = "echo Hello from Windows"
        else:
            command = "echo Hello from Unix-like system"

        result = cmd_tool.execute(command=command, timeout=10)
        print(f"Command execution result: {result}")

        # Try to get current directory
        if platform.system().lower() == "windows":
            dir_cmd = "cd"
        else:
            dir_cmd = "pwd"

        dir_result = cmd_tool.execute(command=dir_cmd, timeout=10)
        print(f"Current directory: {dir_result}")

    except Exception as e:
        print(f"Error with command execution: {e}")


def demo_rag_tools():
    """Demonstrate RAG tools."""
    print("\n=== RAG Tools Demo ===")

    # Create a sample file for RAG indexing
    sample_file = Path("sample_document.txt")
    sample_content = """
Artificial Intelligence and Machine Learning are transforming various industries.
This document discusses the fundamentals of AI, including neural networks,
deep learning, and their applications in real-world scenarios.
Natural Language Processing is a key area of AI that focuses on the interaction
between computers and humans through natural language.
"""
    sample_file.write_text(sample_content)

    try:
        # Create RAG index
        print("--- Creating RAG Index ---")
        rag_index = RAGIndex(base_path=Path("."))
        # Note: Actual indexing might require more setup, so we'll just show the concept

        # RAGSearchTool
        print("\n--- RAGSearchTool ---")
        search_tool = RAGSearchTool()

        # This would normally search through indexed documents
        # For demo purposes, we'll show how it would be used
        print(
            "RAGSearchTool initialized. In a real scenario, this would search through indexed documents."
        )
        print(f"Tool name: {search_tool.name}")
        print(f"Tool description: {search_tool.description}")

    except Exception as e:
        print(f"Error with RAG tools: {e}")
    finally:
        # Clean up sample file
        if sample_file.exists():
            sample_file.unlink()


def demo_tool_registry():
    """Demonstrate tool registry functionality."""
    print("\n=== Tool Registry Demo ===")

    # Create a registry
    registry = ToolRegistry()

    # Register various tools
    tools_to_register = [
        ReadFileTool(),
        WriteFileTool(),
        WebSearchTool(),
        CommandExecutorTool(),
    ]

    for tool in tools_to_register:
        registry.register(tool)
        print(f"Registered tool: {tool.name}")

    print(f"\nTotal registered tools: {len(registry.tools)}")

    # List all tools
    print("\nAll registered tools:")
    for tool in registry.tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # Get a specific tool
    read_tool = registry.get("read_file")
    if read_tool:
        print(f"\nRetrieved 'read_file' tool: {read_tool.name}")

    # Get tools by category
    print(
        f"\nTool categories available: {set(tool.schema.category.value for tool in registry.tools)}"
    )


def demo_custom_tool():
    """Demonstrate creating and using a custom tool."""
    print("\n=== Custom Tool Demo ===")

    # Using the @tool decorator to create a custom tool
    @tool(
        name="get_current_time",
        description="Get the current time in ISO format",
        parameters=[
            {
                "name": "include_date",
                "type": "boolean",
                "description": "Whether to include the date in the response",
                "required": False,
                "default": True,
            }
        ],
    )
    def get_current_time_impl(include_date: bool = True) -> str:
        """Custom tool to get current time."""
        from datetime import datetime

        now = datetime.now()
        if include_date:
            return now.isoformat()
        else:
            return now.strftime("%H:%M:%S")

    # The @tool decorator returns a FunctionTool object
    get_current_time = get_current_time_impl

    # Execute the custom tool
    try:
        # Use the .execute() method on the FunctionTool object
        time_result = get_current_time.execute(include_date=True)
        print(f"Current time (with date): {time_result}")

        time_result_no_date = get_current_time.execute(include_date=False)
        print(f"Current time (time only): {time_result_no_date}")

        print(f"Custom tool name: {get_current_time.name}")
        print(f"Custom tool description: {get_current_time.description}")

    except Exception as e:
        print(f"Error with custom tool: {e}")
        import traceback

        traceback.print_exc()


def demo_async_operations():
    """Demonstrate asynchronous tool operations."""
    print("\n=== Async Operations Demo ===")

    async def async_demo():
        # Create tools
        read_tool = ReadFileTool()
        write_tool = WriteFileTool()

        # Async write
        write_task = write_tool.aexecute(
            path="async_test.txt", content="This file was written asynchronously."
        )

        # Wait for write to complete
        write_result = await write_task
        print(f"Async write result: {write_result}")

        # Async read
        read_task = read_tool.aexecute(path="async_test.txt")
        read_result = await read_task
        print(f"Async read result preview: {read_result[:50]}...")

        # Clean up
        import os

        if os.path.exists("async_test.txt"):
            os.remove("async_test.txt")

    try:
        asyncio.run(async_demo())
    except Exception as e:
        print(f"Error with async operations: {e}")


def demo_tool_parameters():
    """Demonstrate tool parameter validation and usage."""
    print("\n=== Tool Parameters Demo ===")

    # Show parameters for a tool
    read_tool = ReadFileTool()
    print(f"Tool: {read_tool.name}")
    print(f"Description: {read_tool.description}")
    print("Parameters:")
    for param in read_tool.schema.parameters:
        print(
            f"  - {param.name}: {param.description} (type: {param.type}, required: {param.required})"
        )

    # Show schema in JSON format
    schema = read_tool.schema
    print(f"\nTool schema (first 200 chars): {str(schema)[:200]}...")


def main():
    """Run all tool demos."""
    print("Kader Tools Examples")
    print("=" * 40)

    print("\nThis example demonstrates the various tools available in Kader.")
    print(
        "Note: Some tools may require specific setup (e.g., internet access for web tools,"
    )
    print("proper file permissions for file tools, etc.)")

    demo_file_system_tools()
    demo_web_tools()
    demo_command_execution()
    demo_rag_tools()
    demo_tool_registry()
    demo_custom_tool()
    demo_async_operations()
    demo_tool_parameters()

    print("\n[OK] All tool demos completed!")


if __name__ == "__main__":
    main()
