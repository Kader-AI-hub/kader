"""
Kader Tools - Agentic tool definitions.

This module provides a provider-agnostic base class for defining tools
that can be used with any LLM provider.
"""

from kader.tools.exec_commands import (
    CommandExecutorTool,
)

from .base import (
    # Core classes
    BaseTool,
    FunctionTool,
    ParameterSchema,
    # Type aliases
    ParameterType,
    ToolCall,
    # Enums
    ToolCategory,
    ToolRegistry,
    ToolResult,
    ToolResultStatus,
    # Schemas and data classes
    ToolSchema,
    # Decorator
    tool,
)
from .filesys import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadDirectoryTool,
    ReadFileTool,
    SearchInDirectoryTool,
    WriteFileTool,
    get_filesystem_tools,
)
from .filesystem import (
    FilesystemBackend,
)
from .rag import (
    DEFAULT_EMBEDDING_MODEL,
    DocumentChunk,
    RAGIndex,
    RAGSearchTool,
    SearchResult,
)
from .todo import TodoTool
from .web import (
    WebFetchTool,
    WebSearchTool,
)


def get_default_registry() -> ToolRegistry:
    """
    Get a registry populated with all standard tools.

    Includes:
    - Filesystem tools
    - Command executor
    - Web tools (search, fetch)
    """
    registry = ToolRegistry()

    # 1. Filesystem Tools
    for t in get_filesystem_tools():
        registry.register(t)

    # 2. Command Execution
    # 2. Command Execution
    registry.register(CommandExecutorTool())

    # 3. Web Tools
    # Note: These might fail if ollama is missing, so we wrap safely
    try:
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
    except ImportError:
        pass

    return registry


__all__ = [
    # Core classes
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    # Schemas and data classes
    "ToolSchema",
    "ParameterSchema",
    "ToolCall",
    "ToolResult",
    # Enums
    "ToolCategory",
    # Decorator
    "tool",
    # Type aliases
    "ParameterType",
    "ToolResultStatus",
    # RAG
    "RAGIndex",
    "RAGSearchTool",
    "DocumentChunk",
    "SearchResult",
    "DEFAULT_EMBEDDING_MODEL",
    # File System Tools
    "ReadFileTool",
    "ReadDirectoryTool",
    "WriteFileTool",
    "EditFileTool",
    "GrepTool",
    "GlobTool",
    "SearchInDirectoryTool",
    "get_filesystem_tools",
    "FilesystemBackend",
    # Web Tools
    "WebSearchTool",
    "WebFetchTool",
    # Command Execution Tool
    # Command Execution Tool
    "CommandExecutorTool",
    # Todo Tool
    "TodoTool",
    # Helpers
    "get_default_registry",
]
