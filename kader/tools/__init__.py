"""
Kader Tools - Agentic tool definitions.

This module provides a provider-agnostic base class for defining tools
that can be used with any LLM provider.
"""

from .base import (
    # Core classes
    BaseTool,
    FunctionTool,
    ToolRegistry,
    
    # Schemas and data classes
    ToolSchema,
    ParameterSchema,
    ToolCall,
    ToolResult,
    
    # Enums
    ToolCategory,
    
    # Decorator
    tool,
    
    # Type aliases
    ParameterType,
    ToolResultStatus,
)

from .rag import (
    RAGIndex,
    RAGSearchTool,
    DocumentChunk,
    SearchResult,
    DEFAULT_EMBEDDING_MODEL,
)

from .filesys import (
    ReadFileTool,
    ReadDirectoryTool,
    WriteFileTool,
    EditFileTool,
    GrepTool,
    GlobTool,
    SearchInDirectoryTool,
    get_filesystem_tools,
)

from .filesystem import (
    FilesystemBackend,
)

from .web import (
    WebSearchTool,
    WebFetchTool,
)

from .exec_commands import (
    CommandExecutorTool,
)

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
    "CommandExecutorTool",
]

