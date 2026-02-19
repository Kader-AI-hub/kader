"""
Kader Tools - Agentic tool definitions.

This module provides a provider-agnostic base class for defining tools
that can be used with any LLM provider.
"""

from pathlib import Path

from kader.tools.exec_commands import (
    CommandExecutorTool,
)

from .agent import AgentTool
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
from .skills import Skill, SkillLoader, SkillsTool
from .todo import TodoTool
from .web import (
    WebFetchTool,
    WebSearchTool,
)


def get_default_registry(
    skills_dirs: list[Path] | None = None,
) -> ToolRegistry:
    """
    Get a registry populated with all standard tools.

    Includes:
    - Filesystem tools
    - Command executor
    - Web tools (search, fetch)
    - Skills tool (if skills are available)

    Args:
        skills_dirs: Optional list of directories to load skills from.
                    If None, defaults to ~/.kader/skills and ./.kader/
    """
    registry = ToolRegistry()

    # 1. Filesystem Tools
    for t in get_filesystem_tools():
        registry.register(t)

    # 2. Command Execution
    registry.register(CommandExecutorTool())

    # 3. Web Tools
    # Note: These might fail if ollama is missing, so we wrap safely
    try:
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())
    except ImportError:
        pass

    # 4. Skills Tool (only register if skills are available)
    skill_loader = SkillLoader(skills_dirs)
    skills_description = skill_loader.get_description()
    if skills_description and skills_description != "No skills available.":
        registry.register(SkillsTool(skills_dirs))

    return registry


# Module-level cached registry singleton
_cached_default_registry: ToolRegistry | None = None


def get_cached_default_registry() -> ToolRegistry:
    """
    Get a cached registry populated with all standard tools.

    This is more efficient than get_default_registry() when called multiple times,
    as it avoids repeated tool instantiation and registration.

    The cached registry is created once and reused for all subsequent calls.

    Returns:
        Cached ToolRegistry with all standard tools registered.
    """
    global _cached_default_registry
    if _cached_default_registry is None:
        _cached_default_registry = get_default_registry()
    return _cached_default_registry


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
    # Skills Tool
    "SkillsTool",
    "SkillLoader",
    "Skill",
    # Agent Tool
    "AgentTool",
    # Helpers
    "get_default_registry",
    "get_cached_default_registry",
]
