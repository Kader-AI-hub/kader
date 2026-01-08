"""
File System Tools for Agentic Operations.

All tools operate relative to the current working directory (CWD) for security.
Uses FilesystemBackend from filesystem.py for the underlying operations.
"""

import asyncio
from pathlib import Path
from typing import Any

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)
from .filesystem import FilesystemBackend
from .protocol import FileInfo


class ReadFileTool(BaseTool[str]):
    """
    Tool to read the contents of a file.

    Uses FilesystemBackend for secure file access with path containment.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the read file tool.

        Args:
            base_path: Base path for file operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="read_file",
            description=(
                "Read the contents of a file. Returns the file content as text with line numbers. "
                "Supports pagination with offset and limit for large files."
            ),
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the file to read",
                ),
                ParameterSchema(
                    name="offset",
                    type="integer",
                    description="Line offset to start reading from (0-indexed)",
                    required=False,
                    default=0,
                    minimum=0,
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Maximum number of lines to read",
                    required=False,
                    default=2000,
                    minimum=1,
                ),
            ],
            category=ToolCategory.FILE_SYSTEM,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """
        Read file contents.

        Args:
            path: Path to the file
            offset: Line offset to start reading from (0-indexed)
            limit: Maximum number of lines to read

        Returns:
            File contents with line numbers, or error message
        """
        return self._backend.read(path, offset=offset, limit=limit)

    async def aexecute(
        self,
        path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> str:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, path, offset, limit)

    def get_interruption_message(self, path: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute read_file: {path}"


class ReadDirectoryTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool to list the contents of a directory.

    Uses FilesystemBackend for secure directory listing.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the read directory tool.

        Args:
            base_path: Base path for file operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="read_directory",
            description=(
                "List the contents of a directory. Returns a list of files and "
                "subdirectories with their types and sizes."
            ),
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the directory (use '.' or '/' for root)",
                    default=".",
                ),
            ],
            category=ToolCategory.FILE_SYSTEM,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        path: str = ".",
    ) -> list[dict[str, Any]]:
        """
        List directory contents.

        Args:
            path: Path to directory

        Returns:
            List of file/directory information dictionaries
        """
        result: list[FileInfo] = self._backend.ls_info(path)
        # Convert FileInfo TypedDict to regular dict for JSON serialization
        return [dict(item) for item in result]

    async def aexecute(
        self,
        path: str = ".",
    ) -> list[dict[str, Any]]:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, path)

    def get_interruption_message(self, path: str = ".", **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute read_dir: {path}"


class WriteFileTool(BaseTool[dict[str, Any]]):
    """
    Tool to write content to a new file.

    Uses FilesystemBackend for secure file creation.
    Only creates new files; fails if file already exists.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the write file tool.

        Args:
            base_path: Base path for file operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="write_file",
            description=(
                "Create a new file with content. Fails if the file already exists. "
                "Use edit_file tool to modify existing files."
            ),
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the file to create",
                ),
                ParameterSchema(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                ),
            ],
            category=ToolCategory.FILE_SYSTEM,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        """
        Write content to a new file.

        Args:
            path: Path to the file to create
            content: Content to write

        Returns:
            Dictionary with operation result
        """
        result = self._backend.write(path, content)

        if result.error:
            return {"error": result.error}

        return {
            "path": result.path,
            "success": True,
            "bytes_written": len(content.encode("utf-8")),
        }

    async def aexecute(
        self,
        path: str,
        content: str,
    ) -> dict[str, Any]:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, path, content)

    def get_interruption_message(self, path: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute write_file: {path}"


class EditFileTool(BaseTool[dict[str, Any]]):
    """
    Tool to edit an existing file by string replacement.

    Uses FilesystemBackend for secure file editing.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the edit file tool.

        Args:
            base_path: Base path for file operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="edit_file",
            description=(
                "Edit an existing file by replacing text. Replaces exact string matches. "
                "Use replace_all=True to replace all occurrences."
            ),
            parameters=[
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Path to the file to edit",
                ),
                ParameterSchema(
                    name="old_string",
                    type="string",
                    description="Exact string to search for and replace",
                ),
                ParameterSchema(
                    name="new_string",
                    type="string",
                    description="String to replace old_string with",
                ),
                ParameterSchema(
                    name="replace_all",
                    type="boolean",
                    description="If True, replace all occurrences",
                    required=False,
                    default=False,
                ),
            ],
            category=ToolCategory.FILE_SYSTEM,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """
        Edit a file by replacing string occurrences.

        Args:
            path: Path to the file
            old_string: String to replace
            new_string: Replacement string
            replace_all: Whether to replace all occurrences

        Returns:
            Dictionary with operation result
        """
        result = self._backend.edit(path, old_string, new_string, replace_all)

        if result.error:
            return {"error": result.error}

        return {
            "path": result.path,
            "success": True,
            "occurrences": result.occurrences,
        }

    async def aexecute(
        self,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Async version of execute."""
        return await asyncio.to_thread(
            self.execute, path, old_string, new_string, replace_all
        )

    def get_interruption_message(self, path: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute edit_file: {path}"


class GrepTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool to search for patterns in files using regex.

    Uses FilesystemBackend's grep_raw with ripgrep fallback to Python.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the grep tool.

        Args:
            base_path: Base path for search operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="grep",
            description=(
                "Search for a regex pattern in files. Returns matching lines with "
                "file paths and line numbers. Optionally filter by glob pattern."
            ),
            parameters=[
                ParameterSchema(
                    name="pattern",
                    type="string",
                    description="Regex pattern to search for",
                ),
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Directory path to search in (defaults to current directory)",
                    required=False,
                    default=".",
                ),
                ParameterSchema(
                    name="glob",
                    type="string",
                    description="Glob pattern to filter files (e.g., '*.py')",
                    required=False,
                ),
            ],
            category=ToolCategory.SEARCH,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        pattern: str,
        path: str | None = ".",
        glob: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for pattern in files.

        Args:
            pattern: Regex pattern to search for
            path: Directory to search in
            glob: Optional glob pattern to filter files

        Returns:
            List of match dictionaries with path, line, and text
        """
        result = self._backend.grep_raw(pattern, path, glob)

        if isinstance(result, str):
            # Error message
            return [{"error": result}]

        # Convert GrepMatch TypedDict to regular dict
        return [dict(match) for match in result]

    async def aexecute(
        self,
        pattern: str,
        path: str | None = ".",
        glob: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, pattern, path, glob)

    def get_interruption_message(self, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return "execute grep"


class GlobTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool to find files matching a glob pattern.

    Uses FilesystemBackend's glob_info for pattern matching.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        virtual_mode: bool = False,
    ) -> None:
        """
        Initialize the glob tool.

        Args:
            base_path: Base path for search operations (defaults to CWD)
            virtual_mode: If True, use virtual path mode (sandboxed to base_path)
        """
        super().__init__(
            name="glob",
            description=(
                "Find files matching a glob pattern. Supports wildcards: "
                "* (any chars), ** (recursive), ? (single char), [abc] (char set)."
            ),
            parameters=[
                ParameterSchema(
                    name="pattern",
                    type="string",
                    description="Glob pattern to match (e.g., '*.py', '**/*.txt')",
                ),
                ParameterSchema(
                    name="path",
                    type="string",
                    description="Base directory to search from",
                    required=False,
                    default="/",
                ),
            ],
            category=ToolCategory.SEARCH,
        )
        self._backend = FilesystemBackend(
            root_dir=base_path,
            virtual_mode=virtual_mode,
        )

    def execute(
        self,
        pattern: str,
        path: str = "/",
    ) -> list[dict[str, Any]]:
        """
        Find files matching glob pattern.

        Args:
            pattern: Glob pattern to match
            path: Base directory to search from

        Returns:
            List of file info dictionaries
        """
        result: list[FileInfo] = self._backend.glob_info(pattern, path)
        return [dict(item) for item in result]

    async def aexecute(
        self,
        pattern: str,
        path: str = "/",
    ) -> list[dict[str, Any]]:
        """Async version of execute."""
        return await asyncio.to_thread(self.execute, pattern, path)

    def get_interruption_message(self, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return "execute glob"


class SearchInDirectoryTool(BaseTool[list[dict[str, Any]]]):
    """
    Tool to search for content in files using RAG-based semantic search.

    Uses Ollama embeddings and FAISS for intelligent code search.
    """

    def __init__(
        self,
        base_path: Path | None = None,
        embedding_model: str = "all-minilm:22m",
    ) -> None:
        """
        Initialize the search tool.

        Args:
            base_path: Base path for search operations (defaults to CWD)
            embedding_model: Ollama embedding model to use
        """
        super().__init__(
            name="search_in_directory",
            description=(
                "Search for code and content using semantic (meaning-based) search. "
                "Finds relevant files and snippets based on your natural language query."
            ),
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Natural language search query (e.g., 'function to read JSON files')",
                ),
                ParameterSchema(
                    name="top_k",
                    type="integer",
                    description="Number of results to return",
                    required=False,
                    default=5,
                    minimum=1,
                    maximum=20,
                ),
                ParameterSchema(
                    name="rebuild_index",
                    type="boolean",
                    description="Force rebuild the search index (use after file changes)",
                    required=False,
                    default=False,
                ),
            ],
            category=ToolCategory.SEARCH,
        )

        # Lazy import to avoid loading dependencies at module import time
        from .rag import RAGSearchTool

        self._rag_tool = RAGSearchTool(
            base_path=base_path,
            embedding_model=embedding_model,
        )

    def execute(
        self,
        query: str,
        top_k: int = 5,
        rebuild_index: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search for content in the directory.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            rebuild_index: Force rebuild the index

        Returns:
            List of search result dictionaries
        """
        return self._rag_tool.execute(query, top_k, rebuild_index)

    async def aexecute(
        self,
        query: str,
        top_k: int = 5,
        rebuild_index: bool = False,
    ) -> list[dict[str, Any]]:
        """Async version of execute."""
        return await self._rag_tool.aexecute(query, top_k, rebuild_index)

    def get_interruption_message(self, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return "execute search_in_directory"


# Convenience function to get all file system tools
def get_filesystem_tools(
    base_path: Path | None = None,
    virtual_mode: bool = False,
) -> list[BaseTool]:
    """
    Get all file system tools configured with the given base path.

    Args:
        base_path: Base path for all tools (defaults to CWD)
        virtual_mode: If True, use virtual path mode for security

    Returns:
        List of configured file system tools
    """
    bp = base_path or Path.cwd()
    return [
        ReadFileTool(bp, virtual_mode),
        ReadDirectoryTool(bp, virtual_mode),
        WriteFileTool(bp, virtual_mode),
        EditFileTool(bp, virtual_mode),
        GrepTool(bp, virtual_mode),
        GlobTool(bp, virtual_mode),
        SearchInDirectoryTool(bp),
    ]
