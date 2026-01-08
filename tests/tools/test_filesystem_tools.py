"""
Unit tests for file system tools in kader.tools.filesys module.
"""

import tempfile
from pathlib import Path

from kader.tools.filesys import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ReadDirectoryTool,
    ReadFileTool,
    SearchInDirectoryTool,
    WriteFileTool,
    get_filesystem_tools,
)
from kader.tools.filesystem import FilesystemBackend


class TestReadFileTool:
    """Test cases for ReadFileTool."""

    def test_initialization(self):
        """Test ReadFileTool initialization."""
        tool = ReadFileTool()
        assert tool.name == "read_file"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_with_valid_file(self):
        """Test reading a valid file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_content = "Line 1\nLine 2\nLine 3"
            test_file.write_text(test_content)

            tool = ReadFileTool(base_path=Path(temp_dir))
            result = tool.execute(str(test_file.relative_to(temp_dir)))

            assert "Line 1" in result
            assert "Line 2" in result
            assert "Line 3" in result

    def test_execute_with_nonexistent_file(self):
        """Test reading a nonexistent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ReadFileTool(base_path=Path(temp_dir))
            result = tool.execute("nonexistent.txt")

            assert "Error" in result or "not found" in result.lower()

    def test_execute_with_offset_and_limit(self):
        """Test reading with offset and limit parameters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / "test.txt"
            test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
            test_file.write_text(test_content)

            tool = ReadFileTool(base_path=Path(temp_dir))
            result = tool.execute(
                str(test_file.relative_to(temp_dir)), offset=2, limit=2
            )

            # Should contain lines 3 and 4 (0-indexed)
            assert "Line 3" in result
            assert "Line 4" in result
            assert "Line 1" not in result
            assert "Line 2" not in result
            assert "Line 5" not in result

    def test_async_execute(self):
        """Test async execution of ReadFileTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                test_file = Path(temp_dir) / "test.txt"
                test_content = "Test content"
                test_file.write_text(test_content)

                tool = ReadFileTool(base_path=Path(temp_dir))
                result = await tool.aexecute(str(test_file.relative_to(temp_dir)))

                assert "Test content" in result

        asyncio.run(run_test())


class TestReadDirectoryTool:
    """Test cases for ReadDirectoryTool."""

    def test_initialization(self):
        """Test ReadDirectoryTool initialization."""
        tool = ReadDirectoryTool()
        assert tool.name == "read_directory"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_with_valid_directory(self):
        """Test listing a valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files and directories
            (Path(temp_dir) / "file1.txt").write_text("content1")
            (Path(temp_dir) / "file2.py").write_text("content2")
            (Path(temp_dir) / "subdir").mkdir()

            tool = ReadDirectoryTool(base_path=Path(temp_dir))
            result = tool.execute(".")

            assert isinstance(result, list)
            assert len(result) >= 3  # file1.txt, file2.py, subdir

            # Check that we have the expected files by path
            paths = [item["path"] for item in result]
            assert any("file1.txt" in p for p in paths)
            assert any("file2.py" in p for p in paths)
            assert any("subdir" in p for p in paths)

    def test_execute_with_nonexistent_directory(self):
        """Test listing a nonexistent directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = ReadDirectoryTool(base_path=Path(temp_dir))
            result = tool.execute("nonexistent_dir")

            # Should return an empty list or error appropriately
            assert isinstance(result, list)

    def test_async_execute(self):
        """Test async execution of ReadDirectoryTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a test file
                (Path(temp_dir) / "test_file.txt").write_text("test")

                tool = ReadDirectoryTool(base_path=Path(temp_dir))
                result = await tool.aexecute(".")

                assert isinstance(result, list)
                paths = [item["path"] for item in result]
                assert any("test_file.txt" in p for p in paths)

        asyncio.run(run_test())


class TestWriteFileTool:
    """Test cases for WriteFileTool."""

    def test_initialization(self):
        """Test WriteFileTool initialization."""
        tool = WriteFileTool()
        assert tool.name == "write_file"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_create_new_file(self):
        """Test creating a new file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            tool = WriteFileTool(base_path=Path(temp_dir))
            result = tool.execute("new_file.txt", "Hello, World!")

            assert result["success"] is True
            # The path returned should match the input path
            assert result["path"] == "new_file.txt"
            assert result["bytes_written"] == len("Hello, World!")

            # Verify the file was actually created
            created_file = Path(temp_dir) / "new_file.txt"
            assert created_file.exists()
            assert created_file.read_text() == "Hello, World!"

    def test_execute_fail_if_file_exists(self):
        """Test that writing fails if file already exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file first
            existing_file = Path(temp_dir) / "existing.txt"
            existing_file.write_text("existing content")

            tool = WriteFileTool(base_path=Path(temp_dir))
            result = tool.execute("existing.txt", "new content")

            # Should return an error
            assert "error" in result

    def test_async_execute(self):
        """Test async execution of WriteFileTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                tool = WriteFileTool(base_path=Path(temp_dir))
                result = await tool.aexecute("async_test.txt", "Async content")

                assert result["success"] is True
                assert result["bytes_written"] == len("Async content")

                # Verify the file was created
                created_file = Path(temp_dir) / "async_test.txt"
                assert created_file.exists()
                assert created_file.read_text() == "Async content"

        asyncio.run(run_test())


class TestEditFileTool:
    """Test cases for EditFileTool."""

    def test_initialization(self):
        """Test EditFileTool initialization."""
        tool = EditFileTool()
        assert tool.name == "edit_file"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_replace_content(self):
        """Test replacing content in a file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with initial content
            test_file = Path(temp_dir) / "edit_test.txt"
            test_file.write_text("Hello world, this is a test.")

            tool = EditFileTool(base_path=Path(temp_dir))
            result = tool.execute("edit_test.txt", "world", "universe")

            assert result["success"] is True
            assert result["occurrences"] == 1

            # Verify the content was changed
            assert test_file.read_text() == "Hello universe, this is a test."

    def test_execute_replace_all_occurrences(self):
        """Test replacing all occurrences of a string."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with multiple occurrences
            test_file = Path(temp_dir) / "edit_test.txt"
            test_file.write_text("Hello world, world is beautiful. Say hello to world.")

            tool = EditFileTool(base_path=Path(temp_dir))
            result = tool.execute(
                "edit_test.txt", "world", "universe", replace_all=True
            )

            assert result["success"] is True
            assert result["occurrences"] == 3

            # Verify all occurrences were changed
            expected_content = (
                "Hello universe, universe is beautiful. Say hello to universe."
            )
            assert test_file.read_text() == expected_content

    def test_execute_replace_no_matches(self):
        """Test replacing when no matches are found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with content
            test_file = Path(temp_dir) / "edit_test.txt"
            test_file.write_text("Hello world, this is a test.")

            tool = EditFileTool(base_path=Path(temp_dir))
            result = tool.execute("edit_test.txt", "nonexistent", "replacement")

            # When no matches are found, the edit should return an error
            assert "error" in result
            assert "not found" in result["error"]

    def test_async_execute(self):
        """Test async execution of EditFileTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a file with initial content
                test_file = Path(temp_dir) / "async_edit_test.txt"
                test_file.write_text("Hello world")

                tool = EditFileTool(base_path=Path(temp_dir))
                result = await tool.aexecute("async_edit_test.txt", "world", "universe")

                assert result["success"] is True
                assert result["occurrences"] == 1

                # Verify the content was changed
                assert test_file.read_text() == "Hello universe"

        asyncio.run(run_test())


class TestGrepTool:
    """Test cases for GrepTool."""

    def test_initialization(self):
        """Test GrepTool initialization."""
        tool = GrepTool()
        assert tool.name == "grep"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_search_pattern(self):
        """Test searching for a pattern in files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file with content
            test_file = Path(temp_dir) / "grep_test.txt"
            test_content = "Hello world\nThis is a test\nHello again"
            test_file.write_text(test_content)

            tool = GrepTool(base_path=Path(temp_dir))
            result = tool.execute("Hello", ".")

            # Should find matches
            assert isinstance(result, list)
            assert len(result) > 0

            # Check that matches contain the expected content
            for match in result:
                assert "Hello" in match["text"]

    def test_execute_search_with_glob(self):
        """Test searching with a glob pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            (Path(temp_dir) / "test1.py").write_text(
                "def hello(): pass\nprint('hello')"
            )
            (Path(temp_dir) / "test2.txt").write_text("hello world")
            (Path(temp_dir) / "test3.py").write_text(
                "def greet(): pass\nprint('greetings')"
            )

            tool = GrepTool(base_path=Path(temp_dir))
            result = tool.execute("hello", ".", "*.py")

            # Should only search in .py files
            assert isinstance(result, list)
            # All matches should be from .py files
            for match in result:
                assert match["path"].endswith(".py")

    def test_async_execute(self):
        """Test async execution of GrepTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a test file
                test_file = Path(temp_dir) / "async_grep_test.txt"
                test_file.write_text("Hello world\nGoodbye world")

                tool = GrepTool(base_path=Path(temp_dir))
                result = await tool.aexecute("Hello", ".")

                assert isinstance(result, list)
                assert len(result) > 0
                assert any("Hello" in match["text"] for match in result)

        asyncio.run(run_test())


class TestGlobTool:
    """Test cases for GlobTool."""

    def test_initialization(self):
        """Test GlobTool initialization."""
        tool = GlobTool()
        assert tool.name == "glob"
        assert tool._backend is not None
        assert isinstance(tool._backend, FilesystemBackend)

    def test_execute_find_files(self):
        """Test finding files with glob pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "test1.py").write_text("# Python file 1")
            (Path(temp_dir) / "test2.py").write_text("# Python file 2")
            (Path(temp_dir) / "test.txt").write_text("Text file")
            subdir = Path(temp_dir) / "subdir"
            subdir.mkdir()
            (subdir / "nested.py").write_text("# Nested Python file")

            tool = GlobTool(base_path=Path(temp_dir))
            result = tool.execute("**/*.py", "/")

            # Should find all Python files
            assert isinstance(result, list)
            paths = [item["path"] for item in result]
            assert any("test1.py" in p for p in paths)
            assert any("test2.py" in p for p in paths)
            assert any("nested.py" in p for p in paths)

    def test_execute_specific_pattern(self):
        """Test finding files with specific pattern."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            (Path(temp_dir) / "data.json").write_text('{"test": true}')
            (Path(temp_dir) / "config.json").write_text('{"config": true}')
            (Path(temp_dir) / "readme.md").write_text("# Readme")

            tool = GlobTool(base_path=Path(temp_dir))
            result = tool.execute("*.json", "/")

            # Should find only JSON files
            assert isinstance(result, list)
            paths = [item["path"] for item in result]
            json_files = [p for p in paths if p.endswith(".json")]
            assert len(json_files) == 2
            assert any("data.json" in p for p in json_files)
            assert any("config.json" in p for p in json_files)

    def test_async_execute(self):
        """Test async execution of GlobTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a test file
                (Path(temp_dir) / "async_test.py").write_text("# Async test")

                tool = GlobTool(base_path=Path(temp_dir))
                result = await tool.aexecute("*.py", "/")

                assert isinstance(result, list)
                paths = [item["path"] for item in result]
                assert any("async_test.py" in p for p in paths)

        asyncio.run(run_test())


class TestSearchInDirectoryTool:
    """Test cases for SearchInDirectoryTool."""

    def test_initialization(self):
        """Test SearchInDirectoryTool initialization."""
        tool = SearchInDirectoryTool()
        assert tool.name == "search_in_directory"
        assert tool._rag_tool is not None

    def test_execute_search(self):
        """Test semantic search functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file with content
            test_file = Path(temp_dir) / "search_test.txt"
            test_content = (
                "This is a test file containing information about Python programming."
            )
            test_file.write_text(test_content)

            tool = SearchInDirectoryTool(base_path=Path(temp_dir))
            result = tool.execute("Python programming", top_k=5)

            # Should return a list of results
            assert isinstance(result, list)
            # Results may be empty if Ollama is not available, but should not error

    def test_async_execute(self):
        """Test async execution of SearchInDirectoryTool."""
        import asyncio

        async def run_test():
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create a test file
                test_file = Path(temp_dir) / "async_search_test.txt"
                test_file.write_text("Content for async search test")

                tool = SearchInDirectoryTool(base_path=Path(temp_dir))
                result = await tool.aexecute("async search", top_k=5)

                # Should return a list of results
                assert isinstance(result, list)

        asyncio.run(run_test())


class TestGetFilesystemTools:
    """Test cases for get_filesystem_tools function."""

    def test_get_all_filesystem_tools(self):
        """Test getting all filesystem tools."""
        tools = get_filesystem_tools()

        # Should return a list of tools
        assert isinstance(tools, list)
        assert (
            len(tools) == 7
        )  # ReadFileTool, ReadDirectoryTool, WriteFileTool, EditFileTool, GrepTool, GlobTool, SearchInDirectoryTool

        # Check that all expected tools are present
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "read_file",
            "read_directory",
            "write_file",
            "edit_file",
            "grep",
            "glob",
            "search_in_directory",
        ]
        for name in expected_names:
            assert name in tool_names

    def test_get_filesystem_tools_with_base_path(self):
        """Test getting filesystem tools with a specific base path."""
        base_path = Path("/tmp")
        tools = get_filesystem_tools(base_path=base_path)

        # All tools should have the same base path configured
        for tool in tools:
            # For tools that have a backend, check that the backend has the correct cwd
            if hasattr(tool, "_backend"):
                assert tool._backend.cwd == base_path.resolve()

    def test_get_filesystem_tools_with_virtual_mode(self):
        """Test getting filesystem tools with virtual mode."""
        tools = get_filesystem_tools(virtual_mode=True)

        # All tools should have virtual mode enabled
        for tool in tools:
            if hasattr(tool, "_backend"):
                assert tool._backend.virtual_mode is True
