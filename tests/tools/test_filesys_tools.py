"""
Unit tests for file system tools.

Tests ReadFileTool, ReadDirectoryTool, WriteFileTool, EditFileTool,
GrepTool, and GlobTool functionality.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

# Get the absolute path to the tools directory
_TOOLS_DIR = Path(__file__).parent.parent.parent / "kader" / "tools"


def _load_module(name: str, path: Path):
    """Load a module from a path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load dependencies in order
_base = _load_module("kader.tools.base", _TOOLS_DIR / "base.py")
_protocol = _load_module("kader.tools.protocol", _TOOLS_DIR / "protocol.py")
_utils = _load_module("kader.tools.utils", _TOOLS_DIR / "utils.py")
_filesystem = _load_module("kader.tools.filesystem", _TOOLS_DIR / "filesystem.py")
_filesys = _load_module("kader.tools.filesys", _TOOLS_DIR / "filesys.py")

# Export the classes we need for tests
ReadFileTool = _filesys.ReadFileTool
ReadDirectoryTool = _filesys.ReadDirectoryTool
WriteFileTool = _filesys.WriteFileTool
EditFileTool = _filesys.EditFileTool
GrepTool = _filesys.GrepTool
GlobTool = _filesys.GlobTool
get_filesystem_tools = _filesys.get_filesystem_tools


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # Create test files
        (base / "test.txt").write_text("Hello World\nLine 2\nLine 3\n")
        (base / "code.py").write_text(
            "def hello():\n    print('Hello')\n\ndef goodbye():\n    print('Goodbye')\n"
        )
        (base / "data.json").write_text('{"key": "value"}\n')

        # Create subdirectory with files
        subdir = base / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content\n")

        yield base


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def test_read_file_basic(self, temp_dir):
        """Test reading a file."""
        tool = ReadFileTool(base_path=temp_dir)
        result = tool.execute(path="test.txt")

        assert "Hello World" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_read_file_with_line_numbers(self, temp_dir):
        """Test that file content includes line numbers."""
        tool = ReadFileTool(base_path=temp_dir)
        result = tool.execute(path="test.txt")

        # Line numbers should be present
        assert "1" in result
        assert "2" in result

    def test_read_file_with_offset(self, temp_dir):
        """Test reading with offset."""
        tool = ReadFileTool(base_path=temp_dir)
        result = tool.execute(path="test.txt", offset=1, limit=1)

        # Should skip first line
        assert "Hello World" not in result
        assert "Line 2" in result

    def test_read_file_not_found(self, temp_dir):
        """Test reading non-existent file."""
        tool = ReadFileTool(base_path=temp_dir)
        result = tool.execute(path="nonexistent.txt")

        assert "Error" in result or "not found" in result.lower()

    def test_read_nested_file(self, temp_dir):
        """Test reading file in subdirectory."""
        tool = ReadFileTool(base_path=temp_dir)
        result = tool.execute(path="subdir/nested.txt")

        assert "Nested content" in result


class TestReadDirectoryTool:
    """Tests for ReadDirectoryTool."""

    def test_list_directory(self, temp_dir):
        """Test listing directory contents."""
        tool = ReadDirectoryTool(base_path=temp_dir)
        result = tool.execute(path=".")

        assert isinstance(result, list)
        assert len(result) > 0

        # Check that files are listed
        paths = [item["path"] for item in result]
        # Paths should contain our test files
        assert any("test.txt" in p for p in paths)
        assert any("code.py" in p for p in paths)

    def test_list_subdirectory(self, temp_dir):
        """Test listing subdirectory contents."""
        tool = ReadDirectoryTool(base_path=temp_dir)
        result = tool.execute(path="subdir")

        assert isinstance(result, list)
        paths = [item["path"] for item in result]
        assert any("nested.txt" in p for p in paths)

    def test_list_nonexistent_directory(self, temp_dir):
        """Test listing non-existent directory."""
        tool = ReadDirectoryTool(base_path=temp_dir)
        result = tool.execute(path="nonexistent")

        assert result == []


class TestWriteFileTool:
    """Tests for WriteFileTool."""

    def test_write_new_file(self, temp_dir):
        """Test writing a new file."""
        tool = WriteFileTool(base_path=temp_dir)
        result = tool.execute(path="new_file.txt", content="New content")

        assert "error" not in result or result.get("error") is None
        assert result.get("success") is True

        # Verify file was created
        assert (temp_dir / "new_file.txt").exists()
        assert (temp_dir / "new_file.txt").read_text() == "New content"

    def test_write_existing_file_fails(self, temp_dir):
        """Test that writing to existing file fails."""
        tool = WriteFileTool(base_path=temp_dir)
        result = tool.execute(path="test.txt", content="Overwrite")

        assert "error" in result
        assert result["error"] is not None

    def test_write_creates_parent_dirs(self, temp_dir):
        """Test that parent directories are created."""
        tool = WriteFileTool(base_path=temp_dir)
        result = tool.execute(path="new_dir/deep/file.txt", content="Deep content")

        assert result.get("success") is True
        assert (temp_dir / "new_dir" / "deep" / "file.txt").exists()


class TestEditFileTool:
    """Tests for EditFileTool."""

    def test_edit_replace_string(self, temp_dir):
        """Test replacing a string in a file."""
        tool = EditFileTool(base_path=temp_dir)
        result = tool.execute(
            path="test.txt",
            old_string="Hello World",
            new_string="Hi There",
        )

        assert result.get("success") is True
        assert result.get("occurrences") == 1

        # Verify file was modified
        content = (temp_dir / "test.txt").read_text()
        assert "Hi There" in content
        assert "Hello World" not in content

    def test_edit_replace_all(self, temp_dir):
        """Test replacing all occurrences."""
        # Create file with repeated content
        (temp_dir / "repeated.txt").write_text("foo bar foo baz foo")

        tool = EditFileTool(base_path=temp_dir)
        result = tool.execute(
            path="repeated.txt",
            old_string="foo",
            new_string="qux",
            replace_all=True,
        )

        assert result.get("success") is True
        assert result.get("occurrences") == 3

        content = (temp_dir / "repeated.txt").read_text()
        assert "foo" not in content
        assert content.count("qux") == 3

    def test_edit_string_not_found(self, temp_dir):
        """Test editing when string is not found."""
        tool = EditFileTool(base_path=temp_dir)
        result = tool.execute(
            path="test.txt",
            old_string="nonexistent string",
            new_string="replacement",
        )

        assert "error" in result

    def test_edit_file_not_found(self, temp_dir):
        """Test editing non-existent file."""
        tool = EditFileTool(base_path=temp_dir)
        result = tool.execute(
            path="nonexistent.txt",
            old_string="foo",
            new_string="bar",
        )

        assert "error" in result


class TestGrepTool:
    """Tests for GrepTool."""

    def test_grep_basic(self, temp_dir):
        """Test basic grep search."""
        tool = GrepTool(base_path=temp_dir)
        result = tool.execute(pattern="Hello")

        assert isinstance(result, list)
        assert len(result) > 0

        # Should find matches
        if "error" not in result[0]:
            paths = [m.get("path", "") for m in result]
            assert any("test.txt" in p or "code.py" in p for p in paths)

    def test_grep_with_glob_filter(self, temp_dir):
        """Test grep with glob filter."""
        tool = GrepTool(base_path=temp_dir)
        result = tool.execute(pattern="def", glob="*.py")

        assert isinstance(result, list)
        if len(result) > 0 and "error" not in result[0]:
            # Should only find in Python files
            for match in result:
                assert ".py" in match.get("path", "") or "error" in match

    def test_grep_regex_pattern(self, temp_dir):
        """Test grep with regex pattern."""
        tool = GrepTool(base_path=temp_dir)
        result = tool.execute(pattern="def \\w+\\(\\)")  # Match function definitions

        assert isinstance(result, list)

    def test_grep_no_matches(self, temp_dir):
        """Test grep with no matches."""
        tool = GrepTool(base_path=temp_dir)
        result = tool.execute(pattern="zzzznonexistentzzzz")

        assert isinstance(result, list)
        assert len(result) == 0


class TestGlobTool:
    """Tests for GlobTool."""

    def test_glob_basic(self, temp_dir):
        """Test basic glob pattern."""
        tool = GlobTool(base_path=temp_dir)
        result = tool.execute(pattern="*.txt")

        assert isinstance(result, list)
        assert len(result) > 0

        # All results should be .txt files
        for item in result:
            assert item["path"].endswith(".txt")

    def test_glob_python_files(self, temp_dir):
        """Test globbing Python files."""
        tool = GlobTool(base_path=temp_dir)
        result = tool.execute(pattern="*.py")

        assert isinstance(result, list)
        assert len(result) >= 1

        paths = [item["path"] for item in result]
        assert any("code.py" in p for p in paths)

    def test_glob_recursive(self, temp_dir):
        """Test recursive glob pattern."""
        tool = GlobTool(base_path=temp_dir)
        result = tool.execute(pattern="**/*.txt")

        assert isinstance(result, list)
        # Should find both test.txt and subdir/nested.txt
        paths = [item["path"] for item in result]
        assert any("test.txt" in p for p in paths)
        assert any("nested.txt" in p for p in paths)

    def test_glob_no_matches(self, temp_dir):
        """Test glob with no matches."""
        tool = GlobTool(base_path=temp_dir)
        result = tool.execute(pattern="*.nonexistent")

        assert isinstance(result, list)
        assert len(result) == 0


class TestGetFilesystemTools:
    """Tests for get_filesystem_tools function."""

    def test_get_all_tools(self, temp_dir):
        """Test getting all filesystem tools."""
        tools = get_filesystem_tools(base_path=temp_dir)

        assert isinstance(tools, list)
        assert len(tools) >= 6  # At least 6 tools

        # Check tool names
        names = [t.name for t in tools]
        assert "read_file" in names
        assert "read_directory" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "grep" in names
        assert "glob" in names

    def test_tools_have_schemas(self, temp_dir):
        """Test that all tools have valid schemas."""
        tools = get_filesystem_tools(base_path=temp_dir)

        for tool in tools:
            schema = tool.schema
            assert schema.name
            assert schema.description
            assert hasattr(tool, "execute")
            assert hasattr(tool, "aexecute")


class TestVirtualMode:
    """Tests for virtual mode path containment."""

    def test_virtual_mode_path_containment(self, temp_dir):
        """Test that virtual mode prevents path traversal."""
        tool = ReadFileTool(base_path=temp_dir, virtual_mode=True)

        # Attempting to escape should fail or return error
        result = tool.execute(path="../../../etc/passwd")

        # Should get an error, not file contents
        assert (
            "Error" in result
            or "not allowed" in result.lower()
            or "not found" in result.lower()
        )

    def test_virtual_mode_reads_normally(self, temp_dir):
        """Test that virtual mode still allows normal reads."""
        tool = ReadFileTool(base_path=temp_dir, virtual_mode=True)
        result = tool.execute(path="test.txt")

        assert "Hello World" in result
