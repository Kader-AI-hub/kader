"""
Unit tests for the command execution tools functionality.
"""

import subprocess
from unittest.mock import Mock, patch

import pytest

from kader.tools.exec_commands import CommandExecutorTool


class TestCommandExecutorTool:
    """Test cases for CommandExecutorTool."""

    def test_initialization(self):
        """Test CommandExecutorTool initialization."""
        tool = CommandExecutorTool()

        assert tool.name == "execute_command"
        assert "Execute a command line operation" in tool.description
        assert tool.schema.category == "utility"

        # Check parameters
        params = {param.name: param for param in tool.schema.parameters}
        assert "command" in params
        assert "timeout" in params
        assert params["command"].type == "string"
        assert params["timeout"].type == "integer"
        assert params["timeout"].default == 30
        assert params["timeout"].minimum == 1
        assert params["timeout"].maximum == 300

    def test_get_default_shell_windows(self):
        """Test getting default shell on Windows."""
        with patch("platform.system", return_value="Windows"):
            tool = CommandExecutorTool()
            assert tool._shell == "cmd.exe"

    def test_get_default_shell_unix(self):
        """Test getting default shell on Unix-like systems."""
        with patch("platform.system", return_value="Linux"):
            tool = CommandExecutorTool()
            assert tool._shell == "/bin/bash"

        with patch("platform.system", return_value="Darwin"):
            tool = CommandExecutorTool()
            assert tool._shell == "/bin/bash"

    def test_is_command_valid_for_os_windows_with_unix_command(self):
        """Test command validation on Windows with Unix command."""
        with patch("platform.system", return_value="Windows"):
            tool = CommandExecutorTool()

            is_valid, reason = tool._is_command_valid_for_os("ls -la")

            assert is_valid is False
            assert "Unix/Linux command" in reason
            assert "PowerShell" in reason

    def test_is_command_valid_for_os_unix_with_windows_command(self):
        """Test command validation on Unix with Windows command."""
        with patch("platform.system", return_value="Linux"):
            tool = CommandExecutorTool()

            is_valid, reason = tool._is_command_valid_for_os("dir")

            assert is_valid is False
            assert "Windows command" in reason
            assert "Unix equivalent" in reason

    def test_is_command_valid_for_os_unix_with_powershell_command(self):
        """Test command validation on Unix with PowerShell command."""
        with patch("platform.system", return_value="Linux"):
            tool = CommandExecutorTool()

            is_valid, reason = tool._is_command_valid_for_os("Get-Process")

            assert is_valid is False
            assert "PowerShell command" in reason

    def test_is_command_valid_for_os_valid_commands(self):
        """Test command validation with valid commands."""
        with patch("platform.system", return_value="Linux"):
            tool = CommandExecutorTool()

            # Test valid Unix command
            is_valid, reason = tool._is_command_valid_for_os("echo hello")
            assert is_valid is True
            assert reason == ""

        with patch("platform.system", return_value="Windows"):
            tool = CommandExecutorTool()

            # Test valid Windows command
            is_valid, reason = tool._is_command_valid_for_os("echo hello")
            assert is_valid is True
            assert reason == ""

    def test_is_command_valid_for_os_empty_command(self):
        """Test command validation with empty command."""
        with patch("platform.system", return_value="Linux"):
            tool = CommandExecutorTool()

            is_valid, reason = tool._is_command_valid_for_os("")

            assert is_valid is False
            assert "Empty command" in reason

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_success(self, mock_subprocess_run, mock_platform_system):
        """Test successful command execution."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Linux"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Command output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = tool.execute("echo hello", timeout=10)

        # Verify subprocess was called correctly
        mock_subprocess_run.assert_called_once_with(
            "echo hello",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
            executable="/bin/bash",
        )

        # Verify the result
        assert "Command executed successfully" in result
        assert "Command output" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_success_windows(self, mock_subprocess_run, mock_platform_system):
        """Test successful command execution on Windows."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Windows"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Windows output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = tool.execute("echo hello")

        # Verify subprocess was called correctly for Windows
        mock_subprocess_run.assert_called_once_with(
            "echo hello",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # Default timeout
        )

        # Verify the result
        assert "Command executed successfully" in result
        assert "Windows output" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_command_failed(self, mock_subprocess_run, mock_platform_system):
        """Test command execution that fails."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Linux"
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Command failed"
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = tool.execute("invalid_command")

        # Verify the result
        assert "Command failed with exit code 1" in result
        assert "Command failed" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_command_failed_with_output(
        self, mock_subprocess_run, mock_platform_system
    ):
        """Test command execution that fails but has stdout."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Linux"
        mock_result = Mock()
        mock_result.returncode = 2
        mock_result.stdout = "Partial output"
        mock_result.stderr = "Error message"
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = tool.execute("problematic_command")

        # Verify the result includes both stdout and stderr
        assert "Command failed with exit code 2" in result
        assert "Partial output" in result
        assert "Error message" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_command_no_output(self, mock_subprocess_run, mock_platform_system):
        """Test command execution with no output."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Linux"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = tool.execute("no_output_command")

        # Verify the result
        assert "Command executed successfully (no output)" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_command_timeout(self, mock_subprocess_run, mock_platform_system):
        """Test command execution that times out."""
        # Mock platform and subprocess to raise TimeoutExpired
        mock_platform_system.return_value = "Linux"
        mock_subprocess_run.side_effect = subprocess.TimeoutExpired("command", 5)

        tool = CommandExecutorTool()
        result = tool.execute("slow_command", timeout=5)

        # Verify the result
        assert "Command timed out after 5 seconds" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_command_exception(self, mock_subprocess_run, mock_platform_system):
        """Test command execution that raises an exception."""
        # Mock platform and subprocess to raise a general exception
        mock_platform_system.return_value = "Linux"
        mock_subprocess_run.side_effect = Exception("Unexpected error")

        tool = CommandExecutorTool()
        result = tool.execute("error_command")

        # Verify the result
        assert "Execution Error: Unexpected error" in result

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_invalid_command_windows(
        self, mock_subprocess_run, mock_platform_system
    ):
        """Test executing an invalid command on Windows."""
        mock_platform_system.return_value = "Windows"

        tool = CommandExecutorTool()
        result = tool.execute("ls -la")  # Unix command on Windows

        # Should fail validation before execution
        assert "Validation Error" in result
        assert "Unix/Linux command" in result
        assert "PowerShell" in result

        # subprocess.run should not be called
        mock_subprocess_run.assert_not_called()

    @patch("platform.system")
    @patch("subprocess.run")
    def test_execute_invalid_command_unix(
        self, mock_subprocess_run, mock_platform_system
    ):
        """Test executing an invalid command on Unix."""
        mock_platform_system.return_value = "Linux"

        tool = CommandExecutorTool()
        result = tool.execute("dir")  # Windows command on Unix

        # Should fail validation before execution
        assert "Validation Error" in result
        assert "Windows command" in result
        assert "Unix equivalent" in result

        # subprocess.run should not be called
        mock_subprocess_run.assert_not_called()

    @patch("platform.system")
    @patch("subprocess.run")
    @pytest.mark.asyncio
    async def test_aexecute(self, mock_subprocess_run, mock_platform_system):
        """Test asynchronous command execution."""
        # Mock platform and subprocess
        mock_platform_system.return_value = "Linux"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Async output"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        tool = CommandExecutorTool()
        result = await tool.aexecute("echo async", timeout=15)

        # Verify the result
        assert "Command executed successfully" in result
        assert "Async output" in result
