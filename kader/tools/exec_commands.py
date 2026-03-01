"""
Command Execution Tool for Agentic Operations.

This module provides a tool for executing command line operations safely.
It includes OS detection and validation to ensure commands are appropriate
for the host operating system.
"""

import platform
import subprocess

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)


class CommandExecutorTool(BaseTool[str]):
    """
    Tool to execute command line operations.

    Executes commands on the host system with OS-appropriate validation
    and safety checks.
    """

    def __init__(self) -> None:
        """
        Initialize the command executor tool.
        """
        super().__init__(
            name="execute_command",
            description=(
                "Execute a command line operation on the host system. "
                "Automatically validates commands against the host operating system."
            ),
            parameters=[
                ParameterSchema(
                    name="command",
                    type="string",
                    description="The command to execute on the host system",
                ),
                ParameterSchema(
                    name="timeout",
                    type="integer",
                    description="Timeout for the command in seconds (default 30)",
                    required=False,
                    default=30,
                    minimum=1,
                    maximum=300,
                ),
            ],
            category=ToolCategory.UTILITY,
        )

        # Determine the current OS
        self._host_os = platform.system().lower()
        self._shell = self._get_default_shell()

    def _get_default_shell(self) -> str:
        """Get the appropriate shell for the current OS."""
        if self._host_os == "windows":
            return "cmd.exe"
        else:
            return "/bin/bash"

    def _is_command_valid_for_os(self, command: str) -> tuple[bool, str]:
        """
        Check if a command is valid for the current operating system.

        Args:
            command: The command to validate

        Returns:
            Tuple of (valid, reason) where reason explains why invalid
        """
        # Detect command type based on OS
        command_parts = command.strip().split()
        if not command_parts:
            return False, "Empty command"

        first_part = command_parts[0].lower()

        # Cross-platform command validation
        if self._host_os == "windows":
            # Common Unix/Linux commands that typically won't work on Windows
            unix_commands = [
                "ls",
                "grep",
                "awk",
                "sed",
                "find",
                "chmod",
                "chown",
                "cp",
                "mv",
                "rm",
                "mkdir",
                "rmdir",
                "touch",
                "cat",
                "head",
                "tail",
                "wc",
                "sort",
                "uniq",
                "ps",
                "kill",
                "top",
                "df",
                "du",
                "which",
                "whoami",
                "uname",
                "pwd",
                "man",
                "tar",
                "zip",
                "unzip",
                "curl",
                "wget",
            ]

            if first_part in unix_commands:
                return False, (
                    f"The command '{first_part}' is a Unix/Linux command "
                    "and may not be available on Windows. Consider using "
                    f"PowerShell or Windows equivalent command."
                )
        else:  # Unix-like systems (Linux, macOS)
            # Common Windows commands that won't work on Unix
            windows_commands = [
                "dir",
                "copy",
                "move",
                "del",
                "ren",
                "md",
                "rd",
                "cls",
                "ver",
                "vol",
                "label",
                "attrib",
                "xcopy",
                "robocopy",
                "ipconfig",
                "netstat",
                "tasklist",
                "taskkill",
            ]

            if first_part in windows_commands:
                return False, (
                    f"The command '{first_part}' is a Windows command "
                    "and may not be available on this system. Consider using "
                    f"Unix equivalent command."
                )

            # Special handling for PowerShell commands
            if first_part in [
                "get-command",
                "get-help",
                "get-process",
                "stop-process",
                "get-service",
                "start-service",
            ]:
                return False, (
                    f"The command '{first_part}' is a PowerShell command "
                    "and may not be available on this Unix-like system."
                )

        return True, ""

    def _stream_command_output(self, command: str, timeout: int) -> tuple[int, str]:
        """
        Streams the output of a shell command and collects it.

        Args:
            command: The command to execute
            timeout: Timeout in seconds

        Returns:
            Tuple of (return_code, captured_output)
        """
        import time
        from queue import Empty, Queue
        from threading import Thread

        # Prepare kwargs for Popen based on OS
        kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "shell": True,
        }

        if self._host_os != "windows":
            kwargs["executable"] = self._shell

        process = subprocess.Popen(command, **kwargs)
        captured_output = []

        # Use a queue to capture output asynchronously for timeout handling
        q = Queue()

        def enqueue_output(out, queue):
            for line in out:
                queue.put(line)
            try:
                out.close()
            except AttributeError:
                pass  # mock lists in tests might not have close()

        t = Thread(target=enqueue_output, args=(process.stdout, q))
        t.daemon = True
        t.start()

        start_time = time.time()

        while True:
            # Check for timeout
            if timeout > 0 and time.time() - start_time > timeout:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise subprocess.TimeoutExpired(
                    command, timeout, "".join(captured_output).encode()
                )

            try:
                # Read line with a short timeout to allow checking overall execution timeout
                line = q.get(timeout=0.1)

                # Collect output
                captured_output.append(line)
            except Empty:
                # Queue is empty, check if process indicates it's done
                # Note: t.is_alive() is checked because poll() might return before stdout is fully read
                if process.poll() is not None and not t.is_alive():
                    break

        # Ensure thread finished
        t.join(timeout=1)

        # Ensure process finished
        process.wait()

        return process.returncode, "".join(captured_output).strip()

    def execute(self, command: str, timeout: int = 30) -> str:
        """
        Execute a command on the host system and stream its output.

        Args:
            command: The command to execute
            timeout: Timeout in seconds

        Returns:
            Command execution output as string
        """
        try:
            # Validate the command against the current OS
            is_valid, reason = self._is_command_valid_for_os(command)
            if not is_valid:
                return f"Validation Error: {reason}"

            # Execute and stream the command output
            returncode, output = self._stream_command_output(command, timeout)

            # Format result as a string
            if returncode == 0:
                if output:
                    return f"Command executed successfully:\n{output}"
                else:
                    return "Command executed successfully (no output)"
            else:
                output_parts = [f"Command failed with exit code {returncode}"]
                if output:
                    output_parts.append(output)

                return (
                    ":\n".join(output_parts)
                    if len(output_parts) > 1
                    else output_parts[0]
                )

        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Execution Error: {str(e)}"

    async def aexecute(self, command: str, timeout: int = 30) -> str:
        """Async version of execute."""
        import asyncio

        # Since subprocess.run is blocking, we run it in a thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.execute, command, timeout)
        return result

    def get_interruption_message(self, command: str, **kwargs) -> str:
        """Get interruption message for user confirmation."""
        return f"execute {self.name}: {command}"
