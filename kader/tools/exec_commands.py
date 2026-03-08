"""
Command Execution Tool for Agentic Operations.

This module provides a tool for executing command line operations safely.
It includes OS detection, PTY support for interactive commands, and
validation to ensure commands are appropriate for the host operating system.
"""

import os
import platform
import subprocess
from typing import Any, Callable

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)


def _get_pty_module():
    """Get the appropriate PTY module for the current OS."""
    system = platform.system().lower()

    if system == "windows":
        try:
            import pywinpty

            return pywinpty
        except ImportError:
            return None
    else:
        return "pty"


class CommandExecutorTool(BaseTool[str]):
    """
    Tool to execute command line operations.

    Executes commands on the host system with OS-appropriate validation,
    PTY support for interactive commands, and safety checks.
    """

    _output_callback: Callable[[str], None] | None = None
    _input_callback: Callable[[], str | None] | None = None

    @classmethod
    def set_output_callback(cls, callback: Callable[[str], None] | None) -> None:
        """Set the output callback for streaming command output."""
        cls._output_callback = callback

    @classmethod
    def set_input_callback(cls, callback: Callable[[], str | None] | None) -> None:
        """Set the input callback for getting user input during command execution."""
        cls._input_callback = callback

    @classmethod
    def clear_callbacks(cls) -> None:
        """Clear both output and input callbacks."""
        cls._output_callback = None
        cls._input_callback = None

    def __init__(self) -> None:
        """
        Initialize the command executor tool.
        """
        super().__init__(
            name="execute_command",
            description=(
                "Execute a command line operation on the host system. "
                "Automatically validates commands against the host operating system. "
                "Supports interactive commands that require user input."
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
                    description="Timeout for the command in seconds (default 60)",
                    required=False,
                    default=60,
                    minimum=1,
                    maximum=3600,
                ),
                ParameterSchema(
                    name="interactive",
                    type="boolean",
                    description="Run command in interactive mode with PTY (default True)",
                    required=False,
                    default=True,
                ),
            ],
            category=ToolCategory.UTILITY,
        )

        self._host_os = platform.system().lower()
        self._shell = self._get_default_shell()
        self._pty_module = _get_pty_module()

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
        command_parts = command.strip().split()
        if not command_parts:
            return False, "Empty command"

        first_part = command_parts[0].lower()

        if self._host_os == "windows":
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
        else:
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

    def _execute_with_pty(
        self,
        command: str,
        timeout: int,
        output_callback: Callable[[str], None] | None = None,
        input_callback: Callable[[], str | None] | None = None,
    ) -> tuple[int, str]:
        """
        Execute command using PTY for interactive support.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            output_callback: Optional callback for streaming output
            input_callback: Optional callback to get user input when needed

        Returns:
            Tuple of (return_code, captured_output)
        """
        import pty
        import time

        captured_output = []
        system = platform.system().lower()

        if system == "windows":
            return self._execute_with_winpty(
                command,
                timeout,
                output_callback,
                input_callback,
            )

        master_fd, slave_fd = pty.openpty()

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"

        kwargs = {
            "executable": self._shell,
            "env": env,
            "shell": True,
        }

        try:
            process = subprocess.Popen(
                command,
                stdin=slave_fd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **kwargs,
            )
        finally:
            os.close(slave_fd)

        os.close(master_fd)

        start_time = time.time()
        output_buffer = ""

        stdout = process.stdout
        assert stdout is not None

        while True:
            if timeout > 0 and time.time() - start_time > timeout:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise subprocess.TimeoutExpired(
                    command, timeout, "".join(captured_output).encode()
                )

            line = stdout.readline()
            if line:
                captured_output.append(line)
                output_buffer += line
                if output_callback:
                    output_callback(line)
            elif process.poll() is not None:
                break

        process.wait()
        return process.returncode, "".join(captured_output).strip()

    def _execute_with_winpty(
        self,
        command: str,
        timeout: int,
        output_callback: Callable[[str], None] | None = None,
        input_callback: Callable[[], str | None] | None = None,
    ) -> tuple[int, str]:
        """
        Execute command using pywinpty on Windows.

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            output_callback: Optional callback for streaming output
            input_callback: Optional callback to get user input when needed

        Returns:
            Tuple of (return_code, captured_output)
        """
        import time

        import pywinpty

        if self._pty_module is None:
            return self._execute_direct(command, timeout, output_callback)

        captured_output = []
        cols, rows = 80, 24

        try:
            pty = pywinpty.PTY(width=cols, height=rows, visible=False)
        except Exception:
            return self._execute_direct(command, timeout, output_callback)

        try:
            process = pty.spawn(self._shell, ["-c", command])
        except Exception:
            return self._execute_direct(command, timeout, output_callback)

        start_time = time.time()

        while True:
            if timeout > 0 and time.time() - start_time > timeout:
                pty.terminate()
                raise subprocess.TimeoutExpired(
                    command, timeout, "".join(captured_output).encode()
                )

            try:
                data = process.read(blocking=False)
                if data:
                    captured_output.append(data)
                    if output_callback:
                        output_callback(data)
            except Exception:
                pass

            if not process.isalive():
                break

            time.sleep(0.05)

        returncode = process.exitstatus
        return returncode, "".join(captured_output).strip()

    def _execute_direct(
        self,
        command: str,
        timeout: int,
        output_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, str]:
        """
        Execute command directly without PTY (fallback or non-interactive mode).

        Args:
            command: The command to execute
            timeout: Timeout in seconds
            output_callback: Optional callback for streaming output

        Returns:
            Tuple of (return_code, captured_output)
        """
        import time
        from queue import Empty, Queue
        from threading import Thread

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

        q = Queue()

        def enqueue_output(out, queue):
            for line in out:
                queue.put(line)
            try:
                out.close()
            except AttributeError:
                pass

        t = Thread(target=enqueue_output, args=(process.stdout, q))
        t.daemon = True
        t.start()

        start_time = time.time()

        while True:
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
                line = q.get(timeout=0.1)
                captured_output.append(line)
                if output_callback:
                    output_callback(line)
            except Empty:
                if process.poll() is not None and not t.is_alive():
                    break

        t.join(timeout=1)
        process.wait()

        return process.returncode, "".join(captured_output).strip()

    def execute(self, **kwargs: Any) -> str:
        """
        Execute a command on the host system.

        Args:
            **kwargs: Tool arguments:
                command: The command to execute (required)
                timeout: Timeout in seconds (default 60)
                interactive: Run with PTY (default True)
                output_callback: Callback for streaming output
                input_callback: Callback for user input

        Returns:
            Command execution output as string
        """
        command = kwargs.get("command", "")
        timeout = kwargs.get("timeout", 60)
        interactive = kwargs.get("interactive", True)

        output_callback = kwargs.get("output_callback") or type(self)._output_callback
        input_callback = kwargs.get("input_callback") or type(self)._input_callback

        if not command:
            return "Error: command is required"

        is_valid, reason = self._is_command_valid_for_os(command)
        if not is_valid:
            return f"Validation Error: {reason}"

        use_pty = interactive

        if interactive and self._pty_module is None:
            if platform.system().lower() == "windows":
                use_pty = False

        try:
            if use_pty:
                returncode, output = self._execute_with_pty(
                    command,
                    timeout,
                    output_callback,
                    input_callback,
                )
            else:
                returncode, output = self._execute_direct(
                    command,
                    timeout,
                    output_callback,
                )

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

    async def aexecute(self, **kwargs: Any) -> str:
        """
        Async version of execute.

        Args:
            **kwargs: Tool arguments (same as execute)

        Returns:
            Command execution output as string
        """
        import asyncio

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: self.execute(**kwargs))
        return result

    def get_interruption_message(self, **kwargs: Any) -> str:
        """Get interruption message for user confirmation."""
        command = kwargs.get("command", "")
        return f"execute {self.name}: {command}"
