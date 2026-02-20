"""
Software Engineering Tools for Agentic Operations.

This module provides tools for common software engineering tasks including:
- Running test suites (pytest, npm test, cargo test, etc.)
- Running code linters (ruff, pylint, eslint, golangci-lint)
- Running type checkers (mypy, tsc, flycheck)
- Git operations (status, diff, commit, log, branch, pull, push)
- Package management (install, uninstall, list, search)
"""

import asyncio
import os
import subprocess
from typing import Optional, List

from .base import (
    BaseTool,
    ParameterSchema,
    ToolCategory,
)


class TestRunnerTool(BaseTool[str]):
    """
    Tool to run test suites for various programming languages and frameworks.

    Supports pytest (Python), npm test (JavaScript/TypeScript), cargo test (Rust),
    and other test runners. The appropriate runner is inferred from the command
    or can be explicitly specified.
    """

    def __init__(self) -> None:
        """
        Initialize the test runner tool.
        """
        super().__init__(
            name="run_tests",
            description=(
                "Run test suites for various programming languages and frameworks. "
                "Supports pytest (Python), npm test (JavaScript/TypeScript), "
                "cargo test (Rust), go test (Go), and others. "
                "Returns the test output including passed/failed counts."
            ),
            parameters=[
                ParameterSchema(
                    name="command",
                    type="string",
                    description=(
                        "The test command to execute (e.g., 'pytest', 'npm test', "
                        "'cargo test', 'go test'). Can include arguments like "
                        "'pytest -v' or 'npm test -- --coverage'"
                    ),
                ),
                ParameterSchema(
                    name="working_directory",
                    type="string",
                    description=(
                        "Directory to run the test command in. "
                        "Defaults to current working directory."
                    ),
                    required=False,
                ),
            ],
            category=ToolCategory.CODE,
        )

    def execute(
        self,
        command: str,
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a test command synchronously.

        Args:
            command: The test command to run (e.g., 'pytest -v', 'npm test')
            working_directory: Optional directory to run the command in

        Returns:
            Test execution results as a string
        """
        if not command or not command.strip():
            return "Error: No test command provided"

        cwd = working_directory or os.getcwd()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for tests
                cwd=cwd,
            )

            output_parts = []

            if result.returncode == 0:
                output_parts.append("✓ Tests passed successfully")
            else:
                output_parts.append(f"✗ Tests failed with exit code {result.returncode}")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"\n--- STDOUT ---\n{stdout}")
            if stderr:
                output_parts.append(f"\n--- STDERR ---\n{stderr}")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return "Error: Test execution timed out after 300 seconds"
        except FileNotFoundError:
            return f"Error: Command not found. Make sure the test runner is installed: {command}"
        except Exception as e:
            return f"Error executing tests: {str(e)}"

    async def aexecute(
        self,
        command: str,
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a test command asynchronously.

        Args:
            command: The test command to run
            working_directory: Optional directory to run the command in

        Returns:
            Test execution results as a string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute, command, working_directory
        )

    def get_interruption_message(
        self,
        command: str,
        working_directory: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get interruption message for user confirmation."""
        return f"run tests: {command}"


class LinterTool(BaseTool[str]):
    """
    Tool to run code linters for various programming languages.

    Supports ruff (Python), pylint (Python), eslint (JavaScript/TypeScript),
    golangci-lint (Go), and other linters. Can fix issues automatically when
    the fix parameter is set to True.
    """

    SUPPORTED_LINTERS = ["ruff", "pylint", "eslint", "golangci-lint"]

    def __init__(self) -> None:
        """
        Initialize the linter tool.
        """
        super().__init__(
            name="run_linter",
            description=(
                "Run code linters to check code quality and style. "
                f"Supported linters: {', '.join(self.SUPPORTED_LINTERS)}. "
                "Use the 'fix' parameter to automatically fix issues where supported."
            ),
            parameters=[
                ParameterSchema(
                    name="linter",
                    type="string",
                    description=(
                        f"Linter to use: {', '.join(SUPPORTED_LINTERS)}. "
                        "Must be one of the supported linters."
                    ),
                ),
                ParameterSchema(
                    name="path",
                    type="string",
                    description=(
                        "Path to file or directory to lint. "
                        "Defaults to current directory."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="fix",
                    type="boolean",
                    description=(
                        "Whether to automatically fix issues where supported. "
                        "Not supported by all linters."
                    ),
                    required=False,
                    default=False,
                ),
            ],
            category=ToolCategory.CODE,
        )

    def execute(
        self,
        linter: str,
        path: Optional[str] = None,
        fix: bool = False,
    ) -> str:
        """
        Execute a linter synchronously.

        Args:
            linter: The linter to run (ruff, pylint, eslint, golangci-lint)
            path: Optional path to lint (file or directory)
            fix: Whether to automatically fix issues

        Returns:
            Linting results as a string
        """
        if linter not in self.SUPPORTED_LINTERS:
            return f"Error: Unsupported linter '{linter}'. Supported: {', '.join(self.SUPPORTED_LINTERS)}"

        # Build the command
        cmd_parts = [linter]

        # Add fix flag for supported linters
        if fix:
            if linter == "ruff":
                cmd_parts.append("--fix")
            elif linter == "eslint":
                cmd_parts.append("--fix")
            elif linter == "golangci-lint":
                cmd_parts.append("run", "--fix")
            # pylint doesn't have a standard --fix

        # Add path
        if path:
            cmd_parts.append(path)

        command = " ".join(cmd_parts)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output_parts = []

            if result.returncode == 0:
                output_parts.append("✓ Linting passed - no issues found")
            else:
                output_parts.append(f"✗ Linting found issues (exit code {result.returncode})")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"\n--- Output ---\n{stdout}")
            if stderr:
                output_parts.append(f"\n--- Errors ---\n{stderr}")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return "Error: Linting timed out after 120 seconds"
        except FileNotFoundError:
            return f"Error: Linter '{linter}' not found. Please install it first."
        except Exception as e:
            return f"Error running linter: {str(e)}"

    async def aexecute(
        self,
        linter: str,
        path: Optional[str] = None,
        fix: bool = False,
    ) -> str:
        """
        Execute a linter asynchronously.

        Args:
            linter: The linter to run
            path: Optional path to lint
            fix: Whether to automatically fix issues

        Returns:
            Linting results as a string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, linter, path, fix)

    def get_interruption_message(
        self,
        linter: str,
        path: Optional[str] = None,
        fix: bool = False,
        **kwargs,
    ) -> str:
        """Get interruption message for user confirmation."""
        return f"run linter: {linter} on {path or 'current directory'}"


class TypeCheckerTool(BaseTool[str]):
    """
    Tool to run static type checkers for various programming languages.

    Supports mypy (Python), tsc (TypeScript/JavaScript), and other type checkers.
    Helps catch type-related errors before runtime.
    """

    SUPPORTED_CHECKERS = ["mypy", "tsc", "flycheck"]

    def __init__(self) -> None:
        """
        Initialize the type checker tool.
        """
        super().__init__(
            name="run_type_checker",
            description=(
                "Run static type checkers to find type errors. "
                f"Supported checkers: {', '.join(self.SUPPORTED_CHECKERS)}. "
                "Type checking helps catch type-related errors before runtime."
            ),
            parameters=[
                ParameterSchema(
                    name="checker",
                    type="string",
                    description=(
                        f"Type checker to use: {', '.join(SUPPORTED_CHECKERS)}. "
                        "Must be one of the supported checkers."
                    ),
                ),
                ParameterSchema(
                    name="path",
                    type="string",
                    description=(
                        "Path to file or directory to type check. "
                        "Defaults to current directory."
                    ),
                    required=False,
                ),
            ],
            category=ToolCategory.CODE,
        )

    def execute(
        self,
        checker: str,
        path: Optional[str] = None,
    ) -> str:
        """
        Execute a type checker synchronously.

        Args:
            checker: The type checker to run (mypy, tsc, flycheck)
            path: Optional path to check (file or directory)

        Returns:
            Type checking results as a string
        """
        if checker not in SUPPORTED_CHECKERS:
            return f"Error: Unsupported type checker '{checker}'. Supported: {', '.join(SUPPORTED_CHECKERS)}"

        # Build the command
        cmd_parts = [checker]

        # Add path if provided
        if path:
            cmd_parts.append(path)

        # For tsc, always use --noEmit to just check types
        if checker == "tsc":
            cmd_parts.append("--noEmit")

        command = " ".join(cmd_parts)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output_parts = []

            if result.returncode == 0:
                output_parts.append("✓ Type checking passed - no errors found")
            else:
                output_parts.append(f"✗ Type checking found errors (exit code {result.returncode})")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"\n--- Output ---\n{stdout}")
            if stderr:
                output_parts.append(f"\n--- Errors ---\n{stderr}")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return "Error: Type checking timed out after 120 seconds"
        except FileNotFoundError:
            return f"Error: Type checker '{checker}' not found. Please install it first."
        except Exception as e:
            return f"Error running type checker: {str(e)}"

    async def aexecute(
        self,
        checker: str,
        path: Optional[str] = None,
    ) -> str:
        """
        Execute a type checker asynchronously.

        Args:
            checker: The type checker to run
            path: Optional path to check

        Returns:
            Type checking results as a string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute, checker, path)

    def get_interruption_message(
        self,
        checker: str,
        path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get interruption message for user confirmation."""
        return f"run type checker: {checker} on {path or 'current directory'}"


class GitTool(BaseTool[str]):
    """
    Tool to perform Git version control operations.

    Supports common Git operations including status, diff, commit, log,
    branch, pull, push, and more. Operations are performed using 'git' CLI.
    """

    SUPPORTED_OPERATIONS = [
        "status",
        "diff",
        "commit",
        "log",
        "branch",
        "pull",
        "push",
        "fetch",
        "add",
        "checkout",
        "stash",
        "merge",
        "rebase",
        "reset",
        "show",
    ]

    def __init__(self) -> None:
        """
        Initialize the Git tool.
        """
        super().__init__(
            name="git",
            description=(
                "Perform Git version control operations. "
                f"Supported operations: {', '.join(self.SUPPORTED_OPERATIONS)}. "
                "Use 'args' to pass additional Git arguments (e.g., ['--stat', '-n', '5'] for log)."
            ),
            parameters=[
                ParameterSchema(
                    name="operation",
                    type="string",
                    description=(
                        f"Git operation to perform: {', '.join(SUPPORTED_OPERATIONS)}"
                    ),
                ),
                ParameterSchema(
                    name="args",
                    type="array",
                    description=(
                        "Additional arguments for the Git command. "
                        "For example: ['--stat', '-n', '5'] for log, "
                        "['-m', 'commit message'] for commit, "
                        "['origin', 'main'] for push"
                    ),
                    required=False,
                    default=[],
                ),
                ParameterSchema(
                    name="working_directory",
                    type="string",
                    description=(
                        "Git repository root directory. "
                        "Defaults to current working directory."
                    ),
                    required=False,
                ),
            ],
            category=ToolCategory.CODE,
        )

    def execute(
        self,
        operation: str,
        args: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a Git operation synchronously.

        Args:
            operation: The Git operation (status, diff, commit, log, etc.)
            args: Additional arguments for the Git command
            working_directory: Repository directory

        Returns:
            Git operation results as a string
        """
        if operation not in self.SUPPORTED_OPERATIONS:
            return f"Error: Unsupported Git operation '{operation}'. Supported: {', '.join(self.SUPPORTED_OPERATIONS)}"

        args = args or []
        cwd = working_directory or os.getcwd()

        # Build the Git command
        cmd_parts = ["git", operation] + args
        command = " ".join(cmd_parts)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=cwd,
            )

            output_parts = []

            if result.returncode == 0:
                output_parts.append(f"✓ Git {operation} completed successfully")
            else:
                output_parts.append(f"✗ Git {operation} failed (exit code {result.returncode})")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"\n{stdout}")
            if stderr:
                output_parts.append(f"\n--- Errors ---\n{stderr}")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return f"Error: Git {operation} timed out after 60 seconds"
        except FileNotFoundError:
            return "Error: Git command not found. Is Git installed?"
        except Exception as e:
            return f"Error executing Git {operation}: {str(e)}"

    async def aexecute(
        self,
        operation: str,
        args: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a Git operation asynchronously.

        Args:
            operation: The Git operation
            args: Additional arguments
            working_directory: Repository directory

        Returns:
            Git operation results as a string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute, operation, args, working_directory
        )

    def get_interruption_message(
        self,
        operation: str,
        args: Optional[List[str]] = None,
        working_directory: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get interruption message for user confirmation."""
        return f"git {operation} {' '.join(args or [])}"


class PackageManagerTool(BaseTool[str]):
    """
    Tool to manage packages for various programming languages.

    Supports pip (Python), npm/yarn (JavaScript/TypeScript), and cargo (Rust).
    Can install, uninstall, list, and search for packages.
    """

    SUPPORTED_MANAGERS = ["pip", "npm", "yarn", "cargo"]
    SUPPORTED_OPERATIONS = ["install", "uninstall", "list", "search", "update", "outdated"]

    def __init__(self) -> None:
        """
        Initialize the package manager tool.
        """
        super().__init__(
            name="manage_packages",
            description=(
                "Manage packages for various programming languages. "
                f"Supported managers: {', '.join(self.SUPPORTED_MANAGERS)}. "
                f"Supported operations: {', '.join(self.SUPPORTED_OPERATIONS)}. "
                "Installs packages in the specified working directory."
            ),
            parameters=[
                ParameterSchema(
                    name="operation",
                    type="string",
                    description=(
                        f"Package operation: {', '.join(SUPPORTED_OPERATIONS)}"
                    ),
                ),
                ParameterSchema(
                    name="package_name",
                    type="string",
                    description=(
                        "Name of the package to install, uninstall, or search for. "
                        "Not required for 'list' or 'outdated' operations."
                    ),
                    required=False,
                ),
                ParameterSchema(
                    name="manager",
                    type="string",
                    description=(
                        f"Package manager to use: {', '.join(SUPPORTED_MANAGERS)}. "
                        "Must be one of the supported managers."
                    ),
                ),
                ParameterSchema(
                    name="working_directory",
                    type="string",
                    description=(
                        "Project directory for package management. "
                        "Defaults to current working directory."
                    ),
                    required=False,
                ),
            ],
            category=ToolCategory.CODE,
        )

    def execute(
        self,
        operation: str,
        package_name: Optional[str] = None,
        manager: str = "pip",
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a package management operation synchronously.

        Args:
            operation: The package operation (install, uninstall, list, search)
            package_name: Name of the package (for install/uninstall/search)
            manager: Package manager (pip, npm, yarn, cargo)
            working_directory: Project directory

        Returns:
            Package management results as a string
        """
        if manager not in self.SUPPORTED_MANAGERS:
            return f"Error: Unsupported package manager '{manager}'. Supported: {', '.join(self.SUPPORTED_MANAGERS)}"

        if operation not in self.SUPPORTED_OPERATIONS:
            return f"Error: Unsupported operation '{operation}'. Supported: {', '.join(self.SUPPORTED_OPERATIONS)}"

        # Validation: some operations require a package name
        if operation in ["install", "uninstall", "search"] and not package_name:
            return f"Error: Package name is required for '{operation}' operation"

        cwd = working_directory or os.getcwd()

        # Build the command based on manager and operation
        cmd_parts = []

        if manager == "pip":
            cmd_parts = self._build_pip_command(operation, package_name)
        elif manager == "npm":
            cmd_parts = self._build_npm_command(operation, package_name)
        elif manager == "yarn":
            cmd_parts = self._build_yarn_command(operation, package_name)
        elif manager == "cargo":
            cmd_parts = self._build_cargo_command(operation, package_name)

        command = " ".join(cmd_parts)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes for package operations
                cwd=cwd,
            )

            output_parts = []

            if result.returncode == 0:
                output_parts.append(f"✓ Package {operation} completed successfully")
            else:
                output_parts.append(f"✗ Package {operation} failed (exit code {result.returncode})")

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if stdout:
                output_parts.append(f"\n{stdout}")
            if stderr:
                output_parts.append(f"\n--- Errors ---\n{stderr}")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return f"Error: Package {operation} timed out after 180 seconds"
        except FileNotFoundError:
            return f"Error: Package manager '{manager}' not found. Please install it first."
        except Exception as e:
            return f"Error during package {operation}: {str(e)}"

    def _build_pip_command(
        self,
        operation: str,
        package_name: Optional[str],
    ) -> List[str]:
        """Build pip command based on operation."""
        if operation == "install":
            return ["pip", "install", package_name or ""]
        elif operation == "uninstall":
            return ["pip", "uninstall", "-y", package_name or ""]
        elif operation == "list":
            return ["pip", "list"]
        elif operation == "search":
            return ["pip", "search", package_name or ""] if package_name else ["pip", "list"]
        elif operation == "outdated":
            return ["pip", "list", "--outdated"]
        elif operation == "update":
            return ["pip", "install", "--upgrade", package_name or ""]
        return ["pip"]

    def _build_npm_command(
        self,
        operation: str,
        package_name: Optional[str],
    ) -> List[str]:
        """Build npm command based on operation."""
        if operation == "install":
            if package_name:
                return ["npm", "install", package_name]
            return ["npm", "install"]
        elif operation == "uninstall":
            return ["npm", "uninstall", package_name or ""]
        elif operation == "list":
            return ["npm", "ls", "--depth=0"]
        elif operation == "search":
            return ["npm", "search", package_name or ""] if package_name else ["npm", "ls"]
        elif operation == "outdated":
            return ["npm", "outdated"]
        elif operation == "update":
            return ["npm", "update", package_name or ""]
        return ["npm"]

    def _build_yarn_command(
        self,
        operation: str,
        package_name: Optional[str],
    ) -> List[str]:
        """Build yarn command based on operation."""
        if operation == "install":
            if package_name:
                return ["yarn", "add", package_name]
            return ["yarn"]
        elif operation == "uninstall":
            return ["yarn", "remove", package_name or ""]
        elif operation == "list":
            return ["yarn", "list", "--depth=0"]
        elif operation == "search":
            return ["yarn", "search", package_name or ""] if package_name else ["yarn", "list"]
        elif operation == "outdated":
            return ["yarn", "outdated"]
        elif operation == "update":
            return ["yarn", "upgrade", package_name or ""]
        return ["yarn"]

    def _build_cargo_command(
        self,
        operation: str,
        package_name: Optional[str],
    ) -> List[str]:
        """Build cargo command based on operation."""
        if operation == "install":
            return ["cargo", "install", package_name or ""]
        elif operation == "uninstall":
            return ["cargo", "uninstall", package_name or ""]
        elif operation == "list":
            return ["cargo", "tree"]
        elif operation == "search":
            return ["cargo", "search", package_name or ""] if package_name else ["cargo", "tree"]
        elif operation == "outdated":
            return ["cargo", "tree", "--outdated"]
        elif operation == "update":
            return ["cargo", "update", package_name or ""] if package_name else ["cargo", "update"]
        return ["cargo"]

    async def aexecute(
        self,
        operation: str,
        package_name: Optional[str] = None,
        manager: str = "pip",
        working_directory: Optional[str] = None,
    ) -> str:
        """
        Execute a package management operation asynchronously.

        Args:
            operation: The package operation
            package_name: Name of the package
            manager: Package manager
            working_directory: Project directory

        Returns:
            Package management results as a string
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.execute, operation, package_name, manager, working_directory
        )

    def get_interruption_message(
        self,
        operation: str,
        package_name: Optional[str] = None,
        manager: str = "pip",
        working_directory: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get interruption message for user confirmation."""
        msg = f"package {operation} with {manager}"
        if package_name:
            msg += f": {package_name}"
        return msg


def get_software_engineering_tools() -> list[BaseTool[str]]:
    """
    Get a list of all software engineering tools.

    Returns:
        List of software engineering tool instances
    """
    return [
        TestRunnerTool(),
        LinterTool(),
        TypeCheckerTool(),
        GitTool(),
        PackageManagerTool(),
    ]
