from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from kader.tools.base import BaseTool, ParameterSchema, ToolCategory


@dataclass
class Command:
    """Represents a special command with metadata and content."""

    name: str
    description: str
    content: str
    base_dir: Path


class CommandLoader:
    """Loads special commands from command directories."""

    def __init__(
        self, commands_dirs: list[Path] | None = None, priority_dir: Path | None = None
    ) -> None:
        """
        Initialize the command loader.

        Args:
            commands_dirs: List of directories to load commands from.
                         If None, defaults to ~/.kader/commands and ./.kader/
            priority_dir: Optional directory to check first (higher priority).
                         If provided, this directory is checked before others.
        """
        self._priority_dir = priority_dir

        if commands_dirs is None:
            home_commands = Path.home() / ".kader" / "commands"
            cwd_commands = Path.cwd() / ".kader" / "commands"
            commands_dirs = [home_commands, cwd_commands]

        if priority_dir is not None:
            commands_dirs = [priority_dir] + commands_dirs

        self.commands_dirs = commands_dirs

    def _find_command_file(self, name: str) -> tuple[Path, Path] | None:
        """
        Find the command file in the commands directories.

        Supports three formats:
        1. Directory with CONTENT.md: <command-name>/CONTENT.md (higher priority)
        2. Directory with README.md: <command-name>/README.md
        3. Direct file: <command-name>.md
        4. Sub-command: <command-name>/<subcommand>.md

        Args:
            name: Name of the command to find (can be "command" or "command/subcommand")

        Returns:
            Tuple of (command_dir, content_file) if found, None otherwise
        """
        for commands_dir in self.commands_dirs:
            if not commands_dir.exists():
                continue

            # Check if this is a sub-command (contains /)
            if "/" in name:
                parts = name.split("/", 1)
                command_name = parts[0]
                subcommand_name = parts[1]

                command_dir = commands_dir / command_name
                if not command_dir.is_dir():
                    continue

                # For sub-commands, check the specific .md file first
                # Then fall back to CONTENT.md or README.md in the directory
                md_file = command_dir / f"{subcommand_name}.md"
                if md_file.exists():
                    return (commands_dir, md_file)

                # Fall back to CONTENT.md or README.md
                for md_name in ["CONTENT.md", "README.md"]:
                    content_file = command_dir / md_name
                    if content_file.exists():
                        return (commands_dir, content_file)
                continue

            # Regular command (no sub-command)
            # First check for directory with CONTENT.md (higher priority)
            command_dir = commands_dir / name

            # Check for CONTENT.md or README.md in directory
            for md_name in ["CONTENT.md", "README.md"]:
                content_file = command_dir / md_name
                if content_file.exists():
                    return (commands_dir, content_file)

            # Then check for direct .md file
            md_file = commands_dir / f"{name}.md"
            if md_file.exists():
                return (commands_dir, md_file)

        return None

    def load_command(self, name: str) -> Command | None:
        """
        Load a command by name.

        Supports four formats:
        1. Directory: <command-name>/CONTENT.md or README.md
        2. Direct file: <command-name>.md
        3. Sub-command: <command-name>/<subcommand>.md

        Args:
            name: Name of the command to load (can be "command" or "command/subcommand")

        Returns:
            Command object if found, None otherwise
        """
        result = self._find_command_file(name)
        if result is None:
            return None

        commands_dir, content_file = result
        parsed = frontmatter.load(str(content_file))

        metadata = parsed.metadata or {}

        description = str(metadata.get("description", "")).strip()
        if not description:
            first_line = parsed.content.strip().split("\n")[0][:100]
            description = first_line if first_line else "No description available"

            # Determine base_dir and final command name
        if "/" in name:
            # Sub-command: command/subcommand
            command_name = name.split("/", 1)[0]
            full_name = name  # Keep as command/subcommand
            base_dir = commands_dir / command_name
        elif content_file.name in ["CONTENT.md", "README.md"]:
            # Directory format: command/CONTENT.md or command/README.md
            full_name = name
            base_dir = content_file.parent
        else:
            # Direct file format: command.md
            full_name = name
            base_dir = commands_dir / name

        return Command(
            name=full_name,
            description=description,
            content=parsed.content,
            base_dir=base_dir,
        )

    def list_commands(self) -> list[Command]:
        """
        List all available commands from all directories.

        Supports four formats:
        1. Directory: <command-name>/CONTENT.md or README.md
        2. Direct file: <command-name>.md
        3. Sub-commands: <command-name>/<subcommand>.md

        Commands from priority_dir take priority, then ~/.kader/commands,
        then ./.kader/commands.

        Returns:
            List of all available commands
        """
        seen_names: set[str] = set()
        commands: list[Command] = []

        for commands_dir in self.commands_dirs:
            if not commands_dir.exists():
                continue

            # Collect command names from directories and .md files
            command_names: set[str] = set()

            # First, add directory names
            for entry in commands_dir.iterdir():
                if entry.is_dir():
                    command_names.add(entry.name)

            # Then, add .md file names (without extension)
            for entry in commands_dir.iterdir():
                if entry.is_file() and entry.suffix == ".md":
                    cmd_name = entry.stem
                    # Only add if not already found as a directory
                    if cmd_name not in command_names:
                        command_names.add(cmd_name)

            # Load each top-level command
            for name in sorted(command_names):
                if name in seen_names:
                    continue

                command = self.load_command(name)
                if command:
                    commands.append(command)
                    seen_names.add(name)

            # Now find sub-commands within directories
            for entry in sorted(commands_dir.iterdir()):
                if not entry.is_dir():
                    continue

                command_name = entry.name
                # Look for .md files in the subdirectory (sub-commands)
                for md_file in sorted(entry.iterdir()):
                    if not md_file.is_file() or md_file.suffix != ".md":
                        continue

                    # Skip CONTENT.md and README.md as they're handled above
                    if md_file.name in ["CONTENT.md", "README.md"]:
                        continue

                    subcommand_name = md_file.stem
                    full_name = f"{command_name}/{subcommand_name}"

                    if full_name in seen_names:
                        continue

                    command = self.load_command(full_name)
                    if command:
                        commands.append(command)
                        seen_names.add(full_name)

        return commands

    def get_description(self) -> str:
        """
        Get a formatted description of all available commands.

        Returns:
            Formatted string listing all commands
        """
        commands = self.list_commands()

        if not commands:
            return "No special commands available."

        commands_listing = "\n".join(f"  - {c.name}: {c.description}" for c in commands)

        return commands_listing


class CommandsTool(BaseTool[dict[str, Any]]):
    """Tool for loading and managing special commands (not used in default registry)."""

    def __init__(
        self,
        commands_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the commands tool.

        Args:
            commands_dirs: Optional list of custom command directories.
                         Defaults to ~/.kader/commands and ./.kader/
            priority_dir: Optional directory to check first (higher priority).
        """
        self._command_loader = CommandLoader(commands_dirs, priority_dir)

        description = (
            "Execute a special command to perform specific tasks. "
            "Use this when you need to invoke a custom command agent. "
            f"Available commands:\n{self._command_loader.get_description()}"
        )

        super().__init__(
            name="commands_tool",
            description=description,
            category=ToolCategory.UTILITY,
            parameters=[
                ParameterSchema(
                    name="name",
                    type="string",
                    description="The exact name of the command to execute",
                ),
                ParameterSchema(
                    name="task",
                    type="string",
                    description="The task to execute with this command",
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the commands tool.

        Args:
            **kwargs: Must include 'name' parameter, optionally 'task'

        Returns:
            Dictionary with command information
        """
        name = kwargs.get("name", "")
        task = kwargs.get("task", "")

        if not name:
            return {
                "error": "Command name is required",
                "available_commands": self._command_loader.get_description(),
            }

        command = self._command_loader.load_command(name)
        if command is None:
            available = self._command_loader.get_description()
            return {
                "error": f"Command '{name}' not found",
                "available_commands": available,
            }

        return {
            "name": command.name,
            "description": command.description,
            "content": command.content,
            "task": task,
            "base_dir": str(command.base_dir),
        }

    async def aexecute(self, **kwargs: Any) -> dict[str, Any]:
        """Asynchronous execution (delegates to synchronous)."""
        return self.execute(**kwargs)

    def get_interruption_message(self, **kwargs: Any) -> str:
        """
        Get interruption message for user confirmation.

        Args:
            **kwargs: Must include 'name' parameter

        Returns:
            Message describing the command execution action
        """
        name = kwargs.get("name", "")
        return f"execute command: {name}"
