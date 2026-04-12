"""Tool loader for Kader CLI.

Provides functionality to discover and load custom tools from directories,
similar to how callbacks and skills are loaded.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from loguru import logger

from kader.tools.base import BaseTool


class ToolLoader:
    """Loads tools from tools directories."""

    def __init__(
        self,
        tools_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the tool loader.

        Args:
            tools_dirs: List of directories to load tools from.
                        If None, defaults to ~/.kader/custom/tools
                        and ./.kader/custom/tools
            priority_dir: Optional directory to check first (higher priority).
                         If provided, this directory is checked before others.
        """
        self._priority_dir = priority_dir

        if tools_dirs is None:
            home_tools = Path.home() / ".kader" / "custom" / "tools"
            cwd_tools = Path.cwd() / ".kader" / "custom" / "tools"
            tools_dirs = [home_tools, cwd_tools]

        if priority_dir is not None:
            tools_dirs = [priority_dir] + tools_dirs

        self.tools_dirs = tools_dirs

    def _find_tool_file(self, name: str) -> tuple[Path, Path] | None:
        """
        Find the tool file in the tools directories.

        Args:
            name: Name of the tool to find

        Returns:
            Tuple of (tools_dir, tool_file) if found, None otherwise
        """
        for tools_dir in self.tools_dirs:
            if not tools_dir.exists():
                continue

            tool_file = tools_dir / f"{name}.py"
            if tool_file.exists():
                return (tools_dir, tool_file)

            tool_dir = tools_dir / name
            init_file = tool_dir / "__init__.py"
            if tool_dir.exists() and init_file.exists():
                return (tools_dir, init_file)

        return None

    def _get_tool_class(
        self, module_name: str, module_path: Path
    ) -> type[BaseTool] | None:
        """
        Find and return tool classes from a module.

        Args:
            module_name: Name of the module to search
            module_path: Path to the module file

        Returns:
            The first BaseTool subclass found, or None
        """
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module

            spec.loader.exec_module(module)

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseTool)
                    and obj is not BaseTool
                    and not obj.__name__.startswith("_")
                ):
                    return obj

            return None
        except Exception as e:
            logger.warning(f"Failed to load tool class from {module_path}: {e}")
            return None

    def load_tool(self, name: str) -> type[BaseTool] | None:
        """
        Load a tool by name.

        Args:
            name: Name of the tool to load (can include class name,
                  e.g., "my_tool.MyTool" or just "MyTool" or "my_tool")

        Returns:
            Tool class if found, None otherwise
        """
        if "." in name:
            parts = name.rsplit(".", 1)
            module_name = parts[0]
            class_name = parts[1]

            result = self._find_tool_file(module_name)
            if result is None:
                return None

            _, tool_file = result

            tool_class = self._get_tool_class(module_name, tool_file)
            if tool_class is None:
                return None

            if tool_class.__name__ != class_name:
                logger.warning(
                    f"Tool class name mismatch: expected {class_name}, "
                    f"found {tool_class.__name__}"
                )
                return None

            return tool_class

        result = self._find_tool_file(name)
        if result is None:
            return None

        _, tool_file = result

        module_name = name
        return self._get_tool_class(module_name, tool_file)

    def list_tools(self) -> list[type[BaseTool]]:
        """
        List all available tools from all directories.

        Returns:
            List of all available tool classes
        """
        tools: list[type[BaseTool]] = []
        seen_names: set[str] = set()

        for tools_dir in self.tools_dirs:
            if not tools_dir.exists():
                continue

            for tool_entry in sorted(tools_dir.iterdir()):
                if tool_entry.is_file():
                    if not tool_entry.suffix == ".py":
                        continue
                    if tool_entry.stem.startswith("_"):
                        continue

                    name = tool_entry.stem
                    if name in seen_names:
                        continue

                    tool_class = self._get_tool_class(name, tool_entry)
                    if tool_class:
                        tools.append(tool_class)
                        seen_names.add(name)

                elif tool_entry.is_dir():
                    init_file = tool_entry / "__init__.py"
                    if not init_file.exists():
                        continue

                    name = tool_entry.name
                    if name in seen_names:
                        continue

                    tool_class = self._get_tool_class(name, init_file)
                    if tool_class:
                        tools.append(tool_class)
                        seen_names.add(name)

        return tools


def load_tool_from_directory(tool_dir: Path) -> type[BaseTool] | None:
    """
    Load a tool from a specific directory.

    Args:
        tool_dir: Directory containing the tool (with __init__.py)

    Returns:
        Tool class if found, None otherwise
    """
    init_file = tool_dir / "__init__.py"
    if not init_file.exists():
        return None

    module_name = tool_dir.name
    loader = ToolLoader(tools_dirs=[tool_dir.parent])
    return loader._get_tool_class(module_name, init_file)


__all__ = ["ToolLoader", "load_tool_from_directory"]
