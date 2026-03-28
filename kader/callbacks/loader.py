"""Callback loader for Kader framework.

Provides functionality to discover and load custom callbacks from directories,
similar to how skills and commands are loaded.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path

from loguru import logger

from kader.callbacks.base import BaseCallback


class CallbackLoader:
    """Loads callbacks from callback directories."""

    def __init__(
        self,
        callbacks_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the callback loader.

        Args:
            callbacks_dirs: List of directories to load callbacks from.
                            If None, defaults to ~/.kader/custom/callbacks
                            and ./.kader/custom/callbacks
            priority_dir: Optional directory to check first (higher priority).
                         If provided, this directory is checked before others.
        """
        self._priority_dir = priority_dir

        if callbacks_dirs is None:
            home_callbacks = Path.home() / ".kader" / "custom" / "callbacks"
            cwd_callbacks = Path.cwd() / ".kader" / "custom" / "callbacks"
            callbacks_dirs = [home_callbacks, cwd_callbacks]

        if priority_dir is not None:
            callbacks_dirs = [priority_dir] + callbacks_dirs

        self.callbacks_dirs = callbacks_dirs

    def _find_callback_file(self, name: str) -> tuple[Path, Path] | None:
        """
        Find the callback file in the callbacks directories.

        Args:
            name: Name of the callback to find

        Returns:
            Tuple of (callbacks_dir, callback_file) if found, None otherwise
        """
        for callbacks_dir in self.callbacks_dirs:
            if not callbacks_dir.exists():
                continue
            callback_file = callbacks_dir / f"{name}.py"
            if callback_file.exists():
                return (callbacks_dir, callback_file)
        return None

    def _get_callback_class(
        self, module_name: str, module_path: Path
    ) -> type[BaseCallback] | None:
        """
        Find and return callback classes from a module.

        Args:
            module_name: Name of the module to search
            module_path: Path to the module file

        Returns:
            The first BaseCallback subclass found, or None
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
                    issubclass(obj, BaseCallback)
                    and obj is not BaseCallback
                    and not obj.__name__.startswith("_")
                ):
                    return obj

            return None
        except Exception as e:
            logger.warning(f"Failed to load callback class from {module_path}: {e}")
            return None

    def load_callback(self, name: str) -> type[BaseCallback] | None:
        """
        Load a callback by name.

        Args:
            name: Name of the callback to load (can include module path,
                  e.g., "my_callback.MyCallback" or just "MyCallback")

        Returns:
            Callback class if found, None otherwise
        """
        if "." in name:
            parts = name.rsplit(".", 1)
            module_name = parts[0]
            class_name = parts[1]

            result = self._find_callback_file(module_name)
            if result is None:
                return None

            _, callback_file = result

            callback_class = self._get_callback_class(module_name, callback_file)
            if callback_class is None:
                return None

            if callback_class.__name__ != class_name:
                logger.warning(
                    f"Callback class name mismatch: expected {class_name}, "
                    f"found {callback_class.__name__}"
                )
                return None

            return callback_class

        result = self._find_callback_file(name)
        if result is None:
            return None

        _, callback_file = result

        module_name = name
        return self._get_callback_class(module_name, callback_file)

    def list_callbacks(self) -> list[type[BaseCallback]]:
        """
        List all available callbacks from all directories.

        Returns:
            List of all available callback classes
        """
        callbacks: list[type[BaseCallback]] = []
        seen_names: set[str] = set()

        for callbacks_dir in self.callbacks_dirs:
            if not callbacks_dir.exists():
                continue

            for callback_file in sorted(callbacks_dir.iterdir()):
                if not callback_file.is_file() or not callback_file.suffix == ".py":
                    continue

                if callback_file.stem.startswith("_"):
                    continue

                name = callback_file.stem
                if name in seen_names:
                    continue

                callback_class = self._get_callback_class(name, callback_file)
                if callback_class:
                    callbacks.append(callback_class)
                    seen_names.add(name)

        return callbacks
