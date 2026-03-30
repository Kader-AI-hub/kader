"""CLI callbacks integration for Kader CLI.

Provides functionality to load custom callbacks from user and project directories.
"""

from pathlib import Path

from loguru import logger

from cli.settings.settings import KaderSettings
from kader.callbacks.base import BaseCallback
from kader.callbacks.loader import CallbackLoader


def load_callbacks_from_settings(settings: KaderSettings) -> list[BaseCallback]:
    """
    Load callbacks based on settings configuration.

    Loading order:
    1. Project-level callbacks from ./.kader/custom/callbacks (always enabled)
    2. User-level callbacks from settings.json (controlled by 'enabled' field)

    Args:
        settings: KaderSettings containing callback configuration

    Returns:
        List of instantiated callback objects
    """
    callbacks: list[BaseCallback] = []

    project_callbacks_dir = Path.cwd() / ".kader" / "custom" / "callbacks"
    user_callbacks_dir = Path.home() / ".kader" / "custom" / "callbacks"

    project_loader = CallbackLoader(callbacks_dirs=[project_callbacks_dir])
    user_loader = CallbackLoader(callbacks_dirs=[user_callbacks_dir])

    project_callback_classes = project_loader.list_callbacks()
    for callback_class in project_callback_classes:
        try:
            instance = callback_class()
            if isinstance(instance, BaseCallback):
                callbacks.append(instance)
                logger.info(f"Loaded project callback: {callback_class.__name__}")
        except Exception as e:
            logger.warning(
                f"Failed to instantiate project callback {callback_class.__name__}: {e}"
            )

    callback_configs = settings.callbacks or []
    for config in callback_configs:
        if not isinstance(config, dict):
            continue

        name = config.get("name")
        enabled_str = config.get("enabled", "true")
        enabled = (
            enabled_str.lower() == "true"
            if isinstance(enabled_str, str)
            else bool(enabled_str)
        )

        if not name:
            continue

        if not enabled:
            logger.debug(f"Skipping disabled callback: {name}")
            continue

        callback_class = user_loader.load_callback(name)
        if callback_class is None:
            logger.debug(f"User callback not found: {name}")
            continue

        try:
            instance = callback_class()
            if isinstance(instance, BaseCallback):
                callbacks.append(instance)
                logger.info(f"Loaded user callback: {name}")
        except Exception as e:
            logger.warning(f"Failed to instantiate user callback {name}: {e}")

    return callbacks


__all__ = ["load_callbacks_from_settings"]
