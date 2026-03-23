"""Kader CLI Settings module.

Manages persistent user settings stored in ~/.kader/settings.json.
"""

from .settings import KaderSettings, get_settings_path, load_settings, save_settings

__all__ = [
    "KaderSettings",
    "get_settings_path",
    "load_settings",
    "save_settings",
]
