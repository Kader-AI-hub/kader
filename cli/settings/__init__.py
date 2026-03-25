"""Kader CLI Settings module.

Manages persistent user settings stored in ~/.kader/settings.json.
"""

from .settings import (
    KaderSettings,
    ensure_settings_file,
    get_settings_path,
    load_settings,
    migrate_settings,
    save_settings,
)

__all__ = [
    "KaderSettings",
    "get_settings_path",
    "load_settings",
    "save_settings",
    "ensure_settings_file",
    "migrate_settings",
]
