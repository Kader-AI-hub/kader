"""Kader CLI Settings management.

Provides a dataclass for validating and persisting user settings in
~/.kader/settings.json, including model and provider preferences for
the main (planner) and sub (executor) agents.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

# Valid provider names (kept in sync with LLMProviderFactory.PROVIDERS)
VALID_PROVIDERS: set[str] = {
    "ollama",
    "google",
    "mistral",
    "anthropic",
    "openai",
    "moonshot",
    "zai",
    "openrouter",
    "opencode",
    "groq",
}

# Defaults matching the current DEFAULT_MODEL = "glm-5:cloud" (ollama)
_DEFAULT_PROVIDER = "ollama"
_DEFAULT_MAIN_MODEL = "glm-5:cloud"
_DEFAULT_SUB_MODEL = "glm-5:cloud"

# Mapping between JSON hyphenated keys and Python snake_case fields
_JSON_KEY_MAP: dict[str, str] = {
    "main-agent-provider": "main_agent_provider",
    "sub-agent-provider": "sub_agent_provider",
    "main-agent-model": "main_agent_model",
    "sub-agent-model": "sub_agent_model",
    "auto-update": "auto_update",
    "callbacks": "callbacks",
}

_FIELD_KEY_MAP: dict[str, str] = {v: k for k, v in _JSON_KEY_MAP.items()}


@dataclass
class KaderSettings:
    """Persistent user settings for Kader CLI.

    Attributes:
        main_agent_provider: LLM provider name for the planner agent.
        sub_agent_provider: LLM provider name for executor sub-agents.
        main_agent_model: Model identifier for the planner agent.
        sub_agent_model: Model identifier for executor sub-agents.
        auto_update: Whether to automatically update Kader on startup.
        callbacks: List of user-level callbacks to enable.
                   Format: [{"name": "module.ClassName", "enabled": "true/false"}]
    """

    VALID_PROVIDERS: ClassVar[set[str]] = VALID_PROVIDERS

    main_agent_provider: str = field(default=_DEFAULT_PROVIDER)
    sub_agent_provider: str = field(default=_DEFAULT_PROVIDER)
    main_agent_model: str = field(default=_DEFAULT_MAIN_MODEL)
    sub_agent_model: str = field(default=_DEFAULT_SUB_MODEL)
    auto_update: bool = field(default=False)
    callbacks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate provider values after initialisation."""
        for attr in ("main_agent_provider", "sub_agent_provider"):
            value = getattr(self, attr)
            if value and value not in VALID_PROVIDERS:
                raise ValueError(
                    f"Invalid provider '{value}' for {attr}. "
                    f"Valid providers: {sorted(VALID_PROVIDERS)}"
                )

    # ── Serialisation helpers ─────────────────────────────────────────

    def to_dict(self) -> dict[str, str]:
        """Return settings as a dict with hyphenated JSON keys."""
        return {_FIELD_KEY_MAP[f]: getattr(self, f) for f in _FIELD_KEY_MAP}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KaderSettings:
        """Create a ``KaderSettings`` from a dict with hyphenated JSON keys.

        Unknown keys are silently ignored; missing keys use defaults.
        """
        kwargs: dict[str, Any] = {}
        for json_key, field_name in _JSON_KEY_MAP.items():
            if json_key in data:
                if field_name == "auto_update":
                    value = data[json_key]
                    if isinstance(value, bool):
                        kwargs[field_name] = value
                    elif isinstance(value, str):
                        kwargs[field_name] = value.lower() == "true"
                    else:
                        kwargs[field_name] = bool(value)
                elif field_name == "callbacks":
                    value = data[json_key]
                    if isinstance(value, list):
                        kwargs[field_name] = value
                else:
                    kwargs[field_name] = data[json_key]
        return cls(**kwargs)

    def get_main_model_string(self) -> str:
        """Return the full ``provider:model`` string for the main agent."""
        return f"{self.main_agent_provider}:{self.main_agent_model}"

    def get_sub_model_string(self) -> str:
        """Return the full ``provider:model`` string for the sub agent."""
        return f"{self.sub_agent_provider}:{self.sub_agent_model}"


# ── File I/O ──────────────────────────────────────────────────────────


def get_settings_path() -> Path:
    """Return the path to ``~/.kader/settings.json``."""
    return Path.home() / ".kader" / "settings.json"


def load_settings(path: Path | None = None) -> KaderSettings:
    """Load settings from *path* (defaults to ``get_settings_path()``).

    Returns default settings when the file is missing or malformed.
    """
    path = path or get_settings_path()

    if not path.exists():
        return KaderSettings()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return KaderSettings.from_dict(data)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(f"Failed to parse settings at {path}: {exc}")
        return KaderSettings()


def save_settings(settings: KaderSettings, path: Path | None = None) -> None:
    """Persist *settings* to *path* (defaults to ``get_settings_path()``)."""
    path = path or get_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(settings.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_settings_file(path: Path | None = None) -> KaderSettings:
    """Create ``settings.json`` with defaults if it does not exist.

    Returns the loaded (or newly created) settings.
    """
    path = path or get_settings_path()

    if not path.exists():
        settings = KaderSettings()
        save_settings(settings, path)
        return settings

    return load_settings(path)


def migrate_settings(path: Path | None = None) -> KaderSettings:
    """Check for missing keys in settings file and add them with defaults.

    Reads the existing settings file, adds any missing keys with their
    default values, and saves the updated file. Existing keys are preserved.
    Also auto-discovers user-level callbacks from ~/.kader/custom/callbacks
    and adds them to settings if not already present.

    Returns the loaded (potentially migrated) settings.
    """
    path = path or get_settings_path()

    if not path.exists():
        return KaderSettings()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(f"Failed to parse settings at {path}: {exc}")
        return KaderSettings()

    defaults = KaderSettings()
    default_dict = defaults.to_dict()

    migrated = False
    for key, default_value in default_dict.items():
        if key not in data:
            data[key] = default_value
            migrated = True
            logger.info(f"Added missing setting '{key}' with default value")

    data = _migrate_user_callbacks(data)
    if data.get("_callbacks_migrated"):
        migrated = True
        del data["_callbacks_migrated"]

    if migrated:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info(f"Migrated settings file at {path}")

    return KaderSettings.from_dict(data)


def _migrate_user_callbacks(data: dict[str, Any]) -> dict[str, Any]:
    """Auto-discover user-level callbacks and add them to settings.

    Checks ~/.kader/custom/callbacks for callback files and adds any
    that are not already in the settings callbacks list.
    Creates the directory if it doesn't exist.

    Args:
        data: Existing settings data dict

    Returns:
        Updated data dict with discovered callbacks
    """
    user_callbacks_dir = Path.home() / ".kader" / "custom" / "callbacks"

    if not user_callbacks_dir.exists():
        try:
            user_callbacks_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created user callbacks directory: {user_callbacks_dir}")
        except Exception as e:
            logger.warning(f"Failed to create user callbacks directory: {e}")
            return data
        return data

    existing_callbacks = data.get("callbacks", [])
    existing_names = {
        cb.get("name") for cb in existing_callbacks if isinstance(cb, dict)
    }

    discovered: list[dict[str, str]] = []
    for callback_file in user_callbacks_dir.iterdir():
        if not callback_file.is_file() or callback_file.suffix != ".py":
            continue
        if callback_file.stem.startswith("_"):
            continue

        callback_name = callback_file.stem
        if callback_name not in existing_names:
            discovered.append({"name": callback_name, "enabled": "false"})
            logger.info(f"Discovered user callback: {callback_name}")

    if discovered:
        data["callbacks"] = existing_callbacks + discovered
        data["_callbacks_migrated"] = True
        logger.info(f"Added {len(discovered)} user callbacks to settings")

    return data
