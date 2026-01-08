"""
Core types and dataclasses for the Memory module.

Provides common types used across state, session, and conversation management.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class SessionType(str, Enum):
    """Enumeration of session types."""

    AGENT = "AGENT"
    MULTI_AGENT = "MULTI_AGENT"


def get_default_memory_dir() -> Path:
    """Get the default memory directory path ($HOME/.kader/memory)."""
    home = Path.home()
    return home / ".kader" / "memory"


@dataclass
class MemoryConfig:
    """Configuration for memory management.

    Attributes:
        memory_dir: Root directory for memory storage
        auto_save: Whether to automatically save state changes
        max_conversation_length: Maximum messages to keep in conversation history
    """

    memory_dir: Path = field(default_factory=get_default_memory_dir)
    auto_save: bool = True
    max_conversation_length: int = 100

    def __post_init__(self) -> None:
        """Ensure memory_dir is a Path object."""
        if isinstance(self.memory_dir, str):
            self.memory_dir = Path(self.memory_dir)

    def ensure_directories(self) -> None:
        """Create memory directories if they don't exist."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        (self.memory_dir / "sessions").mkdir(exist_ok=True)


def encode_bytes_values(obj: Any) -> Any:
    """Recursively encode any bytes values in an object to base64.

    Handles dictionaries, lists, and nested structures.
    """
    import base64

    if isinstance(obj, bytes):
        return {"__bytes_encoded__": True, "data": base64.b64encode(obj).decode()}
    elif isinstance(obj, dict):
        return {k: encode_bytes_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [encode_bytes_values(item) for item in obj]
    else:
        return obj


def decode_bytes_values(obj: Any) -> Any:
    """Recursively decode any base64-encoded bytes values in an object.

    Handles dictionaries, lists, and nested structures.
    """
    import base64

    if isinstance(obj, dict):
        if obj.get("__bytes_encoded__") is True and "data" in obj:
            return base64.b64decode(obj["data"])
        return {k: decode_bytes_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decode_bytes_values(item) for item in obj]
    else:
        return obj


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Save data to a JSON file.

    Args:
        path: Path to the JSON file
        data: Data to save
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(encode_bytes_values(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> dict[str, Any]:
    """Load data from a JSON file.

    Args:
        path: Path to the JSON file

    Returns:
        Loaded data, or empty dict if file doesn't exist
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return decode_bytes_values(json.load(f))
