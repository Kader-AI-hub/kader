"""Sessions metadata management for Kader CLI.

Provides functionality to build and maintain a sessions metadata lock file
that tracks all Kader CLI sessions with their titles and metadata.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger


class SessionsMetadataManager:
    """Manages the sessions metadata lock file.

    The lock file is stored at ~/.kader/memory/sessions.json.lock and contains
    metadata for all valid Kader CLI sessions.

    Session is considered valid if:
    - It has a conversation.json file
    - Its agent_id is either 'kader_cli' or 'kader_cli_planner'
    """

    def __init__(self, memory_dir: Path | None = None) -> None:
        """Initialize the sessions metadata manager.

        Args:
            memory_dir: Path to the memory directory. Defaults to ~/.kader/memory
        """
        self.memory_dir = memory_dir or Path.home() / ".kader" / "memory"
        self.sessions_dir = self.memory_dir / "sessions"
        self.lock_file = self.memory_dir / "sessions.json.lock"

    def build_metadata(self) -> dict[str, dict[str, Any]]:
        """Build sessions metadata from all session directories.

        Returns:
            Dictionary mapping session_id to session metadata
        """
        metadata: dict[str, dict[str, Any]] = {}

        if not self.sessions_dir.exists():
            return metadata

        valid_agent_ids = {"kader_cli", "kader_cli_planner"}

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            conversation_file = session_dir / "conversation.json"

            if not session_file.exists() or not conversation_file.exists():
                continue

            try:
                session_data = json.loads(session_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            agent_id = session_data.get("agent_id", "")
            if agent_id not in valid_agent_ids:
                continue

            session_id = session_data.get("session_id", session_dir.name)

            metadata[session_id] = {
                "session-title": session_data.get("title", ""),
                "creation-date": session_data.get("created_at", ""),
                "update-date": session_data.get("updated_at", ""),
                "session-directory": str(session_dir),
            }

        return metadata

    def save_metadata(self, metadata: dict[str, dict[str, Any]]) -> None:
        """Save metadata to the lock file.

        Args:
            metadata: Dictionary mapping session_id to session metadata
        """
        try:
            self.memory_dir.mkdir(parents=True, exist_ok=True)
            self.lock_file.write_text(json.dumps(metadata, indent=2))
        except OSError as e:
            logger.error(f"Failed to save sessions metadata: {e}")

    def update(self) -> None:
        """Build and save sessions metadata."""
        metadata = self.build_metadata()
        self.save_metadata(metadata)

    def get_session_title(self, session_id: str) -> str | None:
        """Get the title of a specific session.

        Args:
            session_id: The session identifier

        Returns:
            The session title or None if not found
        """
        if not self.lock_file.exists():
            return None

        try:
            metadata = json.loads(self.lock_file.read_text())
            return metadata.get(session_id, {}).get("session-title")
        except (json.JSONDecodeError, OSError):
            return None


async def aupdate_sessions_metadata(memory_dir: Path | None = None) -> None:
    """Asynchronously update the sessions metadata lock file.

    Args:
        memory_dir: Path to the memory directory. Defaults to ~/.kader/memory
    """
    import asyncio

    manager = SessionsMetadataManager(memory_dir)
    await asyncio.to_thread(manager.update)
