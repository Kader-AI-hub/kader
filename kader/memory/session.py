"""
Session management for agents.

Provides session persistence with file-based storage for
agent state and conversation history.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .state import AgentState
from .types import (
    MemoryConfig,
    SessionType,
    aload_json,
    asave_json,
    get_timestamp,
    load_json,
    save_json,
)


@dataclass
class Session:
    """Session metadata.

    Represents a unique session that groups together an agent's
    state and conversation history.

    Attributes:
        session_id: Unique identifier for the session
        agent_id: Associated agent identifier
        session_type: Type of session (AGENT, MULTI_AGENT)
        created_at: ISO timestamp when session was created
        updated_at: ISO timestamp when session was last updated
        title: Optional session title for display
    """

    session_id: str
    agent_id: str
    session_type: SessionType = SessionType.AGENT
    created_at: str = field(default_factory=get_timestamp)
    updated_at: str = field(default_factory=get_timestamp)
    title: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary for serialization.

        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "session_type": self.session_type.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Session":
        """Create Session from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New Session instance
        """
        return cls(
            session_id=data.get("session_id", ""),
            agent_id=data.get("agent_id", ""),
            session_type=SessionType(data.get("session_type", "AGENT")),
            created_at=data.get("created_at", get_timestamp()),
            updated_at=data.get("updated_at", get_timestamp()),
            title=data.get("title"),
        )


class SessionManager(ABC):
    """Abstract base class for session management.

    Provides interface for persisting agent sessions, state,
    and conversation history.
    """

    @abstractmethod
    def create_session(self, agent_id: str) -> Session:
        """Create a new session for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            New Session instance
        """
        ...

    @abstractmethod
    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        ...

    @abstractmethod
    def list_sessions(self, agent_id: str | None = None) -> list[Session]:
        """List all sessions, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of sessions
        """
        ...

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    def save_agent_state(self, session_id: str, state: AgentState) -> None:
        """Save agent state for a session.

        Args:
            session_id: Session identifier
            state: AgentState to persist
        """
        ...

    @abstractmethod
    def load_agent_state(self, session_id: str, agent_id: str) -> AgentState:
        """Load agent state for a session.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier (used if state doesn't exist)

        Returns:
            AgentState (new if not found)
        """
        ...

    @abstractmethod
    def save_conversation(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save conversation history for a session.

        Args:
            session_id: Session identifier
            messages: List of message dictionaries
        """
        ...

    @abstractmethod
    def load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        ...

    @abstractmethod
    async def async_create_session(self, agent_id: str) -> "Session":
        """Create a new session for an agent (async).

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            New Session instance
        """
        ...

    @abstractmethod
    async def async_get_session(self, session_id: str) -> "Session | None":
        """Retrieve a session by ID (async).

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        ...

    @abstractmethod
    async def async_list_sessions(self, agent_id: str | None = None) -> list["Session"]:
        """List all sessions, optionally filtered by agent (async).

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of sessions
        """
        ...

    @abstractmethod
    async def async_delete_session(self, session_id: str) -> bool:
        """Delete a session (async).

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def async_save_agent_state(self, session_id: str, state: AgentState) -> None:
        """Save agent state for a session (async).

        Args:
            session_id: Session identifier
            state: AgentState to persist
        """
        ...

    @abstractmethod
    async def async_load_agent_state(
        self, session_id: str, agent_id: str
    ) -> AgentState:
        """Load agent state for a session (async).

        Args:
            session_id: Session identifier
            agent_id: Agent identifier (used if state doesn't exist)

        Returns:
            AgentState (new if not found)
        """
        ...

    @abstractmethod
    async def async_save_conversation(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save conversation history for a session (async).

        Args:
            session_id: Session identifier
            messages: List of message dictionaries
        """
        ...

    @abstractmethod
    async def async_load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a session (async).

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        ...


class FileSessionManager(SessionManager):
    """Filesystem-based session manager.

    Stores session data as JSON files in the following structure:

    $HOME/.kader/memory/sessions/
    └── {session_id}/
        ├── session.json       # Session metadata
        ├── state.json         # Agent state
        └── conversation.json  # Conversation history

    Attributes:
        config: Memory configuration
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        """Initialize the file session manager.

        Args:
            config: Optional memory configuration
        """
        self.config = config or MemoryConfig()
        self.config.ensure_directories()

    @property
    def sessions_dir(self) -> Path:
        """Get the sessions directory path."""
        return self.config.memory_dir / "sessions"

    def _session_dir(self, session_id: str) -> Path:
        """Get the directory path for a specific session."""
        return self.sessions_dir / session_id

    def _session_file(self, session_id: str) -> Path:
        """Get the session metadata file path."""
        return self._session_dir(session_id) / "session.json"

    def _state_file(self, session_id: str) -> Path:
        """Get the state file path."""
        return self._session_dir(session_id) / "state.json"

    def _conversation_file(self, session_id: str) -> Path:
        """Get the conversation file path."""
        return self._session_dir(session_id) / "conversation.json"

    def create_session(self, agent_id: str) -> Session:
        """Create a new session for an agent.

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            New Session instance
        """
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            agent_id=agent_id,
        )

        # Create session directory and save metadata
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        save_json(self._session_file(session_id), session.to_dict())

        return session

    def get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return None

        data = load_json(session_file)
        return Session.from_dict(data) if data else None

    def list_sessions(self, agent_id: str | None = None) -> list[Session]:
        """List all sessions, optionally filtered by agent.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of sessions
        """
        sessions: list[Session] = []

        if not self.sessions_dir.exists():
            return sessions

        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue

            session_file = session_dir / "session.json"
            if not session_file.exists():
                continue

            data = load_json(session_file)
            if not data:
                continue

            session = Session.from_dict(data)

            if agent_id is None or session.agent_id == agent_id:
                sessions.append(session)

        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        import shutil

        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False

        shutil.rmtree(session_dir)
        return True

    def save_agent_state(self, session_id: str, state: AgentState) -> None:
        """Save agent state for a session.

        Args:
            session_id: Session identifier
            state: AgentState to persist
        """
        state_file = self._state_file(session_id)
        save_json(state_file, state.to_dict())

        # Update session timestamp
        self._update_session_timestamp(session_id)

    def load_agent_state(self, session_id: str, agent_id: str) -> AgentState:
        """Load agent state for a session.

        Args:
            session_id: Session identifier
            agent_id: Agent identifier (used if state doesn't exist)

        Returns:
            AgentState (new if not found)
        """
        state_file = self._state_file(session_id)
        data = load_json(state_file)

        if data:
            return AgentState.from_dict(data)

        return AgentState(agent_id=agent_id)

    def save_conversation(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save conversation history for a session.

        Args:
            session_id: Session identifier
            messages: List of message dictionaries
        """
        conversation_file = self._conversation_file(session_id)
        save_json(conversation_file, {"messages": messages})

        # Update session timestamp
        self._update_session_timestamp(session_id)

    def load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        conversation_file = self._conversation_file(session_id)
        data = load_json(conversation_file)
        return data.get("messages", [])

    def _update_session_timestamp(self, session_id: str) -> None:
        """Update the session's updated_at timestamp.

        Args:
            session_id: Session identifier
        """
        session = self.get_session(session_id)
        if session:
            session.updated_at = get_timestamp()
            save_json(self._session_file(session_id), session.to_dict())

    async def async_create_session(self, agent_id: str) -> Session:
        return await asyncio.to_thread(self.create_session, agent_id)

    async def async_get_session(self, session_id: str) -> Session | None:
        return await asyncio.to_thread(self.get_session, session_id)

    async def async_list_sessions(self, agent_id: str | None = None) -> list[Session]:
        return await asyncio.to_thread(self.list_sessions, agent_id)

    async def async_delete_session(self, session_id: str) -> bool:
        return await asyncio.to_thread(self.delete_session, session_id)

    async def async_save_agent_state(self, session_id: str, state: AgentState) -> None:
        await asyncio.to_thread(self.save_agent_state, session_id, state)

    async def async_load_agent_state(
        self, session_id: str, agent_id: str
    ) -> AgentState:
        return await asyncio.to_thread(self.load_agent_state, session_id, agent_id)

    async def async_save_conversation(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(self.save_conversation, session_id, messages)

    async def async_load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.load_conversation, session_id)


class AsyncFileSessionManager(FileSessionManager):
    """Async filesystem-based session manager.

    Provides the same functionality as FileSessionManager but with
    non-blocking I/O operations for use in async contexts.

    All methods that perform I/O are async and use aiofiles-backed
    utilities from types.py.
    """

    def _ensure_directories(self) -> None:
        if not hasattr(self, "_dirs_initialized"):
            self.config.ensure_directories()
            self._dirs_initialized = True

    async def async_create_session(self, agent_id: str) -> Session:
        """Create a new session for an agent (async).

        Args:
            agent_id: Unique identifier for the agent

        Returns:
            New Session instance
        """
        self._ensure_directories()
        session_id = str(uuid.uuid4())
        session = Session(
            session_id=session_id,
            agent_id=agent_id,
        )
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        await asave_json(self._session_file(session_id), session.to_dict())
        return session

    async def async_get_session(self, session_id: str) -> Session | None:
        """Retrieve a session by ID (async).

        Args:
            session_id: Session identifier

        Returns:
            Session if found, None otherwise
        """
        session_file = self._session_file(session_id)
        if not session_file.exists():
            return None
        data = await aload_json(session_file)
        return Session.from_dict(data) if data else None

    async def async_list_sessions(self, agent_id: str | None = None) -> list[Session]:
        """List all sessions, optionally filtered by agent (async).

        Uses asyncio.gather to load session files concurrently.

        Args:
            agent_id: Optional agent ID to filter by

        Returns:
            List of sessions sorted by created_at descending
        """
        if not self.sessions_dir.exists():
            return []

        tasks: list[asyncio.Task[Session | None]] = []
        for session_dir in self.sessions_dir.iterdir():
            if not session_dir.is_dir():
                continue
            session_file = session_dir / "session.json"
            if session_file.exists():
                tasks.append(
                    asyncio.create_task(
                        self._async_load_session_file(session_file, agent_id)
                    )
                )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks)
        sessions = [r for r in results if r is not None]
        sessions.sort(key=lambda s: s.created_at, reverse=True)
        return sessions

    async def _async_load_session_file(
        self, path: Path, agent_id: str | None
    ) -> Session | None:
        """Load a single session file asynchronously."""
        try:
            data = await aload_json(path)
            if not data:
                return None
            session = Session.from_dict(data)
            if agent_id is None or session.agent_id == agent_id:
                return session
            return None
        except Exception:
            return None

    async def async_delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data (async).

        Uses asyncio.to_thread since shutil.rmtree is CPU-bound.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        import shutil

        session_dir = self._session_dir(session_id)
        if not session_dir.exists():
            return False
        await asyncio.to_thread(shutil.rmtree, session_dir)
        return True

    async def async_save_agent_state(self, session_id: str, state: AgentState) -> None:
        """Save agent state for a session (async).

        Args:
            session_id: Session identifier
            state: AgentState to persist
        """
        self._ensure_directories()
        await asave_json(self._state_file(session_id), state.to_dict())
        await self._async_update_session_timestamp(session_id)

    async def async_load_agent_state(
        self, session_id: str, agent_id: str
    ) -> AgentState:
        """Load agent state for a session (async).

        Args:
            session_id: Session identifier
            agent_id: Agent identifier (used if state doesn't exist)

        Returns:
            AgentState (new if not found)
        """
        state_file = self._state_file(session_id)
        if not state_file.exists():
            return AgentState(agent_id=agent_id)
        data = await aload_json(state_file)
        if data:
            return AgentState.from_dict(data)
        return AgentState(agent_id=agent_id)

    async def async_save_conversation(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> None:
        """Save conversation history for a session (async).

        Args:
            session_id: Session identifier
            messages: List of message dictionaries
        """
        self._ensure_directories()
        await asave_json(self._conversation_file(session_id), {"messages": messages})

    async def async_load_conversation(self, session_id: str) -> list[dict[str, Any]]:
        """Load conversation history for a session (async).

        Args:
            session_id: Session identifier

        Returns:
            List of message dictionaries
        """
        conversation_file = self._conversation_file(session_id)
        if not conversation_file.exists():
            return []
        data = await aload_json(conversation_file)
        return data.get("messages", [])

    async def _async_update_session_timestamp(self, session_id: str) -> None:
        """Update the session's updated_at timestamp (async)."""
        session = await self.async_get_session(session_id)
        if session:
            session.updated_at = get_timestamp()
            await asave_json(self._session_file(session_id), session.to_dict())

    async def async_update_session_title(self, session_id: str, title: str) -> None:
        """Update the session's title.

        Args:
            session_id: Session identifier
            title: New title for the session
        """
        session = await self.async_get_session(session_id)
        if session:
            session.title = title
            session.updated_at = get_timestamp()
            await asave_json(self._session_file(session_id), session.to_dict())
