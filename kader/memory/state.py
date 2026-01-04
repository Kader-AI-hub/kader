"""
State management for agents.

Provides key-value state storage for persisting agent state
that exists outside of conversation context.
"""

from dataclasses import dataclass, field
from typing import Any

from .types import get_timestamp


@dataclass
class AgentState:
    """Key-value store for persistent agent state.

    AgentState is used for storing stateful information that exists
    outside the direct conversation context. Unlike conversation history,
    agent state is not directly passed to the language model during inference
    but can be accessed and modified by the agent's tools and application logic.

    Attributes:
        agent_id: Unique identifier for the agent
        _state: Internal state dictionary
        created_at: ISO timestamp when state was created
        updated_at: ISO timestamp when state was last updated
    """

    agent_id: str
    _state: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=get_timestamp)
    updated_at: str = field(default_factory=get_timestamp)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state.

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The value or default
        """
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state.

        Args:
            key: The key to set
            value: The value to store
        """
        self._state[key] = value
        self.updated_at = get_timestamp()

    def delete(self, key: str) -> bool:
        """Delete a key from state.

        Args:
            key: The key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key in self._state:
            del self._state[key]
            self.updated_at = get_timestamp()
            return True
        return False

    def get_all(self) -> dict[str, Any]:
        """Get all state as a dictionary.

        Returns:
            Copy of the state dictionary
        """
        return dict(self._state)

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()
        self.updated_at = get_timestamp()

    def update(self, data: dict[str, Any]) -> None:
        """Update state with multiple key-value pairs.

        Args:
            data: Dictionary of key-value pairs to update
        """
        self._state.update(data)
        self.updated_at = get_timestamp()

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._state

    def __len__(self) -> int:
        """Return number of keys in state."""
        return len(self._state)

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization.

        Returns:
            Dictionary representation of the state
        """
        return {
            "agent_id": self.agent_id,
            "state": self._state,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentState":
        """Create AgentState from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New AgentState instance
        """
        state = cls(
            agent_id=data.get("agent_id", ""),
            created_at=data.get("created_at", get_timestamp()),
            updated_at=data.get("updated_at", get_timestamp()),
        )
        state._state = data.get("state", {})
        return state


@dataclass
class RequestState:
    """Request-scoped ephemeral state.

    RequestState is used for storing context that is maintained
    specifically within the scope of a single request. This state
    is NOT persisted to disk.

    Attributes:
        request_id: Unique identifier for the request
        _state: Internal state dictionary
    """

    request_id: str
    _state: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state.

        Args:
            key: The key to retrieve
            default: Default value if key not found

        Returns:
            The value or default
        """
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in state.

        Args:
            key: The key to set
            value: The value to store
        """
        self._state[key] = value

    def delete(self, key: str) -> bool:
        """Delete a key from state.

        Args:
            key: The key to delete

        Returns:
            True if key existed and was deleted, False otherwise
        """
        if key in self._state:
            del self._state[key]
            return True
        return False

    def get_all(self) -> dict[str, Any]:
        """Get all state as a dictionary.

        Returns:
            Copy of the state dictionary
        """
        return dict(self._state)

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()

    def update(self, data: dict[str, Any]) -> None:
        """Update state with multiple key-value pairs.

        Args:
            data: Dictionary of key-value pairs to update
        """
        self._state.update(data)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in state."""
        return key in self._state

    def __len__(self) -> int:
        """Return number of keys in state."""
        return len(self._state)
