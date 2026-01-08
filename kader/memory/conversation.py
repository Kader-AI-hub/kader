"""
Conversation management for agents.

Provides conversation history management and context windowing
to handle token limits and maintain coherent conversations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .types import get_timestamp

# Import Message from providers for compatibility
# This allows the conversation module to work with the existing Message type
try:
    from kader.providers.base import Message
except ImportError:
    # Fallback for standalone usage
    Message = None  # type: ignore


@dataclass
class ConversationMessage:
    """Wrapper around a Message with metadata.

    Provides additional metadata for conversation management
    while maintaining compatibility with the provider's Message type.

    Attributes:
        message: Dictionary representation of the underlying message
        message_id: Index/position in the conversation
        created_at: ISO timestamp when message was added
        updated_at: ISO timestamp when message was last updated
    """

    message: dict[str, Any]
    message_id: int
    created_at: str = field(default_factory=get_timestamp)
    updated_at: str = field(default_factory=get_timestamp)

    @property
    def role(self) -> str:
        """Get the message role."""
        return self.message.get("role", "")

    @property
    def content(self) -> str:
        """Get the message content."""
        return self.message.get("content", "")

    @property
    def tool_calls(self) -> list[dict[str, Any]] | None:
        """Get tool calls if present."""
        return self.message.get("tool_calls")

    @property
    def tool_call_id(self) -> str | None:
        """Get tool call ID if this is a tool response."""
        return self.message.get("tool_call_id")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation
        """
        return {
            "message": self.message,
            "message_id": self.message_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationMessage":
        """Create from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New ConversationMessage instance
        """
        return cls(
            message=data.get("message", {}),
            message_id=data.get("message_id", 0),
            created_at=data.get("created_at", get_timestamp()),
            updated_at=data.get("updated_at", get_timestamp()),
        )

    @classmethod
    def from_message(cls, message: Any, index: int) -> "ConversationMessage":
        """Create from a provider Message object.

        Args:
            message: Message object (from kader.providers.base)
            index: Position in conversation

        Returns:
            New ConversationMessage instance
        """
        # Handle both Message objects and dicts
        if hasattr(message, "to_dict"):
            msg_dict = message.to_dict()
        elif isinstance(message, dict):
            msg_dict = message
        else:
            msg_dict = {"role": "user", "content": str(message)}

        return cls(
            message=msg_dict,
            message_id=index,
        )

    def to_message(self) -> Any:
        """Convert back to a provider Message object.

        Returns:
            Message object if kader.providers.base is available, else dict
        """
        if Message is not None:
            return Message(
                role=self.message.get("role", "user"),
                content=self.message.get("content", ""),
                name=self.message.get("name"),
                tool_call_id=self.message.get("tool_call_id"),
                tool_calls=self.message.get("tool_calls"),
            )
        return self.message


class ConversationManager(ABC):
    """Abstract base class for conversation management.

    Provides interface for managing conversation history and
    applying context management strategies.
    """

    @abstractmethod
    def add_message(self, message: Any) -> ConversationMessage:
        """Add a message to the conversation.

        Args:
            message: Message to add (Message object or dict)

        Returns:
            The wrapped ConversationMessage
        """
        ...

    @abstractmethod
    def add_messages(self, messages: list[Any]) -> list[ConversationMessage]:
        """Add multiple messages to the conversation.

        Args:
            messages: List of messages to add

        Returns:
            List of wrapped ConversationMessages
        """
        ...

    @abstractmethod
    def get_messages(self) -> list[ConversationMessage]:
        """Get all conversation messages.

        Returns:
            List of ConversationMessages
        """
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get manager state for persistence.

        Returns:
            State dictionary
        """
        ...

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        """Restore manager state.

        Args:
            state: State dictionary
        """
        ...

    @abstractmethod
    def apply_window(self) -> list[dict[str, Any]]:
        """Apply context management strategy and return messages.

        Returns:
            List of message dictionaries ready for LLM consumption
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all messages from the conversation."""
        ...


class SlidingWindowConversationManager(ConversationManager):
    """Sliding window conversation manager.

    Maintains a fixed number of recent message pairs to prevent
    exceeding model context limits. Preserves tool call/result pairs.

    Attributes:
        window_size: Maximum number of message pairs to keep
        messages: List of conversation messages
    """

    def __init__(self, window_size: int = 20) -> None:
        """Initialize the sliding window manager.

        Args:
            window_size: Maximum message pairs to keep (default: 20)
        """
        self.window_size = window_size
        self._messages: list[ConversationMessage] = []
        self._next_id = 0

    def add_message(self, message: Any) -> ConversationMessage:
        """Add a message to the conversation.

        Args:
            message: Message to add (Message object or dict)

        Returns:
            The wrapped ConversationMessage
        """
        conv_msg = ConversationMessage.from_message(message, self._next_id)
        self._messages.append(conv_msg)
        self._next_id += 1
        return conv_msg

    def add_messages(self, messages: list[Any]) -> list[ConversationMessage]:
        """Add multiple messages to the conversation.

        Args:
            messages: List of messages to add

        Returns:
            List of wrapped ConversationMessages
        """
        return [self.add_message(msg) for msg in messages]

    def get_messages(self) -> list[ConversationMessage]:
        """Get all conversation messages.

        Returns:
            List of ConversationMessages
        """
        return list(self._messages)

    def get_state(self) -> dict[str, Any]:
        """Get manager state for persistence.

        Returns:
            State dictionary
        """
        return {
            "window_size": self.window_size,
            "next_id": self._next_id,
            "messages": [msg.to_dict() for msg in self._messages],
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore manager state.

        Args:
            state: State dictionary
        """
        self.window_size = state.get("window_size", self.window_size)
        self._next_id = state.get("next_id", 0)
        self._messages = [
            ConversationMessage.from_dict(msg_data)
            for msg_data in state.get("messages", [])
        ]

    def apply_window(self) -> list[dict[str, Any]]:
        """Apply sliding window and return messages for LLM.

        Keeps the most recent messages within the window size,
        while preserving complete tool call/result pairs.

        Returns:
            List of message dictionaries
        """
        if not self._messages:
            return []

        # Calculate how many messages to keep (pairs = user + assistant)
        max_messages = self.window_size * 2

        if len(self._messages) <= max_messages:
            return [msg.message for msg in self._messages]

        # Start from the end and work backwards
        # Ensure we don't break tool call/result pairs
        messages_to_keep = self._messages[-max_messages:]

        # Check if first message is a tool result without its call
        # If so, find and include the tool call
        result = []
        tool_call_ids_needed: set[str] = set()

        for msg in reversed(messages_to_keep):
            if msg.role == "tool" and msg.tool_call_id:
                tool_call_ids_needed.add(msg.tool_call_id)

        # Include any assistant messages with tool calls that are needed
        for msg in reversed(self._messages):
            if msg in messages_to_keep:
                result.insert(0, msg.message)
            elif msg.role == "assistant" and msg.tool_calls:
                # Check if any of the tool calls are needed
                for tc in msg.tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id in tool_call_ids_needed:
                        result.insert(0, msg.message)
                        tool_call_ids_needed.discard(tc_id)
                        break

            if not tool_call_ids_needed:
                break

        return result

    def clear(self) -> None:
        """Clear all messages from the conversation."""
        self._messages.clear()
        self._next_id = 0

    def __len__(self) -> int:
        """Return number of messages."""
        return len(self._messages)


class NullConversationManager(ConversationManager):
    """No-op conversation manager.

    For short interactions or when managing context manually.
    Does not store any messages.
    """

    def add_message(self, message: Any) -> ConversationMessage:
        """Add a message (no-op, returns wrapper).

        Args:
            message: Message to add

        Returns:
            The wrapped ConversationMessage (not stored)
        """
        return ConversationMessage.from_message(message, 0)

    def add_messages(self, messages: list[Any]) -> list[ConversationMessage]:
        """Add multiple messages (no-op).

        Args:
            messages: List of messages to add

        Returns:
            List of wrapped ConversationMessages (not stored)
        """
        return [
            ConversationMessage.from_message(msg, i) for i, msg in enumerate(messages)
        ]

    def get_messages(self) -> list[ConversationMessage]:
        """Get all messages (always empty).

        Returns:
            Empty list
        """
        return []

    def get_state(self) -> dict[str, Any]:
        """Get manager state (empty).

        Returns:
            Empty dict
        """
        return {}

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore manager state (no-op).

        Args:
            state: State dictionary (ignored)
        """
        pass

    def apply_window(self) -> list[dict[str, Any]]:
        """Apply context management (returns empty).

        Returns:
            Empty list
        """
        return []

    def clear(self) -> None:
        """Clear messages (no-op)."""
        pass
