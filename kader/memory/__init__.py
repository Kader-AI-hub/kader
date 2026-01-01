"""
Kader Memory Module

Provides memory management for agents following the AWS Strands agents SDK hierarchy:
- State Management: AgentState for persistent state, RequestState for request-scoped context
- Session Management: FileSessionManager for filesystem-based persistence
- Conversation Management: SlidingWindowConversationManager for context windowing

Memory is stored locally in $HOME/.kader/memory as directories and JSON files.
"""

# Core types
from .types import (
    SessionType,
    MemoryConfig,
    get_timestamp,
    get_default_memory_dir,
    save_json,
    load_json,
    encode_bytes_values,
    decode_bytes_values,
)

# State management
from .state import (
    AgentState,
    RequestState,
)

# Session management
from .session import (
    Session,
    SessionManager,
    FileSessionManager,
)

# Conversation management
from .conversation import (
    ConversationMessage,
    ConversationManager,
    SlidingWindowConversationManager,
    NullConversationManager,
)


__all__ = [
    # Types
    "SessionType",
    "MemoryConfig",
    "get_timestamp",
    "get_default_memory_dir",
    "save_json",
    "load_json",
    "encode_bytes_values",
    "decode_bytes_values",
    # State
    "AgentState",
    "RequestState",
    # Session
    "Session",
    "SessionManager",
    "FileSessionManager",
    # Conversation
    "ConversationMessage",
    "ConversationManager",
    "SlidingWindowConversationManager",
    "NullConversationManager",
]
