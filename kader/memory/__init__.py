"""
Kader Memory Module

Provides memory management for agents following the AWS Strands agents SDK hierarchy:
- State Management: AgentState for persistent state, RequestState for request-scoped context
- Session Management: FileSessionManager for filesystem-based persistence
- Conversation Management: SlidingWindowConversationManager for context windowing

Memory is stored locally in $HOME/.kader/memory as directories and JSON files.
"""

# Core types
# Conversation management
from .conversation import (
    ConversationManager,
    ConversationMessage,
    NullConversationManager,
    SlidingWindowConversationManager,
)

# Session management
from .session import (
    FileSessionManager,
    Session,
    SessionManager,
)

# State management
from .state import (
    AgentState,
    RequestState,
)
from .types import (
    MemoryConfig,
    SessionType,
    decode_bytes_values,
    encode_bytes_values,
    get_default_memory_dir,
    get_timestamp,
    load_json,
    save_json,
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
