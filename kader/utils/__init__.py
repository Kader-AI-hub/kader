"""
Utility modules for Kader.

Provides shared utility functions and helper modules.
"""

from .checkpointer import Checkpointer
from .context_aggregator import ContextAggregator
from .ignore import filter_by_gitignore, get_gitignore_filter, is_ignored
from .session_title import agenerate_session_title, generate_session_title
from .todo_metadata import TodoMetadataHandler

__all__ = [
    "Checkpointer",
    "ContextAggregator",
    "TodoMetadataHandler",
    "agenerate_session_title",
    "filter_by_gitignore",
    "generate_session_title",
    "get_gitignore_filter",
    "is_ignored",
]
