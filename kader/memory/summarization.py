"""
Hierarchical Conversation Manager with Summarization.

Provides a conversation manager that keeps recent messages verbatim
and summarizes older ones into structured context notes. Summaries
are built primarily through extraction from tool calls and results,
minimizing LLM usage to avoid hallucination.

Key design principles:
- Extract files_modified, errors_encountered from actual tool call data
- Use LLM only for natural language summary_text and current_state
- Fall back gracefully when no LLM provider is available
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .conversation import ConversationMessage, SlidingWindowConversationManager
from .types import get_timestamp


@dataclass
class ConversationSummary:
    """Represents a structured summary of conversation history.

    Fields are populated primarily through extraction from tool calls
    and message content, rather than LLM generation, to avoid hallucination.

    Attributes:
        summary_id: Unique identifier for this summary
        original_message_count: Number of messages that were summarized
        summary_text: Brief natural-language overview (LLM-generated or fallback)
        key_decisions: Important choices extracted from user messages
        files_modified: File paths extracted from write_file/edit_file tool calls
        errors_encountered: Error strings extracted from tool results
        current_state: Current progress description (LLM-generated or fallback)
        pending_tasks: Outstanding tasks mentioned in conversation
        timestamp: ISO timestamp when summary was created
    """

    summary_id: str
    original_message_count: int
    summary_text: str
    key_decisions: list[str] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    errors_encountered: list[str] = field(default_factory=list)
    current_state: str = ""
    pending_tasks: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=get_timestamp)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "summary_id": self.summary_id,
            "original_message_count": self.original_message_count,
            "summary_text": self.summary_text,
            "key_decisions": self.key_decisions,
            "files_modified": self.files_modified,
            "errors_encountered": self.errors_encountered,
            "current_state": self.current_state,
            "pending_tasks": self.pending_tasks,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationSummary":
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            New ConversationSummary instance
        """
        return cls(
            summary_id=data.get("summary_id", str(uuid.uuid4())),
            original_message_count=data.get("original_message_count", 0),
            summary_text=data.get("summary_text", ""),
            key_decisions=data.get("key_decisions", []),
            files_modified=data.get("files_modified", []),
            errors_encountered=data.get("errors_encountered", []),
            current_state=data.get("current_state", ""),
            pending_tasks=data.get("pending_tasks", []),
            timestamp=data.get("timestamp", get_timestamp()),
        )


class HierarchicalConversationManager(SlidingWindowConversationManager):
    """Hierarchical conversation manager with summarization.

    Keeps the last N turns in full detail and summarizes older turns
    into structured context notes. Summary fields are populated by
    extracting data from tool calls and results to avoid hallucination.
    The LLM is only used for the natural-language summary_text field.

    Attributes:
        full_context_window: Number of recent message pairs to keep verbatim
        summarization_threshold: Fraction of window_size at which to trigger summarization
        provider: Optional LLM provider for generating summary_text
    """

    # Patterns for extracting error information from tool results
    _ERROR_PATTERNS = re.compile(
        r"(?:error|exception|traceback|fail(?:ed|ure)?|"
        r"errno|permission denied|not found|syntax error)",
        re.IGNORECASE,
    )

    # Tool names that indicate file modification
    _FILE_MODIFY_TOOLS = {"write_file", "edit_file", "create_file"}

    # Tool names that indicate file reading
    _FILE_READ_TOOLS = {"read_file", "view_file", "open_file"}

    def __init__(
        self,
        window_size: int = 20,
        full_context_window: int = 5,
        summarization_threshold: float = 0.8,
        provider: Optional[Any] = None,
    ) -> None:
        """Initialize the hierarchical conversation manager.

        Args:
            window_size: Total message pairs capacity (default: 20)
            full_context_window: Recent pairs to keep verbatim (default: 5)
            summarization_threshold: Fraction of capacity to trigger summarization (default: 0.8)
            provider: Optional LLM provider for generating summary_text.
                      Should implement invoke(messages, config) -> LLMResponse.
        """
        super().__init__(window_size=window_size)
        self.full_context_window = full_context_window
        self.summarization_threshold = summarization_threshold
        self.provider = provider
        self._summaries: list[ConversationSummary] = []
        self._last_summarized_count = 0

    def apply_window(self) -> list[dict[str, Any]]:
        """Apply hierarchical window with summarization.

        Returns messages in format:
        [System message with summary] + [Recent N messages verbatim]

        If the conversation is below the summarization threshold,
        delegates to the parent SlidingWindowConversationManager.

        Returns:
            List of message dictionaries ready for LLM consumption
        """
        if not self._messages:
            return []

        total_pairs = len(self._messages) // 2

        # If under threshold, return all messages (parent behavior)
        if total_pairs <= self.window_size * self.summarization_threshold:
            return [msg.message for msg in self._messages]

        # Split into older (to summarize) and recent (verbatim)
        recent_count = self.full_context_window * 2
        recent_count = min(recent_count, len(self._messages))

        recent_messages = self._messages[-recent_count:]
        older_messages = self._messages[:-recent_count]

        # Generate or retrieve summary for older messages
        if older_messages and len(older_messages) > self._last_summarized_count:
            summary = self._generate_summary(older_messages)
            self._summaries.append(summary)
            self._last_summarized_count = len(older_messages)
        else:
            summary = self._summaries[-1] if self._summaries else None

        # Build result: summary system message + recent messages
        result: list[dict[str, Any]] = []

        if summary:
            summary_msg = {
                "role": "system",
                "content": self._format_summary(summary),
            }
            result.append(summary_msg)

        result.extend(msg.message for msg in recent_messages)

        return result

    def _generate_summary(
        self, messages: list[ConversationMessage]
    ) -> ConversationSummary:
        """Generate a structured summary by extracting data from messages.

        Extracts files_modified, errors_encountered, and key_decisions
        directly from tool calls and message content. Uses LLM only for
        summary_text and current_state fields.

        Args:
            messages: List of older ConversationMessage objects to summarize

        Returns:
            ConversationSummary with extracted and generated fields
        """
        # Extract structured fields from messages
        files_modified = self._extract_files_modified(messages)
        errors_encountered = self._extract_errors(messages)
        key_decisions = self._extract_key_decisions(messages)

        # Generate natural-language summary via LLM (or fallback)
        summary_text, current_state = self._generate_text_summary(messages)

        return ConversationSummary(
            summary_id=str(uuid.uuid4()),
            original_message_count=len(messages),
            summary_text=summary_text,
            key_decisions=key_decisions,
            files_modified=files_modified,
            errors_encountered=errors_encountered,
            current_state=current_state,
        )

    def _extract_files_modified(self, messages: list[ConversationMessage]) -> list[str]:
        """Extract file paths from tool calls that modify files.

        Scans assistant messages with tool_calls for write_file, edit_file,
        etc., and extracts the file_path or path argument.

        Args:
            messages: Conversation messages to scan

        Returns:
            Deduplicated list of modified file paths
        """
        files: set[str] = set()

        for msg in messages:
            if msg.role != "assistant" or not msg.tool_calls:
                continue

            for tool_call in msg.tool_calls:
                fn_info = tool_call.get("function", {})
                if not fn_info and "name" in tool_call:
                    fn_info = tool_call

                tool_name = fn_info.get("name", "")

                if tool_name in self._FILE_MODIFY_TOOLS:
                    raw_args = fn_info.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            continue

                    # Common argument names for file paths
                    for key in ("file_path", "path", "filename", "file"):
                        if key in raw_args:
                            files.add(str(raw_args[key]))
                            break

        return sorted(files)

    def _extract_errors(self, messages: list[ConversationMessage]) -> list[str]:
        """Extract error information from tool result messages.

        Scans tool-role messages for error patterns and extracts
        the first line or a truncated version of the error.

        Args:
            messages: Conversation messages to scan

        Returns:
            Deduplicated list of error descriptions (truncated to 200 chars)
        """
        errors: list[str] = []
        seen: set[str] = set()

        for msg in messages:
            if msg.role != "tool":
                continue

            content = msg.content or ""
            if not self._ERROR_PATTERNS.search(content):
                continue

            # Extract first meaningful error line
            for line in content.split("\n"):
                line = line.strip()
                if line and self._ERROR_PATTERNS.search(line):
                    truncated = line[:200]
                    if truncated not in seen:
                        seen.add(truncated)
                        errors.append(truncated)
                    break

        return errors

    def _extract_key_decisions(self, messages: list[ConversationMessage]) -> list[str]:
        """Extract key decisions from user messages.

        Looks for user messages that contain decision-related keywords
        and extracts a truncated version as a decision note.

        Args:
            messages: Conversation messages to scan

        Returns:
            List of decision descriptions (truncated to 150 chars)
        """
        decision_pattern = re.compile(
            r"\b(?:use|choose|go with|let'?s|prefer|switch to|instead of|"
            r"change to|update|don't|should|must|need to)\b",
            re.IGNORECASE,
        )

        decisions: list[str] = []

        for msg in messages:
            if msg.role != "user":
                continue

            content = msg.content or ""
            if len(content) < 10:
                continue

            if decision_pattern.search(content):
                # Take the first 150 characters as a decision note
                truncated = content[:150].strip()
                if truncated:
                    decisions.append(truncated)

        # Cap at 10 most recent decisions
        return decisions[-10:]

    def _generate_text_summary(
        self, messages: list[ConversationMessage]
    ) -> tuple[str, str]:
        """Generate natural-language summary_text and current_state.

        Uses the LLM provider if available to create a brief overview.
        Falls back to a simple message-count-based description.

        Args:
            messages: Conversation messages to summarize

        Returns:
            Tuple of (summary_text, current_state)
        """
        if not self.provider:
            return (
                f"Previous {len(messages)} messages covering task execution.",
                "In progress",
            )

        # Build a compact representation of the conversation for the LLM
        compact_lines: list[str] = []
        for msg in messages[-30:]:  # Limit input to last 30 messages for cost
            role = msg.role
            content = (msg.content or "")[:300]  # Truncate each message
            if content:
                compact_lines.append(f"{role}: {content}")

        conversation_excerpt = "\n".join(compact_lines)

        summary_prompt = (
            "Summarize this agent conversation history in 2-3 sentences. "
            "Also describe the current state of progress in one sentence.\n\n"
            f"Conversation:\n{conversation_excerpt}\n\n"
            "Respond in this exact JSON format:\n"
            '{"summary_text": "...", "current_state": "..."}'
        )

        try:
            from kader.providers.base import Message, ModelConfig

            response = self.provider.invoke(
                [Message.user(summary_prompt)],
                ModelConfig(temperature=0.0, max_tokens=300),
            )

            text = response.content or ""
            # Try to parse JSON from the response
            # Handle potential markdown code blocks
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

            data = json.loads(text)
            return (
                data.get("summary_text", f"Previous {len(messages)} messages."),
                data.get("current_state", "In progress"),
            )
        except Exception:
            # Fallback on any error
            return (
                f"Previous {len(messages)} messages covering task execution.",
                "In progress",
            )

    def _format_summary(self, summary: ConversationSummary) -> str:
        """Format a summary for inclusion as a system message.

        Creates a structured markdown representation of the summary
        that provides the agent with essential context about older
        conversation history.

        Args:
            summary: The ConversationSummary to format

        Returns:
            Formatted string for use as system message content
        """
        sections: list[str] = [
            f"## Previous Context ({summary.original_message_count} messages summarized)",
            f"\n**Overview:** {summary.summary_text}",
        ]

        if summary.key_decisions:
            items = "\n".join(f"- {d}" for d in summary.key_decisions)
            sections.append(f"\n**Key Decisions:**\n{items}")

        if summary.files_modified:
            items = "\n".join(f"- {f}" for f in summary.files_modified)
            sections.append(f"\n**Files Modified:**\n{items}")

        if summary.current_state:
            sections.append(f"\n**Current State:** {summary.current_state}")

        if summary.errors_encountered:
            items = "\n".join(f"- {e}" for e in summary.errors_encountered)
            sections.append(f"\n**Issues Encountered:**\n{items}")

        if summary.pending_tasks:
            items = "\n".join(f"- {t}" for t in summary.pending_tasks)
            sections.append(f"\n**Pending Tasks:**\n{items}")

        return "\n".join(sections)

    def get_state(self) -> dict[str, Any]:
        """Get manager state for persistence, including summaries.

        Returns:
            State dictionary with parent state plus summaries
        """
        state = super().get_state()
        state["full_context_window"] = self.full_context_window
        state["summarization_threshold"] = self.summarization_threshold
        state["summaries"] = [s.to_dict() for s in self._summaries]
        state["last_summarized_count"] = self._last_summarized_count
        return state

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore manager state, including summaries.

        Args:
            state: State dictionary from get_state()
        """
        super().set_state(state)
        self.full_context_window = state.get(
            "full_context_window", self.full_context_window
        )
        self.summarization_threshold = state.get(
            "summarization_threshold", self.summarization_threshold
        )
        self._summaries = [
            ConversationSummary.from_dict(s) for s in state.get("summaries", [])
        ]
        self._last_summarized_count = state.get("last_summarized_count", 0)
