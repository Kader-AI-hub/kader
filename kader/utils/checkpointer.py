"""
Checkpointer module for generating step-by-step summaries of agent memory.

Uses an LLM provider to analyze conversation history and produce
human-readable markdown summaries.
"""

from pathlib import Path
from typing import Any

from kader.memory.types import (
    aload_json,
    aread_text,
    awrite_text,
    get_default_memory_dir,
    load_json,
)
from kader.providers.base import BaseLLMProvider, Message

CHECKPOINT_SYSTEM_PROMPT = """You are an assistant that summarizes agent conversation histories.
Given a conversation between a user and an AI agent, create a structured summary in markdown format.

Your summary MUST include the following sections:

## Directory Structure
List the directory structure of any files/folders created or modified during the conversation.
Use a tree-like format:
```
project/
├── src/
│   └── main.py
└── README.md
```

## Actions Performed
Summarize the main accomplishments and significant actions taken by the agent.
Focus on high-level outcomes, not individual steps. For example:
- "Implemented user authentication module with login/logout functionality"
- "Fixed database connection issues and added retry logic"
- "Created REST API endpoints for user management"

Do NOT list every single action (like reading files, running commands, etc.).
Only mention the meaningful outcomes and key decisions.

If a section has no relevant content, write "None" under that section.
"""


class Checkpointer:
    """
    Generates step-by-step markdown summaries of agent memory.

    Uses an LLM provider to analyze conversation history from memory files
    and produce human-readable checkpoint summaries.

    Example:
        checkpointer = Checkpointer(provider=llm_provider)
        md_path = checkpointer.generate_checkpoint("session-id/conversation.json")
        print(f"Checkpoint saved to: {md_path}")
    """

    def __init__(self, provider: BaseLLMProvider) -> None:
        """
        Initialize the Checkpointer.

        Args:
            provider: LLM provider to use for generating summaries
        """
        self._provider = provider

    def _load_memory(self, memory_path: Path) -> dict[str, Any]:
        """
        Load memory JSON from the specified path.

        Args:
            memory_path: Absolute path to the memory JSON file

        Returns:
            Dictionary containing the memory data

        Raises:
            FileNotFoundError: If the memory file doesn't exist
        """
        if not memory_path.exists():
            raise FileNotFoundError(f"Memory file not found: {memory_path}")

        return load_json(memory_path)

    def _extract_messages(self, memory_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract messages from memory data.

        Args:
            memory_data: Dictionary containing memory data

        Returns:
            List of message dictionaries
        """
        # Handle different memory formats
        if "messages" in memory_data:
            # Standard conversation format
            messages = memory_data["messages"]
            # Extract inner message if wrapped in ConversationMessage format
            return [
                msg.get("message", msg) if isinstance(msg, dict) else msg
                for msg in messages
            ]
        elif "conversation" in memory_data:
            # Alternative format
            return memory_data["conversation"]
        else:
            # Return empty if no known format
            return []

    def _format_conversation_for_prompt(self, messages: list[dict[str, Any]]) -> str:
        """
        Format messages into a readable string for the LLM prompt.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted string representation of the conversation
        """
        lines = []
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")

            # Handle tool calls
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                lines.append(f"[{i}] {role}: (calling tools)")
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "unknown")
                    args = func.get("arguments", {})
                    lines.append(f"     -> Tool: {name}")
                    lines.append(f"        Args: {args}")
            elif content:
                # Truncate very long content
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                lines.append(f"[{i}] {role}: {content}")

            # Handle tool call ID (tool results)
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id:
                lines.append(f"     (tool result for: {tool_call_id})")

        return "\n".join(lines)

    def _generate_summary(
        self, conversation_text: str, existing_checkpoint: str | None = None
    ) -> str:
        """
        Generate a step-by-step summary using the LLM (synchronous).

        Args:
            conversation_text: Formatted conversation text
            existing_checkpoint: Existing checkpoint content to update, if any

        Returns:
            Markdown summary of the conversation
        """
        if existing_checkpoint:
            user_prompt = f"""Here is the existing checkpoint from previous iterations:

---
{existing_checkpoint}
---

Here is the new conversation to incorporate:

---
{conversation_text}
---

Update the existing checkpoint by incorporating the new information. Merge new items into the existing sections.
Keep all previously documented content and add new content from this iteration."""
        else:
            user_prompt = f"""Please analyze this agent conversation and create a checkpoint summary:

---
{conversation_text}
---

Create a structured summary following the format specified."""

        messages = [
            Message.system(CHECKPOINT_SYSTEM_PROMPT),
            Message.user(user_prompt),
        ]

        response = self._provider.invoke(messages)
        return response.content

    async def _agenerate_summary(
        self, conversation_text: str, existing_checkpoint: str | None = None
    ) -> str:
        """
        Generate a step-by-step summary using the LLM (asynchronous).

        Args:
            conversation_text: Formatted conversation text
            existing_checkpoint: Existing checkpoint content to update, if any

        Returns:
            Markdown summary of the conversation
        """
        if existing_checkpoint:
            user_prompt = f"""Here is the existing checkpoint from previous iterations:

---
{existing_checkpoint}
---

Here is the new conversation to incorporate:

---
{conversation_text}
---

Update the existing checkpoint by incorporating the new information. Merge new items into the existing sections.
Keep all previously documented content and add new content from this iteration."""
        else:
            user_prompt = f"""Please analyze this agent conversation and create a checkpoint summary:

---
{conversation_text}
---

Create a structured summary following the format specified."""

        messages = [
            Message.system(CHECKPOINT_SYSTEM_PROMPT),
            Message.user(user_prompt),
        ]

        response = await self._provider.ainvoke(messages)
        return response.content

    def _load_existing_checkpoint(self, checkpoint_path: Path) -> str | None:
        """
        Load existing checkpoint content if it exists.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Checkpoint content if exists, None otherwise
        """
        if checkpoint_path.exists():
            try:
                return checkpoint_path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _aload_existing_checkpoint(self, checkpoint_path: Path) -> str | None:
        """
        Asynchronously load existing checkpoint content if it exists.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Checkpoint content if exists, None otherwise
        """
        if checkpoint_path.exists():
            try:
                return await aread_text(checkpoint_path)
            except Exception:
                return None
        return None

    def generate_checkpoint(self, memory_path: str) -> str:
        """
        Generate a checkpoint markdown file from an agent's memory (synchronous).

        If a checkpoint already exists, it will be updated instead of overwritten.

        Args:
            memory_path: Relative path within ~/.kader/memory/sessions/
                         (e.g., "session-id/conversation.json")
                         Or absolute path to the memory JSON file.

        Returns:
            Absolute path to the generated markdown file

        Raises:
            FileNotFoundError: If the memory file doesn't exist
            ValueError: If no messages found in memory
        """
        # Resolve path
        path = Path(memory_path)
        if not path.is_absolute():
            base_dir = get_default_memory_dir() / "sessions"
            path = base_dir / memory_path

        # Load and parse memory
        memory_data = self._load_memory(path)
        messages = self._extract_messages(memory_data)

        if not messages:
            raise ValueError(f"No messages found in memory file: {path}")

        # Check for existing checkpoint
        checkpoint_path = path.parent / "checkpoint.md"
        existing_checkpoint = self._load_existing_checkpoint(checkpoint_path)

        # Format and generate summary
        conversation_text = self._format_conversation_for_prompt(messages)
        summary = self._generate_summary(conversation_text, existing_checkpoint)

        # Save checkpoint markdown
        checkpoint_path.write_text(summary, encoding="utf-8")

        return str(checkpoint_path)

    async def agenerate_checkpoint(self, memory_path: str) -> str:
        """
        Generate a checkpoint markdown file from an agent's memory (asynchronous).

        If a checkpoint already exists, it will be updated instead of overwritten.

        Args:
            memory_path: Relative path within ~/.kader/memory/sessions/
                         (e.g., "session-id/conversation.json")
                         Or absolute path to the memory JSON file.

        Returns:
            Absolute path to the generated markdown file

        Raises:
            FileNotFoundError: If the memory file doesn't exist
            ValueError: If no messages found in memory
        """
        # Resolve path
        path = Path(memory_path)
        if not path.is_absolute():
            base_dir = get_default_memory_dir() / "sessions"
            path = base_dir / memory_path

        # Load and parse memory (async)
        if not path.exists():
            raise FileNotFoundError(f"Memory file not found: {path}")
        memory_data = await aload_json(path)
        messages = self._extract_messages(memory_data)

        if not messages:
            raise ValueError(f"No messages found in memory file: {path}")

        # Check for existing checkpoint (async)
        checkpoint_path = path.parent / "checkpoint.md"
        existing_checkpoint = await self._aload_existing_checkpoint(checkpoint_path)

        # Format and generate summary
        conversation_text = self._format_conversation_for_prompt(messages)
        summary = await self._agenerate_summary(conversation_text, existing_checkpoint)

        # Save checkpoint markdown (async)
        await awrite_text(checkpoint_path, summary)

        return str(checkpoint_path)
