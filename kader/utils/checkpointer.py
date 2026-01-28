"""
Checkpointer module for generating step-by-step summaries of agent memory.

Uses OllamaProvider to analyze conversation history and produce
human-readable markdown summaries.
"""

from pathlib import Path
from typing import Any

from kader.memory.types import get_default_memory_dir, load_json
from kader.providers.base import Message
from kader.providers.ollama import OllamaProvider


CHECKPOINT_SYSTEM_PROMPT = """You are an assistant that summarizes agent conversation histories.
Given a conversation between a user and an AI agent, create a clear step-by-step summary in markdown format.

Your summary should:
1. Be organized chronologically
2. Highlight key actions taken by the agent
3. Note any tool calls and their results
4. Summarize the user's requests and the agent's responses
5. Use clear, natural language

Format the output as a markdown document with:
- A title summarizing the overall conversation goal
- Numbered steps describing the progression
- Code blocks for any code or commands mentioned
- Bullet points for details within each step
"""


class Checkpointer:
    """
    Generates step-by-step markdown summaries of agent memory.

    Uses OllamaProvider to analyze conversation history from memory files
    and produce human-readable checkpoint summaries.

    Example:
        checkpointer = Checkpointer()
        md_path = checkpointer.generate_checkpoint("session-id/conversation.json")
        print(f"Checkpoint saved to: {md_path}")
    """

    def __init__(
        self,
        model: str = "gpt-oss:120b-cloud",
        host: str | None = None,
    ) -> None:
        """
        Initialize the Checkpointer.

        Args:
            model: Ollama model identifier (default: "gpt-oss:120b-cloud")
            host: Optional Ollama server host
        """
        self._provider = OllamaProvider(model=model, host=host)

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

    def _format_conversation_for_prompt(
        self, messages: list[dict[str, Any]]
    ) -> str:
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

    def _generate_summary(self, conversation_text: str) -> str:
        """
        Generate a step-by-step summary using the LLM.

        Args:
            conversation_text: Formatted conversation text

        Returns:
            Markdown summary of the conversation
        """
        user_prompt = f"""Please analyze this agent conversation and create a step-by-step summary in markdown:

---
{conversation_text}
---

Create a clear, organized summary of what happened in this conversation."""

        messages = [
            Message.system(CHECKPOINT_SYSTEM_PROMPT),
            Message.user(user_prompt),
        ]

        response = self._provider.invoke(messages)
        return response.content

    def generate_checkpoint(self, memory_path: str) -> str:
        """
        Generate a checkpoint markdown file from an agent's memory.

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

        # Format and generate summary
        conversation_text = self._format_conversation_for_prompt(messages)
        summary = self._generate_summary(conversation_text)

        # Save checkpoint markdown
        checkpoint_path = path.parent / "checkpoint.md"
        checkpoint_path.write_text(summary, encoding="utf-8")

        return str(checkpoint_path)
