"""
Context Aggregator module for aggregating sub-agent checkpoint contexts.

Aggregates checkpoint.md files from sub-agents into a unified context file
in the main session's executors directory.
"""

from pathlib import Path

from kader.memory.types import get_default_memory_dir
from kader.providers.base import Message
from kader.providers.ollama import OllamaProvider

AGGREGATOR_SYSTEM_PROMPT = """You are an assistant that aggregates and merges checkpoint summaries from multiple sub-agents.
Given checkpoint summaries from different sub-agents, create a unified summary that combines all information.

Your merged summary MUST include the following sections:

## Directory Structure
Merge all directory structures from sub-agents into a unified tree.
Use a tree-like format:
```
project/
├── src/
│   └── main.py
└── README.md
```

## Actions Performed
Numbered list of all actions taken by all sub-agents in chronological order:
1. Action description
2. Action description

If a section has no relevant content, write "None" under that section.

IMPORTANT: Remove duplicates and merge similar items. Keep the summary organized and clean.
"""


class ContextAggregator:
    """
    Aggregates checkpoint contexts from sub-agents.

    Reads checkpoint.md files from sub-agent directories and merges them
    into a unified checkpoint.md in the executors directory.

    Example:
        aggregator = ContextAggregator(session_id="main-session-id")
        md_path = aggregator.aggregate("/path/to/subagent/checkpoint.md")
        print(f"Aggregated checkpoint saved to: {md_path}")
    """

    def __init__(
        self,
        session_id: str,
        model: str = "gpt-oss:120b-cloud",
        host: str | None = None,
    ) -> None:
        """
        Initialize the ContextAggregator.

        Args:
            session_id: The main session ID for the executors directory
            model: Ollama model identifier (default: "gpt-oss:120b-cloud")
            host: Optional Ollama server host
        """
        self._session_id = session_id
        self._provider = OllamaProvider(model=model, host=host)

    def _get_executors_dir(self) -> Path:
        """
        Get the executors directory path for the main session.

        Returns:
            Path to ~/.kader/memory/sessions/<session-id>/executors/
        """
        return get_default_memory_dir() / "sessions" / self._session_id / "executors"

    def _get_aggregated_checkpoint_path(self) -> Path:
        """
        Get the path for the aggregated checkpoint file.

        Returns:
            Path to the aggregated checkpoint.md in executors directory
        """
        return self._get_executors_dir() / "checkpoint.md"

    def _load_existing_aggregated(self) -> str | None:
        """
        Load existing aggregated checkpoint if it exists.

        Returns:
            Content of existing aggregated checkpoint, or None if not exists
        """
        checkpoint_path = self._get_aggregated_checkpoint_path()
        if checkpoint_path.exists():
            try:
                return checkpoint_path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _load_subagent_checkpoint(self, checkpoint_path: str | Path) -> str | None:
        """
        Load a sub-agent's checkpoint content.

        Args:
            checkpoint_path: Path to the sub-agent's checkpoint.md file

        Returns:
            Content of the checkpoint file, or None if not found
        """
        checkpoint_path = self._get_executors_dir() / checkpoint_path
        path = Path(checkpoint_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    def _merge_checkpoints(
        self,
        existing_aggregated: str | None,
        new_checkpoint: str,
        subagent_name: str | None = None,
    ) -> str:
        """
        Merge a new sub-agent checkpoint into the existing aggregated checkpoint.

        Args:
            existing_aggregated: Content of existing aggregated checkpoint
            new_checkpoint: Content of the new sub-agent checkpoint
            subagent_name: Optional name of the sub-agent for labeling

        Returns:
            Merged checkpoint content
        """
        # If no existing aggregated checkpoint, use the new checkpoint directly
        if not existing_aggregated:
            return new_checkpoint

        user_prompt = f"""Here is the existing aggregated checkpoint from previous sub-agents:

---
{existing_aggregated}
---

Here is the new checkpoint from sub-agent{f' "{subagent_name}"' if subagent_name else ""}:

---
{new_checkpoint}
---

Merge the new checkpoint into the existing aggregated checkpoint.
Combine items into the appropriate sections, remove duplicates, and keep everything organized."""

        messages = [
            Message.system(AGGREGATOR_SYSTEM_PROMPT),
            Message.user(user_prompt),
        ]

        response = self._provider.invoke(messages)
        return response.content

    async def _amerge_checkpoints(
        self,
        existing_aggregated: str | None,
        new_checkpoint: str,
        subagent_name: str | None = None,
    ) -> str:
        """
        Merge a new sub-agent checkpoint into the existing aggregated checkpoint (async).

        Args:
            existing_aggregated: Content of existing aggregated checkpoint
            new_checkpoint: Content of the new sub-agent checkpoint
            subagent_name: Optional name of the sub-agent for labeling

        Returns:
            Merged checkpoint content
        """
        # If no existing aggregated checkpoint, use the new checkpoint directly
        if not existing_aggregated:
            return new_checkpoint

        user_prompt = f"""Here is the existing aggregated checkpoint from previous sub-agents:

---
{existing_aggregated}
---

Here is the new checkpoint from sub-agent{f' "{subagent_name}"' if subagent_name else ""}:

---
{new_checkpoint}
---

Merge the new checkpoint into the existing aggregated checkpoint.
Combine items into the appropriate sections, remove duplicates, and keep everything organized."""

        messages = [
            Message.system(AGGREGATOR_SYSTEM_PROMPT),
            Message.user(user_prompt),
        ]

        response = await self._provider.ainvoke(messages)
        return response.content

    def aggregate(
        self, subagent_checkpoint_path: str, subagent_name: str | None = None
    ) -> str:
        """
        Aggregate a sub-agent's checkpoint into the main executors checkpoint.

        If checkpoint.md exists in the executors directory, it will be updated
        with the new sub-agent's checkpoint. Otherwise, a new aggregated file is created.

        Args:
            subagent_checkpoint_path: Path to the sub-agent's checkpoint.md file
            subagent_name: Optional name of the sub-agent for labeling in the summary

        Returns:
            Absolute path to the aggregated checkpoint file

        Raises:
            FileNotFoundError: If the sub-agent checkpoint file doesn't exist
            ValueError: If the sub-agent checkpoint is empty
        """
        # Load the sub-agent's checkpoint
        new_checkpoint = self._load_subagent_checkpoint(subagent_checkpoint_path)
        if not new_checkpoint:
            raise FileNotFoundError(
                f"Sub-agent checkpoint not found: {subagent_checkpoint_path}"
            )

        if not new_checkpoint.strip():
            raise ValueError(
                f"Sub-agent checkpoint is empty: {subagent_checkpoint_path}"
            )

        # Load existing aggregated checkpoint if it exists
        existing_aggregated = self._load_existing_aggregated()

        # Merge the checkpoints
        merged_content = self._merge_checkpoints(
            existing_aggregated, new_checkpoint, subagent_name
        )

        # Ensure executors directory exists
        aggregated_path = self._get_aggregated_checkpoint_path()
        aggregated_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the aggregated checkpoint
        aggregated_path.write_text(merged_content, encoding="utf-8")

        return str(aggregated_path)

    async def aaggregate(
        self, subagent_checkpoint_path: str, subagent_name: str | None = None
    ) -> str:
        """
        Aggregate a sub-agent's checkpoint into the main executors checkpoint (async).

        If checkpoint.md exists in the executors directory, it will be updated
        with the new sub-agent's checkpoint. Otherwise, a new aggregated file is created.

        Args:
            subagent_checkpoint_path: Path to the sub-agent's checkpoint.md file
            subagent_name: Optional name of the sub-agent for labeling in the summary

        Returns:
            Absolute path to the aggregated checkpoint file

        Raises:
            FileNotFoundError: If the sub-agent checkpoint file doesn't exist
            ValueError: If the sub-agent checkpoint is empty
        """
        # Load the sub-agent's checkpoint
        new_checkpoint = self._load_subagent_checkpoint(subagent_checkpoint_path)
        if not new_checkpoint:
            raise FileNotFoundError(
                f"Sub-agent checkpoint not found: {subagent_checkpoint_path}"
            )

        if not new_checkpoint.strip():
            raise ValueError(
                f"Sub-agent checkpoint is empty: {subagent_checkpoint_path}"
            )

        # Load existing aggregated checkpoint if it exists
        existing_aggregated = self._load_existing_aggregated()

        # Merge the checkpoints
        merged_content = await self._amerge_checkpoints(
            existing_aggregated, new_checkpoint, subagent_name
        )

        # Ensure executors directory exists
        aggregated_path = self._get_aggregated_checkpoint_path()
        aggregated_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the aggregated checkpoint
        aggregated_path.write_text(merged_content, encoding="utf-8")

        return str(aggregated_path)
