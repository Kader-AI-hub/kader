"""
Agent Tool - Use a ReActAgent as a callable tool.

Allows spawning sub-agents to execute specific tasks with isolated memory contexts.
"""

import uuid
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from kader.memory import (
    HierarchicalConversationManager,
    SlidingWindowConversationManager,
)
from kader.memory.compression import CompressionConfig, ToolOutputCompressor
from kader.memory.types import aread_text, save_json
from kader.prompts import ExecutorAgentPrompt
from kader.providers.base import BaseLLMProvider, Message
from kader.utils import Checkpointer, ContextAggregator

from .base import BaseTool, ParameterSchema, ToolCategory


class PersistentSlidingWindowConversationManager(SlidingWindowConversationManager):
    """
    SlidingWindowConversationManager with JSON persistence.

    Saves the entire message history (dict format) to a JSON file
    after every add_message(s) call.
    """

    def __init__(self, file_path: Path, window_size: int = 20) -> None:
        """
        Initialize with a file path for persistence.
        """
        super().__init__(window_size=window_size)
        self.file_path = file_path

    def _save(self) -> None:
        """Save entire history to JSON."""
        try:
            # We want to save plain dicts
            messages_dicts = [msg.message for msg in self._messages]
            data = {"messages": messages_dicts}
            # Ensure parent temp-directory exists is done by caller usually,
            # but best effort here:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.file_path, data)
        except Exception:
            # Best effort save
            pass

    def add_message(self, message: Any) -> Any:
        # Call super
        result = super().add_message(message)
        # Save
        self._save()
        return result

    def add_messages(self, messages: list[Any]) -> list[Any]:
        # Call super
        result = super().add_messages(messages)
        # Save
        self._save()
        return result


class PersistentHierarchicalConversationManager(HierarchicalConversationManager):
    """
    HierarchicalConversationManager with JSON persistence.

    Saves the entire message history (dict format) to a JSON file
    after every add_message(s) call.
    """

    def __init__(
        self,
        file_path: Path,
        window_size: int = 20,
        full_context_window: int = 5,
        provider: Any = None,
    ) -> None:
        """
        Initialize with a file path for persistence.
        """
        super().__init__(
            window_size=window_size,
            full_context_window=full_context_window,
            provider=provider,
        )
        self.file_path = file_path

    def _save(self) -> None:
        """Save entire history to JSON."""
        try:
            messages_dicts = [msg.message for msg in self._messages]
            data = {"messages": messages_dicts}
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(self.file_path, data)
        except Exception:
            pass

    def add_message(self, message: Any) -> Any:
        result = super().add_message(message)
        self._save()
        return result

    def add_messages(self, messages: list[Any]) -> list[Any]:
        result = super().add_messages(messages)
        self._save()
        return result


class AgentTool(BaseTool[str]):
    """
    Tool that spawns a ReActAgent to execute a specific task.

    Creates an agent with its own memory context and default tools
    (filesystem, web, command executor) to complete the given task.

    When `interrupt_before_tool=True`, the agent will pause before each tool
    execution and use the `tool_confirmation_callback` to get user confirmation.

    Example:
        # Autonomous execution (no interrupts)
        tool = AgentTool(name="research_agent", interrupt_before_tool=False)
        result = tool.execute(task="Find the current stock price of AAPL")

        # Interactive execution with tool confirmation
        def my_callback(tool_info: str) -> Tuple[bool, Optional[str]]:
            user_input = input(f"Execute {tool_info}? [y/n]: ")
            return (user_input.lower() == 'y', None)

        tool = AgentTool(
            name="research_agent",
            interrupt_before_tool=True,
            tool_confirmation_callback=my_callback
        )
        result = tool.execute(task="Find info about topic X")
    """

    def __init__(
        self,
        name: str,
        description: str = "Execute a specific task using an AI agent",
        provider: Optional[BaseLLMProvider] = None,
        model_name: str = "qwen3-coder:480b-cloud",
        interrupt_before_tool: bool = True,
        tool_confirmation_callback: Optional[
            Callable[..., Tuple[bool, Optional[str]]]
        ] = None,
        direct_execution_callback: Optional[Callable[..., None]] = None,
        tool_execution_result_callback: Optional[Callable[..., None]] = None,
        enable_compression: bool = True,
        compression_config: Optional[CompressionConfig] = None,
        memory_manager_type: str = "sliding_window",
    ) -> None:
        """
        Initialize the AgentTool.

        Args:
            name: Name of the agent tool (used as identifier).
            description: Description of what this agent does.
            provider: Optional LLM provider (uses OllamaProvider by default).
            model_name: Model to use for the agent.
            interrupt_before_tool: If True, pause before tool execution for user
                confirmation. The task will only complete when the agent returns
                its final response.
            tool_confirmation_callback: Callback function for tool confirmation.
                Should return (should_execute: bool, additional_context: Optional[str]).
                If not provided and interrupt_before_tool=True, uses stdin prompts.
            enable_compression: If True, compresses sub-agent outputs to reduce token usage (default: True).
            compression_config: Optional custom compression configuration. If None, uses default rules.
            memory_manager_type: Type of memory manager ("sliding_window" or "hierarchical").
        """
        super().__init__(
            name=name,
            description=description,
            parameters=[
                ParameterSchema(
                    name="task",
                    type="string",
                    description="The specific task for the agent to execute",
                    required=True,
                ),
                ParameterSchema(
                    name="context",
                    type="string",
                    description="Context to provide to the agent before executing the task",
                    required=True,
                ),
            ],
            category=ToolCategory.UTILITY,
        )
        self._provider = provider
        self._model_name = model_name
        self._interrupt_before_tool = interrupt_before_tool
        self._tool_confirmation_callback = tool_confirmation_callback
        self._direct_execution_callback = direct_execution_callback
        self._tool_execution_result_callback = tool_execution_result_callback
        self._memory_manager_type = memory_manager_type

        # Compression Configuration
        self._enable_compression = enable_compression
        if enable_compression:
            if compression_config:
                self._compressor = ToolOutputCompressor()
                for tool_name in ToolOutputCompressor.COMPRESSION_RULES:
                    self._compressor.set_config(tool_name, compression_config)
            else:
                self._compressor = ToolOutputCompressor()
        else:
            self._compressor = None

    def _load_aggregated_context(self, main_session_id: str) -> str | None:
        """
        Load the aggregated checkpoint from executors directory if it exists.

        Args:
            main_session_id: The main session ID

        Returns:
            Content of the aggregated checkpoint, or None if not found
        """
        if main_session_id == "standalone":
            return None

        home = Path.home()
        aggregated_path = (
            home
            / ".kader"
            / "memory"
            / "sessions"
            / main_session_id
            / "executors"
            / "checkpoint.md"
        )

        if aggregated_path.exists():
            try:
                return aggregated_path.read_text(encoding="utf-8")
            except Exception:
                return None
        return None

    async def _aload_aggregated_context(self, main_session_id: str) -> str | None:
        """
        Asynchronously load the aggregated checkpoint from executors directory.

        Args:
            main_session_id: The main session ID

        Returns:
            Content of the aggregated checkpoint, or None if not found
        """
        if main_session_id == "standalone":
            return None

        home = Path.home()
        aggregated_path = (
            home
            / ".kader"
            / "memory"
            / "sessions"
            / main_session_id
            / "executors"
            / "checkpoint.md"
        )

        if aggregated_path.exists():
            try:
                return await aread_text(aggregated_path)
            except Exception:
                return None
        return None

    def execute(self, task: str, context: str) -> str:
        """
        Execute a task using a ReActAgent with isolated memory.

        When interrupt_before_tool is True, the agent will pause before each tool
        execution for user confirmation. The task only ends when the agent returns
        its final response.

        Args:
            task: The task to execute.
            context: Context to add to memory before the task.

        Returns:
            A summary of what the agent accomplished.
        """
        # Import here to avoid circular imports
        from kader.agent.agents import ReActAgent

        # Create a fresh memory manager for isolated context
        # Persistence: ~/.kader/memory/sessions/<main-session-id>/executors/<agent-name>-<id>.json
        execution_id = str(uuid.uuid4())
        # Use propagated session ID or 'standalone' if not set
        main_session_id = self._session_id if self._session_id else "standalone"

        home = Path.home()
        memory_dir = (
            home
            / ".kader"
            / "memory"
            / "sessions"
            / main_session_id
            / "executors"
            / f"{self.name}-{execution_id}"
        )
        memory_file = memory_dir / "conversation.json"

        if self._memory_manager_type == "hierarchical":
            memory = PersistentHierarchicalConversationManager(
                file_path=memory_file,
                window_size=20,
                full_context_window=5,
                provider=self._provider,
            )
        else:
            memory = PersistentSlidingWindowConversationManager(
                file_path=memory_file, window_size=20
            )

        # Load aggregated context from previous executors
        aggregated_context = self._load_aggregated_context(main_session_id)
        if aggregated_context:
            full_context = f"## Previous Executor Context\n{aggregated_context}\n\n## Current Task Context\n{context}"
        else:
            full_context = context

        # Add context to memory as user message
        memory.add_message(Message.user(full_context))

        # Get default tools (filesystem, web, command executor) - use cached version
        from kader.tools import get_cached_default_registry

        tools = get_cached_default_registry()

        # Create ExecutorAgentPrompt with tool descriptions
        system_prompt = ExecutorAgentPrompt(tools=tools.tools)

        # Create the ReActAgent with separate memory and executor prompt
        agent = ReActAgent(
            name=f"{self.name}_worker",
            tools=tools,
            system_prompt=system_prompt,
            provider=self._provider,
            memory=memory,
            model_name=self._model_name,
            interrupt_before_tool=self._interrupt_before_tool,
            tool_confirmation_callback=self._tool_confirmation_callback,
            direct_execution_callback=self._direct_execution_callback,
            tool_execution_result_callback=self._tool_execution_result_callback,
        )

        try:
            # Invoke the agent with the task
            # The agent will handle tool interruptions internally
            response = agent.invoke(task)

            # Generate checkpoint and aggregate it
            try:
                checkpointer = Checkpointer(provider=agent.provider)
                checkpoint_path = checkpointer.generate_checkpoint(str(memory_file))
                checkpoint_content = Path(checkpoint_path).read_text(encoding="utf-8")

                # Aggregate the checkpoint into the main executors checkpoint
                if main_session_id != "standalone":
                    aggregator = ContextAggregator(
                        session_id=main_session_id, provider=agent.provider
                    )
                    # Use relative path from executors directory
                    relative_path = f"{self.name}-{execution_id}/checkpoint.md"
                    aggregator.aggregate(relative_path, subagent_name=self.name)

                # Append the agent's response to the checkpoint content if it exists
                response_content = None
                if hasattr(response, "content"):
                    response_content = str(response.content)
                elif isinstance(response, dict):
                    response_content = str(response.get("content", str(response)))
                else:
                    response_content = str(response)

                if response_content and response_content != "None":
                    checkpoint_content += f"\n\nResponse:\n{response_content}"

                # Compress output if compression is enabled
                if self._compressor:
                    checkpoint_content = self._compressor.compress(
                        tool_name="executor",
                        output=checkpoint_content,
                    )

                return checkpoint_content
            except Exception:
                # Fallback to raw response if checkpointing fails
                result = ""
                if hasattr(response, "content"):
                    result = str(response.content)
                elif isinstance(response, dict):
                    result = str(response.get("content", str(response)))
                else:
                    result = str(response)

                # Compress output if compression is enabled
                if self._compressor:
                    result = self._compressor.compress(
                        tool_name="executor",
                        output=result,
                    )
                return result

        except Exception as e:
            return f"Agent execution failed: {str(e)}"

    async def aexecute(self, task: str, context: str) -> str:
        """
        Asynchronously execute a task using a ReActAgent.

        When interrupt_before_tool is True, the agent will pause before each tool
        execution for user confirmation. The task only ends when the agent returns
        its final response.

        Args:
            task: The task to execute.
            context: Context to add to memory before the task.

        Returns:
            A summary of what the agent accomplished.
        """
        # Import here to avoid circular imports
        from kader.agent.agents import ReActAgent

        # Create a fresh memory manager for isolated context
        # Persistence: ~/.kader/memory/sessions/<main-session-id>/executors/<agent-name>-<id>.json
        execution_id = str(uuid.uuid4())
        # Use propagated session ID or 'standalone' if not set
        main_session_id = self._session_id if self._session_id else "standalone"

        home = Path.home()
        memory_dir = (
            home
            / ".kader"
            / "memory"
            / "sessions"
            / main_session_id
            / "executors"
            / f"{self.name}-{execution_id}"
        )
        memory_file = memory_dir / "conversation.json"

        if self._memory_manager_type == "hierarchical":
            memory = PersistentHierarchicalConversationManager(
                file_path=memory_file,
                window_size=20,
                full_context_window=5,
                provider=self._provider,
            )
        else:
            memory = PersistentSlidingWindowConversationManager(
                file_path=memory_file, window_size=20
            )

        # Load aggregated context from previous executors (async)
        aggregated_context = await self._aload_aggregated_context(main_session_id)
        if aggregated_context:
            full_context = f"## Previous Executor Context\n{aggregated_context}\n\n## Current Task Context\n{context}"
        else:
            full_context = context

        # Add context to memory as user message
        memory.add_message(Message.user(full_context))

        # Get default tools (filesystem, web, command executor) - use cached version
        from kader.tools import get_cached_default_registry

        tools = get_cached_default_registry()

        # Create ExecutorAgentPrompt with tool descriptions
        system_prompt = ExecutorAgentPrompt(tools=tools.tools)

        # Create the ReActAgent with separate memory and executor prompt
        agent = ReActAgent(
            name=f"{self.name}_worker",
            tools=tools,
            system_prompt=system_prompt,
            provider=self._provider,
            memory=memory,
            model_name=self._model_name,
            interrupt_before_tool=self._interrupt_before_tool,
            tool_confirmation_callback=self._tool_confirmation_callback,
            direct_execution_callback=self._direct_execution_callback,
            tool_execution_result_callback=self._tool_execution_result_callback,
        )

        try:
            # Invoke the agent asynchronously
            # The agent will handle tool interruptions internally
            response = await agent.ainvoke(task)

            # Generate checkpoint and aggregate it
            try:
                checkpointer = Checkpointer(provider=agent.provider)
                checkpoint_path = await checkpointer.agenerate_checkpoint(
                    str(memory_file)
                )
                checkpoint_content = await aread_text(Path(checkpoint_path))

                # Aggregate the checkpoint into the main executors checkpoint
                if main_session_id != "standalone":
                    aggregator = ContextAggregator(
                        session_id=main_session_id, provider=agent.provider
                    )
                    # Use relative path from executors directory
                    relative_path = f"{self.name}-{execution_id}/checkpoint.md"
                    await aggregator.aaggregate(relative_path, subagent_name=self.name)

                # Append the agent's response to the checkpoint content if it exists
                response_content = None
                if hasattr(response, "content"):
                    response_content = str(response.content)
                elif isinstance(response, dict):
                    response_content = str(response.get("content", str(response)))
                else:
                    response_content = str(response)

                if response_content and response_content != "None":
                    checkpoint_content += f"\n\nResponse:\n{response_content}"

                # Compress output if compression is enabled
                if self._compressor:
                    checkpoint_content = self._compressor.compress(
                        tool_name="executor",
                        output=checkpoint_content,
                    )

                return checkpoint_content
            except Exception:
                # Fallback to raw response if checkpointing fails
                result = ""
                if hasattr(response, "content"):
                    result = str(response.content)
                elif isinstance(response, dict):
                    result = str(response.get("content", str(response)))
                else:
                    result = str(response)

                # Compress output if compression is enabled
                if self._compressor:
                    result = self._compressor.compress(
                        tool_name="executor",
                        output=result,
                    )
                return result

        except Exception as e:
            return f"Agent execution failed: {str(e)}"

    def get_interruption_message(self, task: str, **kwargs: Any) -> str:
        """
        Get a message describing the agent action for user confirmation.

        Args:
            task: The task the agent will execute.

        Returns:
            A formatted string describing the action.
        """
        # Truncate long tasks for readability
        task_preview = task[:100] + "..." if len(task) > 100 else task
        return f"execute {self.name}: {task_preview}"
