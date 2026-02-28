"""
Planner Executor Workflow.

Orchestrates a PlanningAgent with TodoTool and AgentTool to break down tasks
and delegate sub-tasks to executor agents.
"""

from pathlib import Path
from typing import Callable, Optional, Tuple

from kader.agent.agents import PlanningAgent
from kader.memory import (
    HierarchicalConversationManager,
    SlidingWindowConversationManager,
)
from kader.memory.types import get_default_memory_dir
from kader.prompts import KaderPlannerPrompt
from kader.providers.base import BaseLLMProvider, Message
from kader.tools import AgentTool, TodoTool, ToolRegistry
from kader.tools.filesys import GlobTool, GrepTool, ReadDirectoryTool, ReadFileTool
from kader.utils import Checkpointer

from .base import BaseWorkflow


class PlannerExecutorWorkflow(BaseWorkflow):
    """
    Planner-Executor Workflow using PlanningAgent with sub-agent delegation.

    This workflow:
    1. Accepts a user task
    2. PlanningAgent creates a plan using TodoTool
    3. For each sub-task, PlanningAgent can delegate to AgentTool (executor)
    4. Executor outputs are added to planner's memory as assistant messages
    5. PlanningAgent updates TodoTool status and continues until complete

    Example:
        workflow = PlannerExecutorWorkflow(name="project_workflow")
        result = workflow.run("Create a Python project with README and tests")
    """

    def __init__(
        self,
        name: str,
        provider: Optional[BaseLLMProvider] = None,
        model_name: str = "qwen3-coder:480b-cloud",
        interrupt_before_tool: bool = True,
        tool_confirmation_callback: Optional[
            Callable[..., Tuple[bool, Optional[str]]]
        ] = None,
        direct_execution_callback: Optional[Callable[..., None]] = None,
        tool_execution_result_callback: Optional[Callable[..., None]] = None,
        use_persistence: bool = False,
        session_id: Optional[str] = None,
        executor_names: Optional[list[str]] = None,
        memory_manager_type: str = "hierarchical",
        skills_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the Planner-Executor workflow.

        Args:
            name: Name of the workflow instance.
            provider: LLM provider for agents.
            model_name: Model to use if no provider specified.
            interrupt_before_tool: Whether to pause before tool execution.
            tool_confirmation_callback: Callback for tool confirmation UI.
            use_persistence: Enable session persistence.
            session_id: Optional session ID to resume.
            executor_names: Names for executor sub-agents (default: ["executor"]).
            memory_manager_type: Type of memory manager ("sliding_window" or "hierarchical").
            skills_dirs: Optional list of custom skill directories for sub-agents.
                        If None, uses default directories (~/.kader/skills and ./.kader/).
            priority_dir: Optional directory checked first (higher priority) for skills.
        """
        super().__init__(
            name=name,
            provider=provider,
            model_name=model_name,
            interrupt_before_tool=interrupt_before_tool,
        )
        self.tool_confirmation_callback = tool_confirmation_callback
        self.direct_execution_callback = direct_execution_callback
        self.tool_execution_result_callback = tool_execution_result_callback
        self.use_persistence = use_persistence
        self.session_id = session_id
        self.executor_names = executor_names or ["executor"]
        self.memory_manager_type = memory_manager_type
        self.skills_dirs = skills_dirs
        self.priority_dir = priority_dir

        # Build the planner agent with tools
        self._planner = self._build_planner()

    def _load_checkpoint_context(self) -> Optional[str]:
        """
        Load checkpoint context from the session directory if it exists.

        Returns:
            The checkpoint markdown content if file exists, None otherwise.
        """
        if not self.session_id:
            return None

        checkpoint_path = (
            get_default_memory_dir() / "sessions" / self.session_id / "checkpoint.md"
        )

        if checkpoint_path.exists():
            try:
                return checkpoint_path.read_text(encoding="utf-8")
            except Exception:
                return None

        return None

    def _build_planner(self) -> PlanningAgent:
        """Build the PlanningAgent with TodoTool, file system tools, and AgentTool(s)."""
        registry = ToolRegistry()

        # TodoTool is added implicitly by PlanningAgent, but we ensure it's there
        registry.register(TodoTool())

        # Add reading and file system search tools
        registry.register(ReadFileTool())
        registry.register(ReadDirectoryTool())
        registry.register(GlobTool())
        registry.register(GrepTool())

        # Add AgentTool(s) for sub-task delegation
        for executor_name in self.executor_names:
            agent_tool = AgentTool(
                name=executor_name,
                description=(
                    f"Delegate a sub-task to the '{executor_name}' agent. "
                    "Use this when a specific task needs to be executed by a "
                    "specialized worker agent. Provide a clear task description."
                ),
                provider=self.provider,
                model_name=self.model_name,
                interrupt_before_tool=self.interrupt_before_tool,
                tool_confirmation_callback=self.tool_confirmation_callback,
                direct_execution_callback=self.direct_execution_callback,
                tool_execution_result_callback=self.tool_execution_result_callback,
                memory_manager_type=self.memory_manager_type,
                skills_dirs=self.skills_dirs,
                priority_dir=self.priority_dir,
            )
            registry.register(agent_tool)

        # Create memory for the planner based on configured type
        if self.memory_manager_type == "hierarchical":
            memory = HierarchicalConversationManager(
                window_size=20,
                full_context_window=5,
                provider=self.provider,
            )
        else:
            memory = SlidingWindowConversationManager(window_size=20)

        # Load checkpoint context if it exists from previous iterations
        checkpoint_context = self._load_checkpoint_context()

        # Create the Kader system prompt with tool descriptions and context
        system_prompt = KaderPlannerPrompt(
            tools=registry.tools,
            context=checkpoint_context,
        )

        # Build the PlanningAgent
        # Note: The planner itself runs without interruption (TodoTool, AgentTool execute directly)
        # Sub-agents inside AgentTool can still have their own interrupt settings
        planner = PlanningAgent(
            name=f"{self.name}_planner",
            tools=registry,
            system_prompt=system_prompt,
            provider=self.provider,
            memory=memory,
            model_name=self.model_name,
            use_persistence=self.use_persistence,
            interrupt_before_tool=False,  # Planner executes TodoTool/AgentTool directly
            tool_confirmation_callback=self.tool_confirmation_callback,
            direct_execution_callback=self.direct_execution_callback,
            tool_execution_result_callback=self.tool_execution_result_callback,
        )

        if not self.session_id:
            self.session_id = planner.session_id

        return planner

    def _add_executor_output_to_memory(self, executor_name: str, output: str) -> None:
        """
        Add executor agent output to planner's memory as assistant message.

        Args:
            executor_name: Name of the executor that produced the output.
            output: The output/result from the executor.
        """
        # Format the executor output as an assistant message
        formatted_output = f"[{executor_name} completed]: {output}"
        self._planner.memory.add_message(Message.assistant(formatted_output))

    def _create_checkpoint(self) -> Optional[str]:
        """
        Create a checkpoint of the current session using the Checkpointer.

        Returns:
            Path to the checkpoint file if created, None otherwise.
        """
        if not self.session_id or not self.use_persistence:
            return None

        try:
            checkpointer = Checkpointer(provider=self._planner.provider)
            memory_path = f"{self.session_id}/conversation.json"
            checkpoint_path = checkpointer.generate_checkpoint(memory_path)
            return checkpoint_path
        except Exception:
            # Silently fail if checkpointing fails - don't interrupt workflow
            return None

    def run(self, task: str) -> str:
        """
        Execute the planner-executor workflow synchronously.

        Args:
            task: The main task to accomplish.

        Returns:
            Final response from the planner summarizing completed work.
        """
        # Invoke the planner with the task
        # The PlanningAgent will:
        # 1. Create a plan using TodoTool
        # 2. Delegate sub-tasks using AgentTool when needed
        # 3. Update todo status as tasks complete
        # 4. Continue until all tasks are done

        response = self._planner.invoke(task)

        # Create checkpoint after execution completes
        self._create_checkpoint()

        # Extract content from response
        if hasattr(response, "content"):
            return str(response.content)
        elif isinstance(response, dict):
            return str(response.get("content", str(response)))
        else:
            return str(response)

    async def arun(self, task: str) -> str:
        """
        Execute the planner-executor workflow asynchronously.

        Args:
            task: The main task to accomplish.

        Returns:
            Final response from the planner summarizing completed work.
        """
        response = await self._planner.ainvoke(task)

        # Create checkpoint after execution completes
        self._create_checkpoint()

        if hasattr(response, "content"):
            return str(response.content)
        elif isinstance(response, dict):
            return str(response.get("content", str(response)))
        else:
            return str(response)

    @property
    def planner(self) -> PlanningAgent:
        """Get the underlying PlanningAgent instance."""
        return self._planner

    def reset(self) -> None:
        """Reset the workflow by rebuilding the planner with fresh memory."""
        self._planner = self._build_planner()
