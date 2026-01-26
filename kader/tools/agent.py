"""
Agent Tool - Use a ReActAgent as a callable tool.

Allows spawning sub-agents to execute specific tasks with isolated memory contexts.
"""

from typing import Any, Callable, Optional, Tuple

from kader.memory import SlidingWindowConversationManager
from kader.providers.base import BaseLLMProvider

from .base import BaseTool, ParameterSchema, ToolCategory


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
            ],
            category=ToolCategory.UTILITY,
        )
        self._provider = provider
        self._model_name = model_name
        self._interrupt_before_tool = interrupt_before_tool
        self._tool_confirmation_callback = tool_confirmation_callback

    def execute(self, task: str) -> str:
        """
        Execute a task using a ReActAgent with isolated memory.

        When interrupt_before_tool is True, the agent will pause before each tool
        execution for user confirmation. The task only ends when the agent returns
        its final response.

        Args:
            task: The task to execute.

        Returns:
            A summary of what the agent accomplished.
        """
        # Import here to avoid circular imports
        from kader.agent.agents import ReActAgent
        from kader.tools import get_default_registry

        # Create a fresh memory manager for isolated context
        memory = SlidingWindowConversationManager(window_size=20)

        # Get default tools (filesystem, web, command executor)
        tools = get_default_registry()

        # Create the ReActAgent with separate memory
        agent = ReActAgent(
            name=f"{self.name}_worker",
            tools=tools,
            provider=self._provider,
            memory=memory,
            model_name=self._model_name,
            interrupt_before_tool=self._interrupt_before_tool,
            tool_confirmation_callback=self._tool_confirmation_callback,
        )

        try:
            # Invoke the agent with the task
            # The agent will handle tool interruptions internally
            response = agent.invoke(task)

            # Extract and return the response content
            if hasattr(response, "content"):
                return str(response.content)
            elif isinstance(response, dict):
                return str(response.get("content", str(response)))
            else:
                return str(response)

        except Exception as e:
            return f"Agent execution failed: {str(e)}"

    async def aexecute(self, task: str) -> str:
        """
        Asynchronously execute a task using a ReActAgent.

        When interrupt_before_tool is True, the agent will pause before each tool
        execution for user confirmation. The task only ends when the agent returns
        its final response.

        Args:
            task: The task to execute.

        Returns:
            A summary of what the agent accomplished.
        """
        # Import here to avoid circular imports
        from kader.agent.agents import ReActAgent
        from kader.tools import get_default_registry

        # Create a fresh memory manager for isolated context
        memory = SlidingWindowConversationManager(window_size=20)

        # Get default tools (filesystem, web, command executor)
        tools = get_default_registry()

        # Create the ReActAgent with separate memory
        agent = ReActAgent(
            name=f"{self.name}_worker",
            tools=tools,
            provider=self._provider,
            memory=memory,
            model_name=self._model_name,
            interrupt_before_tool=self._interrupt_before_tool,
            tool_confirmation_callback=self._tool_confirmation_callback,
        )

        try:
            # Invoke the agent asynchronously
            # The agent will handle tool interruptions internally
            response = await agent.ainvoke(task)

            # Extract and return the response content
            if hasattr(response, "content"):
                return str(response.content)
            elif isinstance(response, dict):
                return str(response.get("content", str(response)))
            else:
                return str(response)

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
