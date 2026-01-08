"""
Specific Agent Implementations.
"""

from typing import Optional, Union

from kader.agent.base import BaseAgent
from kader.memory import ConversationManager
from kader.prompts import PlanningAgentPrompt, PromptBase, ReActAgentPrompt
from kader.providers.base import BaseLLMProvider
from kader.tools import BaseTool, TodoTool, ToolRegistry


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning and Acting) Agent.

    Uses a ReAct prompt strategy to reason about tasks and use tools.
    """

    def __init__(
        self,
        name: str,
        tools: Union[list[BaseTool], ToolRegistry],
        system_prompt: Optional[Union[str, PromptBase]] = None,
        provider: Optional[BaseLLMProvider] = None,
        memory: Optional[ConversationManager] = None,
        retry_attempts: int = 3,
        model_name: str = "qwen3-coder:480b-cloud",
        session_id: Optional[str] = None,
        use_persistence: bool = False,
        interrupt_before_tool: bool = True,
        tool_confirmation_callback: Optional[callable] = None,
    ) -> None:
        # Resolve tools for prompt context if necessary
        # The base agent handles tool registration, but for the prompt template
        # we might need to pass tool descriptions initially.

        # Temporary logic to get tool names/descriptions for the prompt
        # In a real scenario, this might need dynamic updates or be handled by the Prompt class itself
        # accessing the agent's registry. Here we do a best-effort pre-fill.

        _tools_list = []
        if isinstance(tools, list):
            _tools_list = tools
        elif isinstance(tools, ToolRegistry):
            _tools_list = tools.tools

        tool_names = ", ".join([t.name for t in _tools_list])

        if system_prompt is None:
            system_prompt = ReActAgentPrompt(
                tools=_tools_list,
                tool_names=tool_names,
                input="",  # This acts as a placeholder or initial context
            )

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            provider=provider,
            memory=memory,
            retry_attempts=retry_attempts,
            model_name=model_name,
            session_id=session_id,
            use_persistence=use_persistence,
            interrupt_before_tool=interrupt_before_tool,
            tool_confirmation_callback=tool_confirmation_callback,
        )


class PlanningAgent(BaseAgent):
    """
    Planning Agent.

    Breaks tasks into plans and executes them.
    """

    def __init__(
        self,
        name: str,
        tools: Union[list[BaseTool], ToolRegistry],
        system_prompt: Optional[Union[str, PromptBase]] = None,
        provider: Optional[BaseLLMProvider] = None,
        memory: Optional[ConversationManager] = None,
        retry_attempts: int = 3,
        model_name: str = "qwen3-coder:480b-cloud",
        session_id: Optional[str] = None,
        use_persistence: bool = False,
        interrupt_before_tool: bool = True,
        tool_confirmation_callback: Optional[callable] = None,
    ) -> None:
        # Ensure TodoTool is available
        _todo_tool = TodoTool()
        if isinstance(tools, ToolRegistry):
            if _todo_tool.name not in tools:
                tools.register(_todo_tool)
        elif isinstance(tools, list):
            if not any(t.name == _todo_tool.name for t in tools):
                tools.append(_todo_tool)

        _tools_list = []
        if isinstance(tools, list):
            _tools_list = tools
        elif isinstance(tools, ToolRegistry):
            _tools_list = tools.tools

        if system_prompt is None:
            system_prompt = PlanningAgentPrompt(
                tools=_tools_list, input="", agent_scratchpad=""
            )

        super().__init__(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            provider=provider,
            memory=memory,
            retry_attempts=retry_attempts,
            model_name=model_name,
            session_id=session_id,
            use_persistence=use_persistence,
            interrupt_before_tool=interrupt_before_tool,
            tool_confirmation_callback=tool_confirmation_callback,
        )
