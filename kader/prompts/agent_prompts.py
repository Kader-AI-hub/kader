from datetime import datetime
from pathlib import Path
from typing import Any

from .base import PromptBase


class BasicAssistancePrompt(PromptBase):
    """Basic assistance prompt with date context."""

    def __init__(self, **kwargs: Any) -> None:
        template = "You are a helpful AI assistant. Today is {{ date }}."
        kwargs.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
        super().__init__(template=template, **kwargs)


class ReActAgentPrompt(PromptBase):
    """Prompt for ReAct Agent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(template_path="react_agent.j2", **kwargs)


class PlanningAgentPrompt(PromptBase):
    """Prompt for Planning Agent."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(template_path="planning_agent.j2", **kwargs)


class KaderPlannerPrompt(PromptBase):
    """
    Prompt for Kader Planner Agent.

    Enhanced planning prompt with specific instructions for:
    - Using Agent as a Tool with proper task/context parameters
    - Tracking completed actions to avoid repetition
    """

    def __init__(self, **kwargs: Any) -> None:
        kader_md_path = Path(".kader/KADER.md")
        kwargs.setdefault("kader_md_exists", kader_md_path.exists())
        super().__init__(template_path="kader_planner.j2", **kwargs)


class ExecutorAgentPrompt(PromptBase):
    """
    Prompt for Executor Agent (sub-agents in PlannerExecutorWorkflow).

    Emphasizes:
    - Careful thinking before each action
    - Safe execution with error handling
    - Detailed step-by-step reporting of what was done
    - Structured final answer with files created, summary, and issues
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(template_path="executor_agent.j2", **kwargs)
