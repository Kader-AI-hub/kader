from datetime import datetime
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
