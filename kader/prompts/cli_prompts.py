from typing import Any

from .base import PromptBase


class InitCommandPrompt(PromptBase):
    """
    Prompt for init command.

    Prompts the agent to analyze the codebase and create an KADER.md file
    containing build/lint/test commands and code style guidelines for
    agentic coding agents working in this repository.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(template_path="init_command_prompt.j2", **kwargs)
