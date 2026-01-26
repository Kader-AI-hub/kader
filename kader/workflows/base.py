"""
Base Workflow Class.

Abstract base class for all workflow implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional

from kader.providers.base import BaseLLMProvider


class BaseWorkflow(ABC):
    """
    Abstract base class for workflow implementations.

    Workflows orchestrate multiple agents or agent interactions to accomplish
    complex tasks. They provide a structured way to compose agent behaviors
    and manage the flow of information between agents.

    Subclasses must implement:
    - run: Synchronous workflow execution
    - arun: Asynchronous workflow execution
    """

    def __init__(
        self,
        name: str,
        provider: Optional[BaseLLMProvider] = None,
        model_name: str = "qwen3-coder:480b-cloud",
        interrupt_before_tool: bool = True,
    ) -> None:
        """
        Initialize the base workflow.

        Args:
            name: Name of the workflow instance.
            provider: LLM provider for agents in the workflow.
            model_name: Model to use if no provider specified.
            interrupt_before_tool: Whether to pause before tool execution.
        """
        self.name = name
        self.provider = provider
        self.model_name = model_name
        self.interrupt_before_tool = interrupt_before_tool

    @abstractmethod
    def run(self, task: str) -> str:
        """
        Execute the workflow synchronously.

        Args:
            task: The task or goal to accomplish.

        Returns:
            A summary of the workflow execution and results.
        """
        ...

    @abstractmethod
    async def arun(self, task: str) -> str:
        """
        Execute the workflow asynchronously.

        Args:
            task: The task or goal to accomplish.

        Returns:
            A summary of the workflow execution and results.
        """
        ...
