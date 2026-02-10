"""Base command class for Kader CLI commands.

Provides common patterns and infrastructure for CLI commands.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ..app import KaderApp


class BaseCommand(ABC):
    """Abstract base class for CLI commands.

    Provides common infrastructure including:
    - App dependency injection via constructor
    - Tool confirmation callback for agent-based commands
    - Abstract execute method for command implementation
    """

    def __init__(self, app: "KaderApp") -> None:
        """Initialize the command handler.

        Args:
            app: The KaderApp instance to interact with.
        """
        self.app = app

    def _tool_confirmation_callback(self, message: str) -> Tuple[bool, Optional[str]]:
        """Callback for tool confirmation from agent.

        This default implementation auto-approves all tools.
        Subclasses can override this to implement custom confirmation logic.

        Args:
            message: The tool execution message.

        Returns:
            Tuple of (should_execute, additional_context).
        """
        return (True, None)

    @abstractmethod
    async def execute(self) -> None:
        """Execute the command.

        This method must be implemented by subclasses to provide
        the command's main logic.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        pass
