"""Refresh command for Kader CLI.

Refreshes settings and reloads callbacks without restarting the CLI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from cli.commands.base import BaseCommand

if TYPE_CHECKING:
    pass


class RefreshCommand(BaseCommand):
    """Handles the /refresh command for Kader CLI.

    Reloads settings and callbacks, recreates the workflow with new configuration.
    """

    async def execute(self) -> None:
        """Execute the refresh command."""
        try:
            self.app._refresh_settings()
            self.app.console.print(
                r"  [kader.green]\[+][/kader.green] Settings and callbacks refreshed successfully"
            )
        except Exception as e:
            logger.error(f"Failed to refresh settings: {e}")
            self.app.console.print(rf"  [kader.red]\[-] Failed to refresh: {e}")
