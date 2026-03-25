"""Update command for Kader CLI.

Checks for updates and updates Kader if a newer version is available.
"""

import os
import subprocess
import sys
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

from outdated import check_outdated

from .base import BaseCommand

if TYPE_CHECKING:
    pass


class UpdateCommand(BaseCommand):
    """Handles the /update command for Kader CLI."""

    async def execute(self) -> None:
        """Execute the update command.

        Checks for updates using the outdated library and updates Kader
        if a newer version is available.
        """
        console = self.app.console
        current_version = get_version("kader")

        is_outdated, latest_version = check_outdated("kader", current_version)

        if is_outdated:
            console.print(
                f"  [kader.yellow]Updating Kader from v{current_version} to v{latest_version}...[/kader.yellow]"
            )
            subprocess.run(["uv", "tool", "upgrade", "kader"], check=True)
            console.print(
                f"  [kader.green]✓ Updated to v{latest_version}. Restarting...[/kader.green]"
            )
            os.execv(sys.executable, [sys.executable, "-m", "cli"])
        else:
            console.print(
                f"  [kader.green]✓ You are running the latest version v{current_version}[/kader.green]"
            )
