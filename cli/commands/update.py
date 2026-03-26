"""Update command for Kader CLI.

Checks for updates and updates Kader if a newer version is available.
"""

import os
import subprocess
import sys
from importlib.metadata import version as get_version
from typing import TYPE_CHECKING

from .base import BaseCommand

if TYPE_CHECKING:
    pass


def check_outdated(package_name: str, current_version: str) -> tuple[bool, str]:
    """Check if a package is outdated.

    Args:
        package_name: The name of the package to check.
        current_version: The current version of the package.

    Returns:
        A tuple of (is_outdated, latest_version).
    """
    from urllib.request import urlopen

    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        with urlopen(url, timeout=10) as response:
            data = response.read()
            latest_version = data.split(b'"version":"')[1].split(b'"')[0].decode()
            is_outdated = latest_version != current_version
            return is_outdated, latest_version
    except Exception:
        return False, current_version


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
