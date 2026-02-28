"""Initialize command for Kader CLI.

Creates a .kader directory in the current working directory and generates
KADER.md file using an AgentTool to analyze the codebase.
"""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from kader.prompts.cli_prompts import InitCommandPrompt
from kader.tools.agent import AgentTool

from .base import BaseCommand

if TYPE_CHECKING:
    pass


class InitializeCommand(BaseCommand):
    """Handles the /init command for Kader CLI."""

    async def execute(self) -> None:
        """Execute the initialization command.

        Creates .kader directory and generates KADER.md using an AgentTool
        with the current session's LLM provider.
        """
        console = self.app.console

        # Step 1: Create .kader directory
        kader_dir = Path.cwd() / ".kader"
        try:
            kader_dir.mkdir(exist_ok=True)
            console.print(
                f"  [kader.cyan]▶[/kader.cyan] Created directory: `{kader_dir}`"
            )
        except Exception as e:
            console.print(f"  [kader.red]✗[/kader.red] Failed to create directory: {e}")
            return

        # Step 2: Check if KADER.md already exists
        kader_md_path = kader_dir / "KADER.md"
        if kader_md_path.exists():
            console.print(
                f"  [kader.yellow][!][/kader.yellow] "
                f"KADER.md already exists at `{kader_md_path}`"
            )
            console.print(
                "  [kader.cyan]▶[/kader.cyan] "
                "Re-generating KADER.md with updated analysis..."
            )

        # Step 3: Use AgentTool to generate KADER.md content
        console.print(
            "  [kader.cyan]▶[/kader.cyan] Analyzing codebase and generating KADER.md..."
        )

        try:
            provider = self.app._workflow.planner.provider
            model_name = self.app._current_model

            prompt = InitCommandPrompt()
            task = str(prompt)

            agent_tool = AgentTool(
                name="init_agent",
                description="Agent to initialize KADER.md by analyzing codebase",
                provider=provider,
                model_name=model_name,
                interrupt_before_tool=False,
                tool_confirmation_callback=self._tool_confirmation_callback,
            )

            if self.app._current_session_id:
                agent_tool.set_session_id(self.app._current_session_id)

            context = (
                f"Working directory: {Path.cwd()}\n"
                f"Target file: {kader_md_path}\n"
                f"Use filesystem tools to explore the codebase "
                f"and create the KADER.md file."
            )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.app._agent_executor,
                lambda: agent_tool.execute(task=task, context=context),
            )

            console.print(
                f"  [kader.green]✓[/kader.green] "
                f"Successfully created KADER.md at `{kader_md_path}`"
            )

            # Show preview
            preview = result[:500] + "..." if len(result) > 500 else result
            console.print(f"\n```markdown\n{preview}\n```")

        except Exception as e:
            console.print(f"  [kader.red]✗[/kader.red] Error generating KADER.md: {e}")
