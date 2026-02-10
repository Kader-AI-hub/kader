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
        from ..widgets import ConversationView

        conversation = self.app.query_one("#conversation-view", ConversationView)

        # Step 1: Create .kader directory
        kader_dir = Path.cwd() / ".kader"
        try:
            kader_dir.mkdir(exist_ok=True)
            conversation.add_message(
                f"[>] Created directory: `{kader_dir}`", "assistant"
            )
        except Exception as e:
            conversation.add_message(
                f"(-) Failed to create directory: {e}", "assistant"
            )
            return

        # Step 2: Check if KADER.md already exists
        kader_md_path = kader_dir / "KADER.md"
        if kader_md_path.exists():
            conversation.add_message(
                f"(!) KADER.md already exists at `{kader_md_path}`", "assistant"
            )
            conversation.add_message(
                "[>] Re-generating KADER.md with updated analysis...", "assistant"
            )

        # Step 3: Use AgentTool to generate KADER.md content
        conversation.add_message(
            "[>] Analyzing codebase and generating KADER.md...", "assistant"
        )

        try:
            # Get the current provider from the workflow
            provider = self.app._workflow.planner.provider
            model_name = self.app._current_model

            # Create the prompt for the agent - this IS the task
            prompt = InitCommandPrompt()
            task = str(prompt)

            # Create AgentTool with the current provider and model
            agent_tool = AgentTool(
                name="init_agent",
                description="Agent to initialize KADER.md by analyzing codebase",
                provider=provider,
                model_name=model_name,
                interrupt_before_tool=False,  # Auto-execute tools for init
                tool_confirmation_callback=self._tool_confirmation_callback,
            )

            # Set the session ID for the agent tool
            if self.app._current_session_id:
                agent_tool.set_session_id(self.app._current_session_id)

            # Context provides additional information about the working directory
            context = (
                f"Working directory: {Path.cwd()}\n"
                f"Target file: {kader_md_path}\n"
                f"Use filesystem tools to explore the codebase and create the KADER.md file."
            )

            # Run agent execution in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.app._agent_executor,
                lambda: agent_tool.execute(task=task, context=context),
            )

            conversation.add_message(
                f"(+) Successfully created KADER.md at `{kader_md_path}`",
                "assistant",
            )

            # Show preview of the result
            preview = result[:500] + "..." if len(result) > 500 else result
            conversation.add_message(f"```markdown\n{preview}\n```", "assistant")

            self.app.notify("KADER.md created successfully!", severity="information")

        except Exception as e:
            conversation.add_message(f"(-) Error generating KADER.md: {e}", "assistant")
            self.app.notify(f"Error: {e}", severity="error")
