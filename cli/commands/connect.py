"""Connect command for setting up provider API keys in Kader CLI."""

import asyncio

from kader.config import ENV_FILE_PATH, save_env_var
from kader.providers import LLMProviderFactory

from .base import BaseCommand


class ConnectCommand(BaseCommand):
    """Command to connect a provider by setting its API key.

    Displays an interactive list of available providers, prompts the user
    for their API key, and saves it to ~/.kader/.env.
    """

    async def execute(self) -> None:
        """Execute the connect command with interactive provider selection."""
        from prompt_toolkit.shortcuts import choice as pt_choice
        from prompt_toolkit.styles import Style

        custom_style = Style.from_dict(
            {
                "selected": "reverse",
                "radio-selected": "reverse",
                "radio-checked": "reverse",
                "selected-option": "reverse",
                "pointer": "reverse",
            }
        )

        # Build provider options for the choice menu
        provider_names = list(LLMProviderFactory.PROVIDERS.keys())
        options = []
        for name in provider_names:
            env_key = LLMProviderFactory.get_provider_env_key(name)
            options.append((name, f"{name.title()} — {env_key}"))

        self.app.console.print()
        self.app.console.print("  [bold cyan]Connect a Provider[/bold cyan]")
        self.app.console.print(
            "  [dim]Select a provider below to set its API key.[/dim]"
        )

        loop = asyncio.get_event_loop()
        try:
            provider_choice = await loop.run_in_executor(
                None,
                lambda: pt_choice(
                    message="  Provider:",
                    options=options,
                    style=custom_style,
                ),
            )
        except (Exception, KeyboardInterrupt):
            self.app._print_system_message("Provider selection cancelled.")
            return

        env_key = LLMProviderFactory.get_provider_env_key(provider_choice)

        self.app.console.print()
        self.app.console.print(
            f"  [dim]Enter API key for[/dim] [bold]{provider_choice.title()}[/bold]"
            f" [dim]({env_key})[/dim]:"
        )

        try:
            from prompt_toolkit.formatted_text import HTML

            api_key = await self.app._prompt_session.prompt_async(
                HTML(f"<ansiwhite>  {env_key}> </ansiwhite>")
            )
            api_key = api_key.strip()
            if not api_key:
                self.app._print_system_message(
                    f"No API key entered. {provider_choice.title()} not connected."
                )
                return

            success = save_env_var(ENV_FILE_PATH, env_key, api_key)
            if success:
                self.app.console.print()
                self.app.console.print(
                    f"  [kader.green]✓[/kader.green] [bold]{provider_choice.title()}[/bold]"
                    " connected successfully!"
                )
                self.app.console.print(f"  [dim]API key saved to {ENV_FILE_PATH}[/dim]")
                self.app.console.print(
                    "  [dim]Use [/dim][bold]/models[/bold][dim] to browse available models.[/dim]"
                )
            else:
                self.app._print_system_message(
                    f"Failed to save API key for {provider_choice.title()}."
                )
        except (EOFError, KeyboardInterrupt):
            self.app._print_system_message(
                f"API key input cancelled. {provider_choice.title()} not connected."
            )
