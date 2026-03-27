"""
LLM callback implementations for Kader framework.

Provides callback classes that hook into LLM invocation lifecycle.
"""

from typing import Any

from .base import BaseCallback, CallbackContext


class LLMCallback(BaseCallback):
    """
    Callback for LLM invocation events.

    Provides hooks before and after LLM calls with support for
    filtering by model names. Can modify messages and config before
    the call, and transform responses after.

    Example:
        class LoggingLLMCallback(LLMCallback):
            def on_llm_start(self, context, messages, config):
                print(f"LLM called with {len(messages)} messages")
                return messages, config

            def on_llm_end(self, context, messages, response):
                print(f"LLM responded")
                return response
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """
        Initialize LLM callback.

        Args:
            model_names: List of model names to respond to. None means all models.
                        If specified, callback only fires for matching models.
            enabled: Whether this callback is active.
        """
        super().__init__(enabled=enabled)
        self._model_names = model_names

    @property
    def model_names(self) -> list[str] | None:
        """Get the list of model names this callback responds to."""
        return self._model_names

    def _matches_model(self, model_name: str | None) -> bool:
        """Check if this callback should respond to the given model."""
        if self._model_names is None:
            return True
        if model_name is None:
            return True
        return model_name in self._model_names

    def on_llm_start(
        self,
        context: CallbackContext,
        messages: list[dict[str, Any]],
        config: Any | None,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        """
        Called before LLM is invoked.

        Override this method to modify or validate messages and config
        before the LLM call. The returned tuple contains modified messages
        and config.

        Args:
            context: Callback context with event info.
            messages: Messages being sent to LLM.
            config: ModelConfig being used for the call.

        Returns:
            Tuple of (modified_messages, modified_config).
        """
        return messages, config

    def on_llm_end(
        self,
        context: CallbackContext,
        messages: list[dict[str, Any]],
        response: Any,
    ) -> Any:
        """
        Called after LLM response is received.

        Override this method to modify or log the LLM response
        after execution.

        Args:
            context: Callback context with event info.
            messages: Messages that were sent to LLM.
            response: The LLM response.

        Returns:
            Modified response (can be transformed).
        """
        return response


class LoggingLLMCallback(LLMCallback):
    """
    Callback that logs LLM invocation events.

    Useful for debugging and monitoring LLM calls.

    Example:
        agent = BaseAgent(
            name="my_agent",
            system_prompt="...",
            callbacks=[LoggingLLMCallback()]
        )
    """

    def __init__(
        self,
        model_names: list[str] | None = None,
        log_messages: bool = True,
        log_responses: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        Initialize logging callback.

        Args:
            model_names: List of model names to log. None means all models.
            log_messages: Whether to log LLM messages.
            log_responses: Whether to log LLM responses.
            enabled: Whether this callback is active.
        """
        super().__init__(model_names=model_names, enabled=enabled)
        self._log_messages = log_messages
        self._log_responses = log_responses

    def on_llm_start(
        self,
        context: CallbackContext,
        messages: list[dict[str, Any]],
        config: Any | None,
    ) -> tuple[list[dict[str, Any]], Any | None]:
        """Log LLM call before invocation."""
        model_name = None
        if config and hasattr(config, "model"):
            model_name = config.model

        if not self._matches_model(model_name):
            return messages, config

        print(f"[Callback] {context.agent_name}: LLM call start")
        if self._log_messages and messages:
            print(f"[Callback] Messages ({len(messages)}):")
            for i, msg in enumerate(messages):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content_str = f"[{len(content)} items]"
                else:
                    content_str = (
                        content[:100] + "..." if len(content) > 100 else content
                    )
                print(f"[Callback]   [{i}] {role}: {content_str}")

        return messages, config

    def on_llm_end(
        self,
        context: CallbackContext,
        messages: list[dict[str, Any]],
        response: Any,
    ) -> Any:
        """Log LLM response after invocation."""
        model_name = None
        if hasattr(response, "model"):
            model_name = response.model

        if not self._matches_model(model_name):
            return response

        if self._log_responses:
            content = getattr(response, "content", None)
            content_str = (
                content[:200] + "..." if content and len(content) > 200 else content
            )
            print(f"[Callback] {context.agent_name}: LLM call end")
            if content_str:
                print(f"[Callback] Response: {content_str}")

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print(
                    f"[Callback] Usage: {usage.prompt_tokens}p / "
                    f"{usage.completion_tokens}c / {usage.total_tokens}t"
                )

        return response
