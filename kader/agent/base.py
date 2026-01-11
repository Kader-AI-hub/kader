"""
Base Agent Implementation.

Defines the BaseAgent class which serves as the foundation for creating specific agents
with tools, memory, and LLM provider integration.
"""

from pathlib import Path
from typing import AsyncIterator, Iterator, Optional, Union

import yaml
from tenacity import RetryError, stop_after_attempt, wait_exponential

from kader.memory import (
    ConversationManager,
    FileSessionManager,
    SlidingWindowConversationManager,
)
from kader.prompts.base import PromptBase
from kader.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    Message,
    ModelConfig,
    StreamChunk,
)
from kader.providers.ollama import OllamaProvider
from kader.tools import BaseTool, ToolRegistry

from .logger import agent_logger


class BaseAgent:
    """
    Base class for Agents.

    Combines tools, memory, and an LLM provider to perform tasks.
    Supports synchronous and asynchronous invocation and streaming.
    Includes built-in retry logic using tenacity.
    Supports session persistence via FileSessionManager.
    """

    def __init__(
        self,
        name: str,
        system_prompt: Union[str, PromptBase],
        tools: Union[list[BaseTool], ToolRegistry, None] = None,
        provider: Optional[BaseLLMProvider] = None,
        memory: Optional[ConversationManager] = None,
        retry_attempts: int = 3,
        model_name: str = "qwen3-coder:480b-cloud",
        session_id: Optional[str] = None,
        use_persistence: bool = False,
        interrupt_before_tool: bool = True,
        tool_confirmation_callback: Optional[callable] = None,
    ) -> None:
        """
        Initialize the Base Agent.

        Args:
            name: Name of the agent.
            system_prompt: The system prompt definition.
            tools: List of tools or a ToolRegistry.
            provider: LLM provider instance. If None, uses OllamaProvider.
            memory: Conversation/Memory manager. If None, uses SlidingWindowConversationManager.
            retry_attempts: Number of retry attempts for LLM calls (default: 3).
            model_name: Default model name if creating a default Ollama provider.
            session_id: Optional session ID to load/resume.
            use_persistence: If True, enables session persistence (auto-enabled if session_id provided).
            interrupt_before_tool: If True, pauses and asks for user confirmation before executing tools.
            tool_confirmation_callback: Optional callback function for tool confirmation.
                Signature: (message: str) -> tuple[bool, Optional[str]]
                Returns (should_execute, user_elaboration_if_declined).
        """
        self.name = name
        self.system_prompt = system_prompt
        self.retry_attempts = retry_attempts
        self.interrupt_before_tool = interrupt_before_tool
        self.tool_confirmation_callback = tool_confirmation_callback

        # Persistence Configuration
        self.session_id = session_id
        self.use_persistence = use_persistence or (session_id is not None)
        self.session_manager = FileSessionManager() if self.use_persistence else None

        # Initialize Logger if agent uses persistence (logs only if there's a session)
        self.logger_id = None
        if self.use_persistence:
            # Only create logger if we have a session_id or if the session manager will create one
            session_id_for_logger = self.session_id
            if not session_id_for_logger and self.session_manager:
                # If no session_id yet but persistence is enabled, we'll get one during _load_session
                pass  # We'll set up the logger in _load_session if needed
            if session_id_for_logger:
                self.logger_id = agent_logger.setup_logger(
                    self.name, session_id_for_logger
                )

        # Initialize Provider
        if provider:
            self.provider = provider
        else:
            self.provider = OllamaProvider(model=model_name)

        # Initialize Memory
        if memory:
            self.memory = memory
        else:
            self.memory = SlidingWindowConversationManager()

        # Initialize Tools
        self._tool_registry = ToolRegistry()
        if tools:
            if isinstance(tools, ToolRegistry):
                self._tool_registry = tools
            elif isinstance(tools, list):
                for tool in tools:
                    self._tool_registry.register(tool)

        if self.use_persistence:
            self._load_session()

        # Propagate session to tools
        self._propagate_session_to_tools()

        # Update config with tools if provider supports it
        self._update_provider_tools()

    def _load_session(self) -> None:
        """Load conversation history from session storage."""
        if not self.session_manager:
            return

        if not self.session_id:
            session = self.session_manager.create_session(self.name)
            self.session_id = session.session_id

        # Initialize logger if we now have a session_id and logging hasn't been set up yet
        if self.use_persistence and not self.logger_id and self.session_id:
            self.logger_id = agent_logger.setup_logger(self.name, self.session_id)

        # Propagate session to tools
        self._propagate_session_to_tools()

        # Load conversation history
        try:
            # We don't check if session exists first because load_conversation
            # handles missing sessions by returning empty list (usually)
            # or we catch the error. FileSessionManager.load_conversation returns list[dict].
            history = self.session_manager.load_conversation(self.session_id)
            if history:
                # Add loaded messages to memory
                # ConversationManager supports adding dicts directly
                self.memory.add_messages(history)
        except Exception:
            # If session doesn't exist or error, we start fresh (or could log warning)
            # For now, we silently proceed with empty memory
            pass

    def _propagate_session_to_tools(self) -> None:
        """Propagate current session ID to all registered tools."""
        if not self.session_id:
            return

        for tool in self._tool_registry.tools:
            if hasattr(tool, "set_session_id"):
                tool.set_session_id(self.session_id)

    def _save_session(self) -> None:
        """Save current conversation history to session storage."""
        if not self.session_manager or not self.session_id:
            return

        try:
            # Get all messages from memory
            # Convert ConversationMessage to dict (using .message property)
            messages = [msg.message for msg in self.memory.get_messages()]
            self.session_manager.save_conversation(self.session_id, messages)
        except Exception:
            # Log error or handle silently? Best not to crash main flow on save failure
            pass

    @property
    def tools_map(self) -> dict[str, BaseTool]:
        """
        Get a dictionary mapping tool names to tool instances.

        Returns:
            Dictionary of {tool_name: tool_instance}
        """
        # Access private attribute of ToolRegistry if needed, or iterate
        # ToolRegistry has .tools property which returns a list
        return {tool.name: tool for tool in self._tool_registry.tools}

    def _update_provider_tools(self) -> None:
        """Update the provider's default config with registered tools."""
        if not self._tool_registry.tools:
            return

        # We need to update the default config of the provider to include these tools
        # Since we can't easily modify the internal default_config of the provider cleanly
        # from here without accessing protected members, strict encapsulation might prevent this.
        # However, for this implementation, we will pass tools during invoke if they exist.
        pass

    def _get_run_config(self, config: Optional[ModelConfig] = None) -> ModelConfig:
        """Prepare execution config with tools."""
        base_config = config or ModelConfig()

        # If tools are available and not explicitly disabled or overridden
        if self._tool_registry.tools and not base_config.tools:
            # Detect provider type to format tools correctly
            # Defaulting to 'openai' format as it's the de-facto standard
            provider_type = "openai"
            if isinstance(self.provider, OllamaProvider):
                provider_type = "ollama"

            base_config = ModelConfig(
                temperature=base_config.temperature,
                max_tokens=base_config.max_tokens,
                top_p=base_config.top_p,
                top_k=base_config.top_k,
                frequency_penalty=base_config.frequency_penalty,
                stop_sequences=base_config.stop_sequences,
                stream=base_config.stream,
                tools=self._tool_registry.to_provider_format(provider_type),
                tool_choice=base_config.tool_choice,
                extra=base_config.extra,
            )

        return base_config

    def _prepare_messages(
        self, messages: Union[str, list[Message], list[dict]]
    ) -> list[Message]:
        """Prepare messages adding system prompt and history."""
        # Normalize input to list of Message objects
        input_msgs: list[Message] = []
        if isinstance(messages, str):
            input_msgs = [Message.user(messages)]
        elif isinstance(messages, list):
            if not messages:
                pass
            elif isinstance(messages[0], dict):
                # Convert dicts to Messages
                pass  # simplified for now, assuming user passes Message objects or string
                # But we should handle it better
                input_msgs = [
                    Message(**msg) if isinstance(msg, dict) else msg for msg in messages
                ]
            else:
                input_msgs = messages  # Assuming list[Message]

        # Add to memory
        for msg in input_msgs:
            self.memory.add_message(msg)

        # Retrieve context (system prompt + history)
        # 1. Start with System Prompt
        if isinstance(self.system_prompt, PromptBase):
            sys_prompt_content = self.system_prompt.resolve_prompt()
        else:
            sys_prompt_content = str(self.system_prompt)

        final_messages = [Message.system(sys_prompt_content)]

        # 2. Get history from memory (windowed)
        # memory.apply_window() returns list[dict], need to convert back to Message
        history_dicts = self.memory.apply_window()

        # We need to act smart here. invoke/stream usually take the *new* messages
        # plus history. Memory managers usually store everything.
        # If we added input_msgs to memory, apply_window should return them too if relevant.
        # So we just use what Memory gives us.

        for msg_dict in history_dicts:
            # Basic conversion from dict back to Message
            # Note: conversation.py Message support might be limited to dicts
            msg = Message(
                role=msg_dict.get("role"),
                content=msg_dict.get("content"),
                name=msg_dict.get("name"),
                tool_call_id=msg_dict.get("tool_call_id"),
                tool_calls=msg_dict.get("tool_calls"),
            )
            final_messages.append(msg)

        return final_messages

    def _format_tool_call_for_display(self, tool_call_dict: dict) -> str:
        """
        Format a tool call for display to the user.

        Args:
            tool_call_dict: The tool call dictionary from LLM response.

        Returns:
            The tool's interruption message.
        """
        import json

        fn_info = tool_call_dict.get("function", {})
        if not fn_info and "name" in tool_call_dict:
            fn_info = tool_call_dict

        tool_name = fn_info.get("name", "unknown")
        arguments = fn_info.get("arguments", {})

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass

        # Get the tool's interruption message if available
        tool = self._tool_registry.get(tool_name)
        if tool and isinstance(arguments, dict):
            return tool.get_interruption_message(**arguments)

        # Fallback for unknown tools
        return f"execute {tool_name}"

    def _confirm_tool_execution(
        self, tool_call_dict: dict, llm_content: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Ask user for confirmation before executing a tool.

        Args:
            tool_call_dict: The tool call dictionary from LLM response.

        Returns:
            Tuple of (should_execute: bool, user_input: Optional[str]).
            If should_execute is False, user_input contains additional context.
        """
        display_str = self._format_tool_call_for_display(tool_call_dict)

        if llm_content and len(llm_content) > 0:
            display_str = f"{llm_content}\n\n{display_str}"

        # Use callback if provided (e.g., for GUI/TUI)
        if self.tool_confirmation_callback:
            return self.tool_confirmation_callback(display_str)

        # Default: use console input
        print(display_str)

        while True:
            user_input = input("\nExecute this tool? (yes/no): ").strip().lower()

            if user_input in ("yes", "y"):
                return True, None
            elif user_input in ("no", "n"):
                elaboration = input(
                    "Please provide more context or instructions: "
                ).strip()
                return False, elaboration if elaboration else None
            else:
                print("Please enter 'yes' or 'no'.")

    async def _aconfirm_tool_execution(
        self, tool_call_dict: dict, llm_content: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """
        Async version - Ask user for confirmation before executing a tool.

        Note: This uses synchronous input() as async stdin is complex.
        For production use, consider using aioconsole or similar.

        Args:
            tool_call_dict: The tool call dictionary from LLM response.

        Returns:
            Tuple of (should_execute: bool, user_input: Optional[str]).
        """
        # For simplicity, we use the sync version in async context
        # In production, use asyncio.to_thread or aioconsole
        import asyncio

        return await asyncio.to_thread(
            self._confirm_tool_execution, tool_call_dict, llm_content
        )

    def _process_tool_calls(
        self, response: LLMResponse
    ) -> Union[list[Message], tuple[bool, str]]:
        """
        Execute tool calls from response and return tool messages.

        Args:
            response: The LLM response containing tool calls.

        Returns:
            List of Message objects representing tool results, or
            Tuple of (False, user_input) if user declined tool execution.
        """
        tool_messages = []
        if response.has_tool_calls:
            for tool_call_dict in response.tool_calls:
                # Check for interrupt before tool execution
                if self.interrupt_before_tool:
                    should_execute, user_input = self._confirm_tool_execution(
                        tool_call_dict, response.content
                    )
                    if not should_execute:
                        # Return the user's elaboration to be processed
                        return (False, user_input)

                # Need to convert dict to ToolCall object or handle manually
                # ToolRegistry.run takes ToolCall
                from kader.tools.base import ToolCall

                # Create ToolCall object
                # Some providers might differ in specific dict keys, relying on normalization
                try:
                    tool_call = ToolCall(
                        id=tool_call_dict.get("id", ""),
                        name=tool_call_dict.get("function", {}).get("name", ""),
                        arguments=tool_call_dict.get("function", {}).get(
                            "arguments", {}
                        ),
                        raw_arguments=str(
                            tool_call_dict.get("function", {}).get("arguments", {})
                        ),
                    )
                except Exception:
                    # Fallback or simplified parsing if structure differs
                    tool_call = ToolCall(
                        id=tool_call_dict.get("id", ""),
                        name=tool_call_dict.get("function", {}).get("name", ""),
                        arguments={},  # Error case
                    )

                # Execute tool
                tool_result = self._tool_registry.run(tool_call)

                # add result to memory
                # But here we just return messages, caller handles memory add
                tool_msg = Message.tool(
                    tool_call_id=tool_result.tool_call_id, content=tool_result.content
                )
                tool_messages.append(tool_msg)

        return tool_messages

    async def _aprocess_tool_calls(
        self, response: LLMResponse
    ) -> Union[list[Message], tuple[bool, str]]:
        """
        Async version of _process_tool_calls.

        Returns:
            List of Message objects representing tool results, or
            Tuple of (False, user_input) if user declined tool execution.
        """
        tool_messages = []
        if response.has_tool_calls:
            for tool_call_dict in response.tool_calls:
                # Check for interrupt before tool execution
                if self.interrupt_before_tool:
                    should_execute, user_input = await self._aconfirm_tool_execution(
                        tool_call_dict, response.content
                    )
                    if not should_execute:
                        return (False, user_input)

                from kader.tools.base import ToolCall

                # Check structure - Ollama/OpenAI usually: {'id':..., 'type': 'function', 'function': {'name':.., 'arguments':..}}
                fn_info = tool_call_dict.get("function", {})
                if not fn_info and "name" in tool_call_dict:
                    # Handle flat structure if any
                    fn_info = tool_call_dict

                tool_call = ToolCall(
                    id=tool_call_dict.get("id", "call_default"),
                    name=fn_info.get("name", ""),
                    arguments=fn_info.get("arguments", {}),
                )

                # Execute tool async
                tool_result = await self._tool_registry.arun(tool_call)

                tool_msg = Message.tool(
                    tool_call_id=tool_result.tool_call_id, content=tool_result.content
                )
                tool_messages.append(tool_msg)

        return tool_messages

    # -------------------------------------------------------------------------
    # Synchronous Methods
    # -------------------------------------------------------------------------

    def invoke(
        self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None
    ) -> LLMResponse:
        """
        Synchronously invoke the agent.

        Handles message preparation, LLM invocation with retries, and tool execution loop.
        """
        # Retry decorator wrapper logic
        # Since tenacity decorators wrap functions, we define an inner function or use the decorator on a method
        # but we want dynamic retry attempts (from self) which decorators strictly speaking don't support easily without specialized usage.
        # We will use the functional API of tenacity for dynamic configuration.
        from tenacity import Retrying

        runner = Retrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        )

        final_response = None

        # Main Agent Loop (Limit turns to avoid infinite loops)
        max_turns = 10
        current_turn = 0

        while current_turn < max_turns:
            current_turn += 1

            # Prepare full context
            full_history = self._prepare_messages(messages if current_turn == 1 else [])
            # Note: _prepare_messages adds input to memory. On subsequent turns (tools),
            # we don't re-add the user input. self.memory already has it + previous turns.

            # Call LLM with retry
            try:
                response = runner(
                    self.provider.invoke, full_history, self._get_run_config(config)
                )
            except RetryError as e:
                # Should not happen with reraise=True, but just in case
                raise e

            # Add assistant response to memory
            self.memory.add_message(response.to_message())

            # Log the interaction if logger is active
            if self.logger_id:
                # Extract token usage info if available
                token_usage = None
                if hasattr(response, "usage"):
                    token_usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }

                # Log the LLM response
                agent_logger.log_llm_response(self.logger_id, str(response.content))

                # Log token usage and calculate cost
                if token_usage:
                    agent_logger.log_token_usage(
                        self.logger_id,
                        token_usage["prompt_tokens"],
                        token_usage["completion_tokens"],
                        token_usage["total_tokens"],
                    )

                    # estimate the cost...
                    estimated_cost = self.provider.estimate_cost(token_usage)

                    # Calculate and log cost
                    agent_logger.calculate_cost(
                        self.logger_id,
                        estimated_cost.total_cost,
                    )

            # Save session update
            if self.use_persistence:
                self._save_session()

            # Check for tool calls
            if response.has_tool_calls:
                tool_result = self._process_tool_calls(response)

                # Check if user declined tool execution
                if isinstance(tool_result, tuple) and tool_result[0] is False:
                    # User declined - add their input as a new message and continue
                    user_elaboration = tool_result[1]
                    if user_elaboration:
                        self.memory.add_message(Message.user(user_elaboration))
                    else:
                        # User provided no elaboration, return current response
                        final_response = response
                        break
                    continue

                tool_msgs = tool_result

                # Add tool outputs to memory
                for tm in tool_msgs:
                    self.memory.add_message(tm)

                    # Log tool usage
                    if self.logger_id:
                        # Extract tool name and arguments
                        tool_name = "unknown"
                        arguments = {}
                        if hasattr(tm, "tool_call_id"):
                            # This is a tool message, need to find the tool name
                            # We'll check the original response to find the tool
                            for tool_call in response.tool_calls:
                                fn_info = tool_call.get("function", {})
                                if fn_info.get("name"):
                                    tool_name = fn_info.get("name", "unknown")
                                    arguments = fn_info.get("arguments", {})
                                    agent_logger.log_tool_usage(
                                        self.logger_id, tool_name, arguments
                                    )
                                    break

                # Save session update after tool results
                if self.use_persistence:
                    self._save_session()

                # Loop continues to feed tool outputs back to LLM
                continue
            else:
                # No tools, final response
                final_response = response
                break

        return final_response

    def stream(
        self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None
    ) -> Iterator[StreamChunk]:
        """
        Synchronously stream the agent response.

        Note: Tool execution breaks streaming flow typically.
        If tools are called, we consume the stream to execute tools, then stream the final answer.
        """
        # For simplicity in this base implementation, we'll only stream if there are no tool calls initially,
        # or we buffer if we detect tools. Logic can get complex.

        # Current simplified approach:
        # 1. Prepare messages
        full_history = self._prepare_messages(messages)

        # 2. Stream from provider
        # We need to handle retries for the *start* of the stream
        from tenacity import Retrying

        runner = Retrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        )

        # We can't retry the *iteration* easily if it fails mid-stream without complex logic.
        # We will retry obtaining the iterator.
        stream_iterator = runner(
            self.provider.stream, full_history, self._get_run_config(config)
        )

        yield from stream_iterator

        # Update session at end if needed
        # Note: Streaming complicates memory/persistence because getting the full message
        # requires aggregating chunks. The current implementation of base.stream DOES NOT
        # auto-aggregate into memory (it just yields).
        # The USER of stream() is responsible for re-assembling the message and adding to memory
        # if they want history.
        # BUT, wait. _prepare_messages DOES add input messages to memory.
        # The RESPONSE is not added here.
        # TODO: A robust stream implementation should aggregate and save.
        # For now, we only save the input part since _prepare_messages called it.
        if self.use_persistence:
            self._save_session()

    # -------------------------------------------------------------------------
    # Asynchronous Methods
    # -------------------------------------------------------------------------

    async def ainvoke(
        self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None
    ) -> LLMResponse:
        """Asynchronous invocation with retries and tool loop."""
        from tenacity import AsyncRetrying

        runner = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True,
        )

        max_turns = 10
        current_turn = 0
        final_response = None

        while current_turn < max_turns:
            current_turn += 1
            full_history = self._prepare_messages(messages if current_turn == 1 else [])

            response = await runner(
                self.provider.ainvoke, full_history, self._get_run_config(config)
            )

            self.memory.add_message(response.to_message())

            # Log the interaction if logger is active
            if self.logger_id:
                # Extract token usage info if available
                token_usage = None
                if hasattr(response, "usage"):
                    token_usage = {
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(
                            response.usage, "completion_tokens", 0
                        ),
                        "total_tokens": getattr(response.usage, "total_tokens", 0),
                    }

                # Log the LLM response
                agent_logger.log_llm_response(self.logger_id, str(response.content))

                # Log token usage and calculate cost
                if token_usage:
                    agent_logger.log_token_usage(
                        self.logger_id,
                        token_usage["prompt_tokens"],
                        token_usage["completion_tokens"],
                        token_usage["total_tokens"],
                    )

                    # estimate the cost...
                    estimated_cost = self.provider.estimate_cost(token_usage)

                    # Calculate and log cost
                    agent_logger.calculate_cost(
                        self.logger_id,
                        estimated_cost.total_cost,
                    )

            # Save session update
            if self.use_persistence:
                self._save_session()

            if response.has_tool_calls:
                tool_result = await self._aprocess_tool_calls(response)

                # Check if user declined tool execution
                if isinstance(tool_result, tuple) and tool_result[0] is False:
                    # User declined - add their input as a new message and continue
                    user_elaboration = tool_result[1]
                    if user_elaboration:
                        self.memory.add_message(Message.user(user_elaboration))
                    else:
                        final_response = response
                        break
                    continue

                tool_msgs = tool_result

                for tm in tool_msgs:
                    self.memory.add_message(tm)

                    # Log tool usage
                    if self.logger_id:
                        # Extract tool name and arguments
                        tool_name = "unknown"
                        arguments = {}
                        if hasattr(tm, "tool_call_id"):
                            # This is a tool message, need to find the tool name
                            # We'll check the original response to find the tool
                            for tool_call in response.tool_calls:
                                fn_info = tool_call.get("function", {})
                                if fn_info.get("name"):
                                    tool_name = fn_info.get("name", "unknown")
                                    arguments = fn_info.get("arguments", {})
                                    agent_logger.log_tool_usage(
                                        self.logger_id, tool_name, arguments
                                    )
                                    break

                # Save session update
                if self.use_persistence:
                    self._save_session()
                continue
            else:
                final_response = response
                break

        return final_response

    async def astream(
        self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None
    ) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming with memory aggregation."""
        # Prepare messages
        full_history = self._prepare_messages(messages)

        # Determine config
        run_config = self._get_run_config(config)

        # Get stream iterator directly (cannot use tenacity on async generator creation easily)
        stream_iterator = self.provider.astream(full_history, run_config)

        aggregated_content = ""
        aggregated_tool_calls = []

        async for chunk in stream_iterator:
            aggregated_content += chunk.content
            if chunk.tool_calls:
                # TODO: robust tool call aggregation if streaming partial JSON
                # For now, assume provider yields complete tool calls in chunks or we just collect them
                aggregated_tool_calls.extend(chunk.tool_calls)
            yield chunk

        # Create Message and add to memory
        # Note: If no content and no tools, we don't add (or adds empty message)

        # If we have tool calls, we might need to properly format them
        final_msg = Message(
            role="assistant",
            content=aggregated_content,
            tool_calls=aggregated_tool_calls if aggregated_tool_calls else None,
        )

        self.memory.add_message(final_msg)

        if self.use_persistence:
            self._save_session()

    # -------------------------------------------------------------------------
    # Serialization Methods
    # -------------------------------------------------------------------------

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Serialize agent configuration to YAML.

        Args:
           path: File path to save YAML.
        """
        system_prompt_str = (
            self.system_prompt.resolve_prompt()
            if isinstance(self.system_prompt, PromptBase)
            else str(self.system_prompt)
        )
        data = {
            "name": self.name,
            "system_prompt": system_prompt_str,
            "retry_attempts": self.retry_attempts,
            "provider": {
                "model": self.provider.model,
                # Add other provider settings if possible
            },
            "tools": self._tool_registry.names,
            # Memory state could be saved here too, but usually configured separately
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(
        cls, path: Union[str, Path], tool_registry: Optional[ToolRegistry] = None
    ) -> "BaseAgent":
        """
        Load agent from YAML configuration.

        Args:
            path: Path to YAML file.
            tool_registry: Registry containing *available* tools to re-hydrate the agent.
                           The agent's tools will be selected from this registry based on names in YAML.

        Returns:
            Instantiated BaseAgent.
        """
        path_obj = Path(path)
        with open(path_obj, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        name = data.get("name", "unnamed_agent")
        system_prompt = data.get("system_prompt", "")
        retry_attempts = data.get("retry_attempts", 3)
        provider_config = data.get("provider", {})
        model_name = provider_config.get("model", "qwen3-coder:480b-cloud")

        # Handle persistence parameter
        # If persistence key exists and is True, use_persistence is True
        # If persistence key exists and is False, use_persistence is False
        # If persistence key doesn't exist, use_persistence defaults to True
        use_persistence = data.get("persistence", True)

        # Reconstruct tools
        tools = []
        tool_names = data.get("tools", [])

        # Use provided registry or fallback to default
        registry = tool_registry
        if registry is None:
            # Lazy import to avoid circular dependencies if any
            try:
                from kader.tools import get_default_registry

                registry = get_default_registry()
            except ImportError:
                pass

        if tool_names and registry:
            for t_name in tool_names:
                t = registry.get(t_name)
                if t:
                    tools.append(t)

        return cls(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            retry_attempts=retry_attempts,
            model_name=model_name,
            use_persistence=use_persistence,
        )
