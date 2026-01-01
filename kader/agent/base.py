"""
Base Agent Implementation.

Defines the BaseAgent class which serves as the foundation for creating specific agents
with tools, memory, and LLM provider integration.
"""

import yaml
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Union, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

from kader.providers.base import (
    BaseLLMProvider,
    Message,
    ModelConfig,
    LLMResponse,
    StreamChunk,
)
from kader.providers.ollama import OllamaProvider
from kader.tools import BaseTool, ToolRegistry, ToolResult
from kader.memory import ConversationManager, NullConversationManager, SlidingWindowConversationManager
from kader.prompts.base import PromptBase


class BaseAgent:
    """
    Base class for Agents.
    
    Combines tools, memory, and an LLM provider to perform tasks.
    Supports synchronous and asynchronous invocation and streaming.
    Includes built-in retry logic using tenacity.
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: Union[str, PromptBase],
        tools: Union[list[BaseTool], ToolRegistry, None] = None,
        provider: Optional[BaseLLMProvider] = None,
        memory: Optional[ConversationManager] = None,
        retry_attempts: int = 3,
        model_name: str = "gpt-oss:120b-cloud",
    ) -> None:
        """
        Initialize the Base Agent.
        
        Args:
            name: Name of the agent.
            system_prompt: The system prompt definition.
            tools: List of tools or a ToolRegistry.
            provider: LLM provider instance. If None, uses OllamaProvider.
            memory: Conversation/Memory manager. If None, uses NullConversationManager.
            retry_attempts: Number of retry attempts for LLM calls (default: 3).
            model_name: Default model name if creating a default Ollama provider.
        """
        self.name = name
        self.system_prompt = system_prompt
        self.retry_attempts = retry_attempts
        
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
                    
        # Update config with tools if provider supports it
        self._update_provider_tools()

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
            
        # Get tool schemas in the format expected by the provider
        # Note: BaseLLMProvider generic interface uses OpenAI/Standard format typically
        # But we can try to be specific if we know the provider type
        # For now, we use the standard OpenAI format which most providers support
        provider_tools = self._tool_registry.to_provider_format("openai")
        
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
                extra=base_config.extra
            )
        
        return base_config

    def _prepare_messages(self, messages: Union[str, list[Message], list[dict]]) -> list[Message]:
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
                 pass # simplified for now, assuming user passes Message objects or string
                 # But we should handle it better
                 input_msgs = [
                     Message(**msg) if isinstance(msg, dict) else msg 
                     for msg in messages
                 ]
            else:
                input_msgs = messages # Assuming list[Message]

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
                tool_calls=msg_dict.get("tool_calls")
            )
            final_messages.append(msg)
            
        return final_messages

    def _process_tool_calls(self, response: LLMResponse) -> list[Message]:
        """
        Execute tool calls from response and return tool messages.
        
        Args:
            response: The LLM response containing tool calls.
            
        Returns:
            List of Message objects representing tool results.
        """
        tool_messages = []
        if response.has_tool_calls:
            for tool_call_dict in response.tool_calls:
                # Need to convert dict to ToolCall object or handle manually
                # ToolRegistry.run takes ToolCall
                from kader.tools.base import ToolCall
                
                # Create ToolCall object
                # Some providers might differ in specific dict keys, relying on normalization
                try:
                    tool_call = ToolCall(
                        id=tool_call_dict.get("id", ""),
                        name=tool_call_dict.get("function", {}).get("name", ""),
                        arguments=tool_call_dict.get("function", {}).get("arguments", {}),
                        raw_arguments=str(tool_call_dict.get("function", {}).get("arguments", {}))
                    )
                except Exception:
                    # Fallback or simplified parsing if structure differs
                    tool_call = ToolCall(
                        id=tool_call_dict.get("id", ""),
                        name=tool_call_dict.get("function", {}).get("name", ""),
                        arguments={}, # Error case
                    )
                
                # Execute tool
                tool_result = self._tool_registry.run(tool_call)
                
                # add result to memory
                # But here we just return messages, caller handles memory add
                tool_msg = Message.tool(
                    tool_call_id=tool_result.tool_call_id,
                    content=tool_result.content
                )
                tool_messages.append(tool_msg)
                
        return tool_messages

    async def _aprocess_tool_calls(self, response: LLMResponse) -> list[Message]:
        """Async version of _process_tool_calls."""
        tool_messages = []
        if response.has_tool_calls:
            for tool_call_dict in response.tool_calls:
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
                    tool_call_id=tool_result.tool_call_id,
                    content=tool_result.content
                )
                tool_messages.append(tool_msg)
                
        return tool_messages

    # -------------------------------------------------------------------------
    # Synchronous Methods
    # -------------------------------------------------------------------------

    def invoke(self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None) -> LLMResponse:
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
            reraise=True
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
                    self.provider.invoke,
                    full_history,
                    self._get_run_config(config)
                )
            except RetryError as e:
                # Should not happen with reraise=True, but just in case
                raise e
            
            # Add assistant response to memory
            self.memory.add_message(response.to_message())
            
            # Check for tool calls
            if response.has_tool_calls:
                tool_msgs = self._process_tool_calls(response)
                
                # Add tool outputs to memory
                for tm in tool_msgs:
                    self.memory.add_message(tm)
                    
                # Loop continues to feed tool outputs back to LLM
                continue
            else:
                # No tools, final response
                final_response = response
                break
                
        return final_response

    def stream(self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None) -> Iterator[StreamChunk]:
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
            reraise=True
        )
        
        # We can't retry the *iteration* easily if it fails mid-stream without complex logic.
        # We will retry obtaining the iterator.
        stream_iterator = runner(
            self.provider.stream,
            full_history,
            self._get_run_config(config)
        )
        
        yield from stream_iterator
        # Note: Proper tool handling in stream requires buffering the stream to check for tool_calls, 
        # executing them, and then re-invoking. This is advanced for a basic stream implementation.
        # This implementation assumes direct streaming response.

    # -------------------------------------------------------------------------
    # Asynchronous Methods
    # -------------------------------------------------------------------------

    async def ainvoke(self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None) -> LLMResponse:
        """Asynchronous invocation with retries and tool loop."""
        from tenacity import AsyncRetrying
        
        runner = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True
        )
        
        max_turns = 10
        current_turn = 0
        final_response = None
        
        while current_turn < max_turns:
            current_turn += 1
            full_history = self._prepare_messages(messages if current_turn == 1 else [])
            
            response = await runner(
                self.provider.ainvoke,
                full_history,
                self._get_run_config(config)
            )
            
            self.memory.add_message(response.to_message())
            
            if response.has_tool_calls:
                tool_msgs = await self._aprocess_tool_calls(response)
                for tm in tool_msgs:
                    self.memory.add_message(tm)
                continue
            else:
                final_response = response
                break
                
        return final_response

    async def astream(self, messages: Union[str, list[Message]], config: Optional[ModelConfig] = None) -> AsyncIterator[StreamChunk]:
        """Asynchronous streaming."""
        from tenacity import AsyncRetrying
        
        runner = AsyncRetrying(
            stop=stop_after_attempt(self.retry_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=10),
            reraise=True
        )
        
        full_history = self._prepare_messages(messages)
        
        stream_iterator = await runner(
            self.provider.astream,
            full_history,
            self._get_run_config(config)
        )
        
        async for chunk in stream_iterator:
            yield chunk

    # -------------------------------------------------------------------------
    # Serialization Methods
    # -------------------------------------------------------------------------

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Serialize agent configuration to YAML.
        
        Args:
           path: File path to save YAML.
        """
        system_prompt_str = self.system_prompt.resolve_prompt() if isinstance(self.system_prompt, PromptBase) else str(self.system_prompt)
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
    def from_yaml(cls, path: Union[str, Path], tool_registry: Optional[ToolRegistry] = None) -> "BaseAgent":
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
        model_name = provider_config.get("model", "gpt-oss:120b-cloud")
        
        # Reconstruct tools
        tools = []
        tool_names = data.get("tools", [])
        if tool_names and tool_registry:
            for t_name in tool_names:
                t = tool_registry.get(t_name)
                if t:
                    tools.append(t)
        
        return cls(
            name=name,
            system_prompt=system_prompt,
            tools=tools,
            retry_attempts=retry_attempts,
            model_name=model_name
        )
