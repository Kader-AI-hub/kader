"""
Base class for Agentic Tools.

A versatile, provider-agnostic base class for defining tools that can be used
with any LLM provider (OpenAI, Google, Anthropic, Mistral, and others).
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypeAlias,
    TypeVar,
)

# Type Aliases
ParameterType: TypeAlias = Literal[
    "string", "integer", "number", "boolean", "array", "object"
]
ToolResultStatus: TypeAlias = Literal["success", "error", "pending"]


class ToolCategory(str, Enum):
    """Categories of tools for organization and filtering."""

    FILE_SYSTEM = "file_system"
    CODE = "code"
    WEB = "web"
    SEARCH = "search"
    DATABASE = "database"
    API = "api"
    UTILITY = "utility"
    CUSTOM = "custom"


@dataclass
class ParameterSchema:
    """Schema for a single tool parameter."""

    name: str
    type: ParameterType
    description: str
    required: bool = True

    # Additional constraints
    enum: list[str] | None = None
    default: Any = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None

    # For array types
    items_type: ParameterType | None = None

    # For object types
    properties: list["ParameterSchema"] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format (OpenAI/standard format)."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern

        # Array items
        if self.type == "array" and self.items_type:
            schema["items"] = {"type": self.items_type}

        # Nested object properties
        if self.type == "object" and self.properties:
            schema["properties"] = {
                prop.name: prop.to_json_schema() for prop in self.properties
            }
            schema["required"] = [
                prop.name for prop in self.properties if prop.required
            ]

        return schema


@dataclass
class ToolSchema:
    """Complete schema definition for a tool."""

    name: str
    description: str
    parameters: list[ParameterSchema] = field(default_factory=list)

    # Optional metadata
    category: ToolCategory = ToolCategory.CUSTOM
    version: str = "1.0.0"
    deprecated: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema format for parameters."""
        properties = {param.name: param.to_json_schema() for param in self.parameters}
        required = [param.name for param in self.parameters if param.required]

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.to_json_schema(),
        }

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google (Gemini) tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.to_json_schema(),
        }

    def to_mistral_format(self) -> dict[str, Any]:
        """Convert to Mistral tool format (same as OpenAI)."""
        return self.to_openai_format()

    def to_ollama_format(self) -> dict[str, Any]:
        """Convert to Ollama tool format (same as OpenAI)."""
        return self.to_openai_format()

    def to_provider_format(self, provider: str) -> dict[str, Any]:
        """
        Convert to a specific provider's format.

        Args:
            provider: Provider name (openai, anthropic, google, mistral, ollama)

        Returns:
            Tool schema in the provider's format
        """
        formatters = {
            "openai": self.to_openai_format,
            "anthropic": self.to_anthropic_format,
            "google": self.to_google_format,
            "gemini": self.to_google_format,
            "mistral": self.to_mistral_format,
            "ollama": self.to_ollama_format,
        }

        formatter = formatters.get(provider.lower())
        if formatter:
            return formatter()

        # Default to OpenAI format as it's most common
        return self.to_openai_format()


@dataclass
class ToolCall:
    """Represents a tool call from an LLM."""

    id: str  # Unique identifier for the tool call
    name: str  # Name of the tool to call
    arguments: dict[str, Any]  # Parsed arguments
    raw_arguments: str | None = None  # Original JSON string (if available)

    @classmethod
    def from_openai(cls, tool_call: dict[str, Any]) -> "ToolCall":
        """Create from OpenAI tool call format."""
        function = tool_call.get("function", {})
        raw_args = function.get("arguments", "{}")
        return cls(
            id=tool_call.get("id", ""),
            name=function.get("name", ""),
            arguments=json.loads(raw_args) if raw_args else {},
            raw_arguments=raw_args,
        )

    @classmethod
    def from_anthropic(cls, tool_use: dict[str, Any]) -> "ToolCall":
        """Create from Anthropic tool use format."""
        return cls(
            id=tool_use.get("id", ""),
            name=tool_use.get("name", ""),
            arguments=tool_use.get("input", {}),
            raw_arguments=json.dumps(tool_use.get("input", {})),
        )

    @classmethod
    def from_google(cls, function_call: dict[str, Any]) -> "ToolCall":
        """Create from Google (Gemini) function call format."""
        return cls(
            id=function_call.get("id", ""),
            name=function_call.get("name", ""),
            arguments=function_call.get("args", {}),
            raw_arguments=json.dumps(function_call.get("args", {})),
        )

    @classmethod
    def from_provider(cls, tool_call: dict[str, Any], provider: str) -> "ToolCall":
        """
        Create from a specific provider's format.

        Args:
            tool_call: Tool call data from the provider
            provider: Provider name

        Returns:
            Normalized ToolCall instance
        """
        parsers = {
            "openai": cls.from_openai,
            "anthropic": cls.from_anthropic,
            "google": cls.from_google,
            "gemini": cls.from_google,
            "mistral": cls.from_openai,  # Mistral uses OpenAI format
            "ollama": cls.from_openai,  # Ollama uses OpenAI format
        }

        parser = parsers.get(provider.lower())
        if parser:
            return parser(tool_call)

        # Default to OpenAI format
        return cls.from_openai(tool_call)


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_call_id: str  # ID of the tool call this result is for
    content: str  # String content of the result
    status: ToolResultStatus = "success"

    # Structured data (optional)
    data: Any = None

    # Error information (if status is "error")
    error_type: str | None = None
    error_message: str | None = None

    # Execution metadata
    execution_time_ms: float | None = None

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI tool result format."""
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool result format."""
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": self.tool_call_id,
            "content": self.content,
        }
        if self.status == "error":
            result["is_error"] = True
        return result

    def to_google_format(self) -> dict[str, Any]:
        """Convert to Google (Gemini) function response format."""
        return {
            "function_response": {
                "name": "",  # Needs to be filled by the caller
                "response": {
                    "content": self.content,
                    "status": self.status,
                },
            },
        }

    def to_provider_format(self, provider: str) -> dict[str, Any]:
        """
        Convert to a specific provider's format.

        Args:
            provider: Provider name

        Returns:
            Tool result in the provider's format
        """
        formatters = {
            "openai": self.to_openai_format,
            "anthropic": self.to_anthropic_format,
            "google": self.to_google_format,
            "gemini": self.to_google_format,
            "mistral": self.to_openai_format,
            "ollama": self.to_openai_format,
        }

        formatter = formatters.get(provider.lower())
        if formatter:
            return formatter()

        return self.to_openai_format()

    @classmethod
    def success(cls, tool_call_id: str, content: str, data: Any = None) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            tool_call_id=tool_call_id,
            content=content,
            status="success",
            data=data,
        )

    @classmethod
    def error(
        cls,
        tool_call_id: str,
        error_message: str,
        error_type: str = "ExecutionError",
    ) -> "ToolResult":
        """Create an error tool result."""
        return cls(
            tool_call_id=tool_call_id,
            content=f"Error: {error_message}",
            status="error",
            error_type=error_type,
            error_message=error_message,
        )


# Type variable for tool return types
T = TypeVar("T")


class BaseTool(ABC, Generic[T]):
    """
    Abstract base class for agentic tools.

    Provides a unified interface for defining tools that can be used with
    any LLM provider including OpenAI, Google, Anthropic, Mistral, and others.

    Subclasses must implement:
    - execute: Synchronous tool execution
    - aexecute: Asynchronous tool execution

    Example:
        class ReadFileTool(BaseTool[str]):
            def __init__(self):
                super().__init__(
                    name="read_file",
                    description="Read the contents of a file",
                    parameters=[
                        ParameterSchema(
                            name="path",
                            type="string",
                            description="Path to the file to read",
                        ),
                    ],
                )

            def execute(self, path: str) -> str:
                with open(path, "r") as f:
                    return f.read()
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: list[ParameterSchema] | None = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize the tool.

        Args:
            name: Unique name for the tool (used in function calls)
            description: Human-readable description of what the tool does
            parameters: List of parameter schemas
            category: Category for organization
            version: Version string for the tool
        """
        self._schema = ToolSchema(
            name=name,
            description=description,
            parameters=parameters or [],
            category=category,
            version=version,
        )

        # Execution tracking
        self._execution_count = 0
        self._total_execution_time_ms = 0.0
        self._last_execution_time_ms: float | None = None

        # Session Context
        self._session_id: str | None = None

    def set_session_id(self, session_id: str) -> None:
        """
        Set the session ID for the tool.

        Args:
            session_id: The session ID to associate with this tool instance.
        """
        self._session_id = session_id

    @property
    def name(self) -> str:
        """Get the tool name."""
        return self._schema.name

    @property
    def description(self) -> str:
        """Get the tool description."""
        return self._schema.description

    @property
    def schema(self) -> ToolSchema:
        """Get the full tool schema."""
        return self._schema

    @property
    def execution_count(self) -> int:
        """Get the total number of executions."""
        return self._execution_count

    @property
    def average_execution_time_ms(self) -> float:
        """Get the average execution time in milliseconds."""
        if self._execution_count == 0:
            return 0.0
        return self._total_execution_time_ms / self._execution_count

    def to_provider_format(self, provider: str) -> dict[str, Any]:
        """
        Get the tool definition in a specific provider's format.

        Args:
            provider: Provider name (openai, anthropic, google, mistral, ollama)

        Returns:
            Tool definition in the provider's format
        """
        return self._schema.to_provider_format(provider)

    def validate_arguments(self, arguments: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate the provided arguments against the schema.

        Args:
            arguments: Dictionary of argument name to value

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        # Check required parameters
        for param in self._schema.parameters:
            if param.required and param.name not in arguments:
                errors.append(f"Missing required parameter: {param.name}")

        # Check parameter types (basic validation)
        for param in self._schema.parameters:
            if param.name not in arguments:
                continue

            value = arguments[param.name]

            # Type checking
            type_checks = {
                "string": lambda v: isinstance(v, str),
                "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float))
                and not isinstance(v, bool),
                "boolean": lambda v: isinstance(v, bool),
                "array": lambda v: isinstance(v, list),
                "object": lambda v: isinstance(v, dict),
            }

            checker = type_checks.get(param.type)
            if checker and not checker(value):
                errors.append(
                    f"Parameter '{param.name}' should be {param.type}, got {type(value).__name__}"
                )

        return len(errors) == 0, errors

    def _update_tracking(self, execution_time_ms: float) -> None:
        """Update execution tracking metrics."""
        self._execution_count += 1
        self._total_execution_time_ms += execution_time_ms
        self._last_execution_time_ms = execution_time_ms

    def reset_tracking(self) -> None:
        """Reset execution tracking metrics."""
        self._execution_count = 0
        self._total_execution_time_ms = 0.0
        self._last_execution_time_ms = None

    # -------------------------------------------------------------------------
    # Abstract Methods - Must be implemented by subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def execute(self, **kwargs: Any) -> T:
        """
        Synchronously execute the tool.

        Args:
            **kwargs: Tool arguments matching the parameter schema

        Returns:
            The tool's result
        """
        ...

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> T:
        """
        Asynchronously execute the tool.

        Args:
            **kwargs: Tool arguments matching the parameter schema

        Returns:
            The tool's result
        """
        ...

    @abstractmethod
    def get_interruption_message(self, **kwargs: Any) -> str:
        """
        Get a human-readable message describing the tool action for user confirmation.

        This method should return a message that clearly describes what the tool
        is about to do, suitable for displaying to the user before execution.

        Args:
            **kwargs: Tool arguments matching the parameter schema

        Returns:
            A formatted string describing the action, e.g., "execute read_file: /path/to/file"
        """
        ...

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def run(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute the tool from a ToolCall and return a ToolResult.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with the execution result
        """
        import time

        start_time = time.perf_counter()

        try:
            # Validate arguments
            is_valid, errors = self.validate_arguments(tool_call.arguments)
            if not is_valid:
                return ToolResult.error(
                    tool_call_id=tool_call.id,
                    error_message="; ".join(errors),
                    error_type="ValidationError",
                )

            # Execute the tool
            result = self.execute(**tool_call.arguments)

            # Calculate execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_tracking(execution_time_ms)

            # Convert result to string if needed
            content = result if isinstance(result, str) else json.dumps(result)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                status="success",
                data=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_tracking(execution_time_ms)

            return ToolResult.error(
                tool_call_id=tool_call.id,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    async def arun(self, tool_call: ToolCall) -> ToolResult:
        """
        Asynchronously execute the tool from a ToolCall.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with the execution result
        """
        import time

        start_time = time.perf_counter()

        try:
            # Validate arguments
            is_valid, errors = self.validate_arguments(tool_call.arguments)
            if not is_valid:
                return ToolResult.error(
                    tool_call_id=tool_call.id,
                    error_message="; ".join(errors),
                    error_type="ValidationError",
                )

            # Execute the tool asynchronously
            result = await self.aexecute(**tool_call.arguments)

            # Calculate execution time
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_tracking(execution_time_ms)

            # Convert result to string if needed
            content = result if isinstance(result, str) else json.dumps(result)

            return ToolResult(
                tool_call_id=tool_call.id,
                content=content,
                status="success",
                data=result,
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            self._update_tracking(execution_time_ms)

            return ToolResult.error(
                tool_call_id=tool_call.id,
                error_message=str(e),
                error_type=type(e).__name__,
            )

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"


class ToolRegistry:
    """
    Registry for managing multiple tools.

    Provides a central location to register, retrieve, and manage tools.

    Example:
        registry = ToolRegistry()
        registry.register(ReadFileTool())
        registry.register(WriteFileTool())

        tools = registry.to_provider_format("openai")
        tool = registry.get("read_file")
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool.

        Args:
            tool: The tool to register

        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Name of the tool to unregister

        Returns:
            True if the tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> BaseTool | None:
        """
        Get a tool by name.

        Args:
            name: Name of the tool

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)

    def get_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: The category to filter by

        Returns:
            List of tools in the category
        """
        return [
            tool for tool in self._tools.values() if tool.schema.category == category
        ]

    @property
    def tools(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    @property
    def names(self) -> list[str]:
        """Get all registered tool names."""
        return list(self._tools.keys())

    def to_provider_format(self, provider: str) -> list[dict[str, Any]]:
        """
        Get all tools in a specific provider's format.

        Args:
            provider: Provider name

        Returns:
            List of tool definitions in the provider's format
        """
        return [tool.to_provider_format(provider) for tool in self._tools.values()]

    def run(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call using the registry.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with the execution result
        """
        tool = self.get(tool_call.name)
        if tool is None:
            return ToolResult.error(
                tool_call_id=tool_call.id,
                error_message=f"Tool '{tool_call.name}' not found",
                error_type="ToolNotFoundError",
            )
        return tool.run(tool_call)

    async def arun(self, tool_call: ToolCall) -> ToolResult:
        """
        Asynchronously execute a tool call using the registry.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with the execution result
        """
        tool = self.get(tool_call.name)
        if tool is None:
            return ToolResult.error(
                tool_call_id=tool_call.id,
                error_message=f"Tool '{tool_call.name}' not found",
                error_type="ToolNotFoundError",
            )
        return await tool.arun(tool_call)

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over registered tools."""
        return iter(self._tools.values())

    def __repr__(self) -> str:
        """String representation of the registry."""
        return f"ToolRegistry(tools={list(self._tools.keys())})"


def tool(
    name: str | None = None,
    description: str | None = None,
    parameters: list[ParameterSchema] | None = None,
    category: ToolCategory = ToolCategory.CUSTOM,
) -> Callable[[Callable[..., T]], "FunctionTool[T]"]:
    """
    Decorator to create a tool from a function.

    Example:
        @tool(
            name="greet",
            description="Greet a user by name",
            parameters=[
                ParameterSchema(name="name", type="string", description="Name to greet"),
            ],
        )
        def greet(name: str) -> str:
            return f"Hello, {name}!"

    Args:
        name: Tool name (defaults to function name)
        description: Tool description (defaults to function docstring)
        parameters: Parameter schemas
        category: Tool category

    Returns:
        Decorator that creates a FunctionTool
    """

    def decorator(func: Callable[..., T]) -> "FunctionTool[T]":
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {tool_name}"

        return FunctionTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            category=category,
            func=func,
        )

    return decorator


class FunctionTool(BaseTool[T]):
    """
    A tool created from a function.

    This is used by the @tool decorator to wrap functions as tools.
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable[..., T],
        parameters: list[ParameterSchema] | None = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize a function-based tool.

        Args:
            name: Tool name
            description: Tool description
            func: The function to wrap
            parameters: Parameter schemas
            category: Tool category
            version: Tool version
        """
        super().__init__(
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            version=version,
        )
        self._func = func

    def execute(self, **kwargs: Any) -> T:
        """Execute the wrapped function synchronously."""
        return self._func(**kwargs)

    async def aexecute(self, **kwargs: Any) -> T:
        """
        Execute the wrapped function asynchronously.

        If the function is a coroutine, it will be awaited.
        Otherwise, it will be run in a thread pool.
        """
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(self._func):
            return await self._func(**kwargs)
        else:
            return await asyncio.to_thread(self._func, **kwargs)

    def get_interruption_message(self, **kwargs: Any) -> str:
        """
        Get interruption message for user confirmation.

        For function-based tools, generates a message using the tool name
        and the first string argument value (if any).
        """
        # Try to find a meaningful argument to display
        for key, value in kwargs.items():
            if isinstance(value, str) and value:
                return f"execute {self.name}: {value}"

        # Fallback to just the tool name
        return f"execute {self.name}"
