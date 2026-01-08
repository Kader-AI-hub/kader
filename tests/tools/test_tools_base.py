"""
Unit tests for the base tool functionality.
"""

from typing import Any

import pytest

from kader.tools.base import (
    BaseTool,
    ParameterSchema,
    ToolCall,
    ToolCategory,
    ToolRegistry,
    ToolResult,
    ToolSchema,
    tool,
)


class ConcreteTool(BaseTool[str]):
    """Concrete implementation of BaseTool for testing."""

    def __init__(self):
        super().__init__(
            name="test_tool",
            description="A test tool",
            parameters=[
                ParameterSchema(
                    name="param1", type="string", description="A test parameter"
                )
            ],
            category=ToolCategory.UTILITY,
        )

    def execute(self, **kwargs: Any) -> str:
        return f"Executed with {kwargs.get('param1', 'default')}"

    async def aexecute(self, **kwargs: Any) -> str:
        return self.execute(**kwargs)

    def get_interruption_message(self, **kwargs: Any) -> str:
        """Get interruption message for user confirmation."""
        param1 = kwargs.get("param1", "")
        if param1:
            return f"execute test_tool: {param1}"
        return "execute test_tool"


class TestParameterSchema:
    """Test cases for ParameterSchema."""

    def test_parameter_schema_creation(self):
        """Test creating a parameter schema."""
        param = ParameterSchema(
            name="test_param",
            type="string",
            description="A test parameter",
            required=True,
            default="default_value",
            minimum=0,
            maximum=100,
            min_length=1,
            max_length=10,
            pattern=r"^[a-z]+$",
            items_type="string",
            properties=[
                ParameterSchema(
                    name="nested_param", type="string", description="Nested parameter"
                )
            ],
        )

        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is True
        assert param.default == "default_value"
        assert param.minimum == 0
        assert param.maximum == 100
        assert param.min_length == 1
        assert param.max_length == 10
        assert param.pattern == r"^[a-z]+$"
        assert param.items_type == "string"
        assert len(param.properties) == 1
        assert param.properties[0].name == "nested_param"

    def test_parameter_schema_to_json_schema(self):
        """Test converting parameter schema to JSON schema."""
        param = ParameterSchema(
            name="test_param",
            type="string",
            description="A test parameter",
            enum=["option1", "option2"],
            default="option1",
            minimum=0,
            maximum=100,
            min_length=1,
            max_length=10,
            pattern=r"^[a-z]+$",
        )

        schema = param.to_json_schema()

        expected = {
            "type": "string",
            "description": "A test parameter",
            "enum": ["option1", "option2"],
            "default": "option1",
            "minimum": 0,
            "maximum": 100,
            "minLength": 1,
            "maxLength": 10,
            "pattern": r"^[a-z]+$",
        }

        assert schema == expected

    def test_parameter_schema_to_json_schema_array(self):
        """Test converting array parameter schema to JSON schema."""
        param = ParameterSchema(
            name="test_array",
            type="array",
            description="An array parameter",
            items_type="string",
        )

        schema = param.to_json_schema()

        expected = {
            "type": "array",
            "description": "An array parameter",
            "items": {"type": "string"},
        }

        assert schema == expected

    def test_parameter_schema_to_json_schema_object(self):
        """Test converting object parameter schema to JSON schema."""
        nested_param = ParameterSchema(
            name="nested", type="string", description="Nested parameter", required=True
        )
        param = ParameterSchema(
            name="test_object",
            type="object",
            description="An object parameter",
            properties=[nested_param],
        )

        schema = param.to_json_schema()

        expected = {
            "type": "object",
            "description": "An object parameter",
            "properties": {
                "nested": {"type": "string", "description": "Nested parameter"}
            },
            "required": ["nested"],
        }

        assert schema == expected


class TestToolSchema:
    """Test cases for ToolSchema."""

    def test_tool_schema_creation(self):
        """Test creating a tool schema."""
        param = ParameterSchema(name="param1", type="string", description="A parameter")
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            parameters=[param],
            category=ToolCategory.UTILITY,
            version="1.0.0",
            deprecated=False,
        )

        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert len(schema.parameters) == 1
        assert schema.category == ToolCategory.UTILITY
        assert schema.version == "1.0.0"
        assert schema.deprecated is False

    def test_tool_schema_to_json_schema(self):
        """Test converting tool schema to JSON schema."""
        param1 = ParameterSchema(
            name="param1", type="string", description="First parameter", required=True
        )
        param2 = ParameterSchema(
            name="param2",
            type="integer",
            description="Second parameter",
            required=False,
        )
        schema = ToolSchema(
            name="test_tool", description="A test tool", parameters=[param1, param2]
        )

        json_schema = schema.to_json_schema()

        expected = {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "First parameter"},
                "param2": {"type": "integer", "description": "Second parameter"},
            },
            "required": ["param1"],
        }

        assert json_schema == expected

    def test_tool_schema_to_openai_format(self):
        """Test converting tool schema to OpenAI format."""
        param = ParameterSchema(name="param1", type="string", description="A parameter")
        schema = ToolSchema(
            name="test_tool", description="A test tool", parameters=[param]
        )

        openai_format = schema.to_openai_format()

        expected = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "A parameter"}
                    },
                    "required": ["param1"],
                },
            },
        }

        assert openai_format == expected

    def test_tool_schema_to_provider_format(self):
        """Test converting tool schema to different provider formats."""
        param = ParameterSchema(name="param1", type="string", description="A parameter")
        schema = ToolSchema(
            name="test_tool", description="A test tool", parameters=[param]
        )

        # Test OpenAI format
        openai_format = schema.to_provider_format("openai")
        assert openai_format["type"] == "function"

        # Test Anthropic format
        anthropic_format = schema.to_provider_format("anthropic")
        assert "name" in anthropic_format
        assert "description" in anthropic_format
        assert "input_schema" in anthropic_format

        # Test Google format
        google_format = schema.to_provider_format("google")
        assert "name" in google_format
        assert "description" in google_format
        assert "parameters" in google_format

        # Test unknown provider (should default to OpenAI)
        unknown_format = schema.to_provider_format("unknown")
        assert unknown_format["type"] == "function"


class TestToolCall:
    """Test cases for ToolCall."""

    def test_tool_call_creation(self):
        """Test creating a tool call."""
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={"param1": "value1"},
            raw_arguments='{"param1": "value1"}',
        )

        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param1": "value1"}
        assert tool_call.raw_arguments == '{"param1": "value1"}'

    def test_tool_call_from_openai(self):
        """Test creating tool call from OpenAI format."""
        openai_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"param1": "value1"}'},
        }

        tool_call = ToolCall.from_openai(openai_call)

        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param1": "value1"}
        assert tool_call.raw_arguments == '{"param1": "value1"}'

    def test_tool_call_from_anthropic(self):
        """Test creating tool call from Anthropic format."""
        anthropic_call = {
            "id": "call_123",
            "name": "test_tool",
            "input": {"param1": "value1"},
        }

        tool_call = ToolCall.from_anthropic(anthropic_call)

        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param1": "value1"}
        assert tool_call.raw_arguments == '{"param1": "value1"}'

    def test_tool_call_from_google(self):
        """Test creating tool call from Google format."""
        google_call = {
            "id": "call_123",
            "name": "test_tool",
            "args": {"param1": "value1"},
        }

        tool_call = ToolCall.from_google(google_call)

        assert tool_call.id == "call_123"
        assert tool_call.name == "test_tool"
        assert tool_call.arguments == {"param1": "value1"}
        assert tool_call.raw_arguments == '{"param1": "value1"}'

    def test_tool_call_from_provider(self):
        """Test creating tool call from different provider formats."""
        # Test OpenAI format
        openai_call = {
            "id": "call_123",
            "function": {"name": "test_tool", "arguments": '{"param1": "value1"}'},
        }
        tool_call = ToolCall.from_provider(openai_call, "openai")
        assert tool_call.name == "test_tool"

        # Test Anthropic format
        anthropic_call = {
            "id": "call_123",
            "name": "test_tool",
            "input": {"param1": "value1"},
        }
        tool_call = ToolCall.from_provider(anthropic_call, "anthropic")
        assert tool_call.name == "test_tool"

        # Test unknown provider (should default to OpenAI)
        tool_call = ToolCall.from_provider(openai_call, "unknown")
        assert tool_call.name == "test_tool"


class TestToolResult:
    """Test cases for ToolResult."""

    def test_tool_result_creation(self):
        """Test creating a tool result."""
        result = ToolResult(
            tool_call_id="call_123",
            content="Success",
            status="success",
            data={"result": "data"},
            execution_time_ms=100.0,
        )

        assert result.tool_call_id == "call_123"
        assert result.content == "Success"
        assert result.status == "success"
        assert result.data == {"result": "data"}
        assert result.execution_time_ms == 100.0

    def test_tool_result_success_classmethod(self):
        """Test creating a successful tool result using class method."""
        result = ToolResult.success("call_123", "Success content", {"data": "value"})

        assert result.tool_call_id == "call_123"
        assert result.content == "Success content"
        assert result.status == "success"
        assert result.data == {"data": "value"}
        assert result.error_type is None
        assert result.error_message is None

    def test_tool_result_error_classmethod(self):
        """Test creating an error tool result using class method."""
        result = ToolResult.error("call_123", "Error message", "CustomError")

        assert result.tool_call_id == "call_123"
        assert result.content == "Error: Error message"
        assert result.status == "error"
        assert result.error_type == "CustomError"
        assert result.error_message == "Error message"
        assert result.data is None

    def test_tool_result_to_openai_format(self):
        """Test converting tool result to OpenAI format."""
        result = ToolResult(tool_call_id="call_123", content="Success")

        openai_format = result.to_openai_format()

        expected = {"role": "tool", "tool_call_id": "call_123", "content": "Success"}

        assert openai_format == expected

    def test_tool_result_to_anthropic_format(self):
        """Test converting tool result to Anthropic format."""
        result = ToolResult(
            tool_call_id="call_123", content="Success", status="success"
        )

        anthropic_format = result.to_anthropic_format()

        expected = {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": "Success",
        }

        assert anthropic_format == expected

    def test_tool_result_to_anthropic_format_error(self):
        """Test converting error tool result to Anthropic format."""
        result = ToolResult(tool_call_id="call_123", content="Error", status="error")

        anthropic_format = result.to_anthropic_format()

        expected = {
            "type": "tool_result",
            "tool_use_id": "call_123",
            "content": "Error",
            "is_error": True,
        }

        assert anthropic_format == expected

    def test_tool_result_to_provider_format(self):
        """Test converting tool result to different provider formats."""
        result = ToolResult(tool_call_id="call_123", content="Success")

        # Test OpenAI format
        openai_format = result.to_provider_format("openai")
        assert openai_format["role"] == "tool"

        # Test Anthropic format
        anthropic_format = result.to_provider_format("anthropic")
        assert anthropic_format["type"] == "tool_result"

        # Test unknown provider (should default to OpenAI)
        unknown_format = result.to_provider_format("unknown")
        assert unknown_format["role"] == "tool"


class TestBaseTool:
    """Test cases for BaseTool."""

    def test_base_tool_initialization(self):
        """Test BaseTool initialization."""
        tool = ConcreteTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.schema.name == "test_tool"
        assert tool.schema.category == ToolCategory.UTILITY
        assert tool.execution_count == 0
        assert tool.average_execution_time_ms == 0.0

    def test_base_tool_properties(self):
        """Test BaseTool properties."""
        tool = ConcreteTool()

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert isinstance(tool.schema, ToolSchema)
        assert tool.execution_count == 0
        assert tool.average_execution_time_ms == 0.0

    def test_base_tool_to_provider_format(self):
        """Test getting tool in provider format."""
        tool = ConcreteTool()

        openai_format = tool.to_provider_format("openai")
        assert openai_format["function"]["name"] == "test_tool"

        anthropic_format = tool.to_provider_format("anthropic")
        assert anthropic_format["name"] == "test_tool"

    def test_validate_arguments_valid(self):
        """Test validating valid arguments."""
        tool = ConcreteTool()
        args = {"param1": "test_value"}

        is_valid, errors = tool.validate_arguments(args)

        assert is_valid is True
        assert errors == []

    def test_validate_arguments_missing_required(self):
        """Test validating arguments with missing required parameter."""
        tool = ConcreteTool()
        args = {}  # Missing required param1

        is_valid, errors = tool.validate_arguments(args)

        assert is_valid is False
        assert "Missing required parameter: param1" in errors

    def test_validate_arguments_wrong_type(self):
        """Test validating arguments with wrong type."""
        param = ParameterSchema(
            name="param1", type="integer", description="An integer parameter"
        )
        tool = ConcreteTool()
        tool._schema.parameters = [param]
        args = {"param1": "not_an_integer"}

        is_valid, errors = tool.validate_arguments(args)

        assert is_valid is False
        assert "Parameter 'param1' should be integer, got str" in errors

    def test_validate_arguments_correct_type(self):
        """Test validating arguments with correct type."""
        param = ParameterSchema(
            name="param1", type="integer", description="An integer parameter"
        )
        tool = ConcreteTool()
        tool._schema.parameters = [param]
        args = {"param1": 42}

        is_valid, errors = tool.validate_arguments(args)

        assert is_valid is True
        assert errors == []

    def test_run_success(self):
        """Test running a tool successfully."""
        tool = ConcreteTool()
        tool_call = ToolCall(
            id="call_123", name="test_tool", arguments={"param1": "test_value"}
        )

        result = tool.run(tool_call)

        assert result.status == "success"
        assert result.content == "Executed with test_value"
        assert result.tool_call_id == "call_123"
        assert result.data == "Executed with test_value"
        assert result.execution_time_ms is not None
        assert tool.execution_count == 1

    def test_run_validation_error(self):
        """Test running a tool with validation error."""
        tool = ConcreteTool()
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={},  # Missing required param
        )

        result = tool.run(tool_call)

        assert result.status == "error"
        assert "Missing required parameter: param1" in result.content
        assert result.error_type == "ValidationError"

    def test_run_execution_error(self):
        """Test running a tool with execution error."""

        # Create a tool that raises an exception
        class FailingTool(BaseTool[str]):
            def __init__(self):
                super().__init__(
                    name="failing_tool", description="A failing tool", parameters=[]
                )

            def execute(self, **kwargs: Any) -> str:
                raise ValueError("Something went wrong")

            async def aexecute(self, **kwargs: Any) -> str:
                raise ValueError("Something went wrong")

            def get_interruption_message(self, **kwargs: Any) -> str:
                """Get interruption message for user confirmation."""
                return "execute failing_tool"

        tool = FailingTool()
        tool_call = ToolCall(id="call_123", name="failing_tool", arguments={})

        result = tool.run(tool_call)

        assert result.status == "error"
        assert "Something went wrong" in result.content
        assert result.error_type == "ValueError"

    @pytest.mark.asyncio
    async def test_arun_success(self):
        """Test running a tool asynchronously successfully."""
        tool = ConcreteTool()
        tool_call = ToolCall(
            id="call_123", name="test_tool", arguments={"param1": "async_value"}
        )

        result = await tool.arun(tool_call)

        assert result.status == "success"
        assert result.content == "Executed with async_value"
        assert result.tool_call_id == "call_123"
        assert tool.execution_count == 1

    @pytest.mark.asyncio
    async def test_arun_validation_error(self):
        """Test running a tool asynchronously with validation error."""
        tool = ConcreteTool()
        tool_call = ToolCall(
            id="call_123",
            name="test_tool",
            arguments={},  # Missing required param
        )

        result = await tool.arun(tool_call)

        assert result.status == "error"
        assert "Missing required parameter: param1" in result.content
        assert result.error_type == "ValidationError"

    def test_reset_tracking(self):
        """Test resetting execution tracking."""
        tool = ConcreteTool()

        # Simulate some executions
        tool._execution_count = 5
        tool._total_execution_time_ms = 500.0
        tool._last_execution_time_ms = 100.0

        tool.reset_tracking()

        assert tool.execution_count == 0
        assert tool.average_execution_time_ms == 0.0
        assert tool._last_execution_time_ms is None


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def test_tool_registry_initialization(self):
        """Test ToolRegistry initialization."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.names == []
        assert registry.tools == []

    def test_register_tool(self):
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTool()

        registry.register(tool)

        assert len(registry) == 1
        assert "test_tool" in registry
        assert registry.get("test_tool") == tool
        assert registry.names == ["test_tool"]
        assert registry.tools == [tool]

    def test_register_duplicate_tool(self):
        """Test registering a duplicate tool."""
        registry = ToolRegistry()
        tool1 = ConcreteTool()
        tool2 = ConcreteTool()  # Same name

        registry.register(tool1)

        with pytest.raises(ValueError, match="Tool 'test_tool' is already registered"):
            registry.register(tool2)

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        tool = ConcreteTool()

        registry.register(tool)
        result = registry.unregister("test_tool")

        assert result is True
        assert len(registry) == 0
        assert "test_tool" not in registry
        assert registry.get("test_tool") is None

    def test_unregister_nonexistent_tool(self):
        """Test unregistering a nonexistent tool."""
        registry = ToolRegistry()

        result = registry.unregister("nonexistent")

        assert result is False
        assert len(registry) == 0

    def test_get_tool_by_category(self):
        """Test getting tools by category."""
        registry = ToolRegistry()

        utility_tool = ConcreteTool()
        search_tool = ConcreteTool()
        search_tool._schema.name = "search_tool"
        search_tool._schema.category = ToolCategory.SEARCH

        registry.register(utility_tool)
        registry.register(search_tool)

        utility_tools = registry.get_by_category(ToolCategory.UTILITY)
        search_tools = registry.get_by_category(ToolCategory.SEARCH)

        assert len(utility_tools) == 1
        assert utility_tools[0].name == "test_tool"

        assert len(search_tools) == 1
        assert search_tools[0].name == "search_tool"

    def test_to_provider_format(self):
        """Test getting all tools in provider format."""
        registry = ToolRegistry()
        tool = ConcreteTool()

        registry.register(tool)

        openai_format = registry.to_provider_format("openai")

        assert len(openai_format) == 1
        assert openai_format[0]["function"]["name"] == "test_tool"

    def test_run_registered_tool(self):
        """Test running a registered tool."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        tool_call = ToolCall(
            id="call_123", name="test_tool", arguments={"param1": "run_value"}
        )

        result = registry.run(tool_call)

        assert result.status == "success"
        assert result.content == "Executed with run_value"
        assert result.tool_call_id == "call_123"

    def test_run_nonexistent_tool(self):
        """Test running a nonexistent tool."""
        registry = ToolRegistry()

        tool_call = ToolCall(id="call_123", name="nonexistent_tool", arguments={})

        result = registry.run(tool_call)

        assert result.status == "error"
        assert "Tool 'nonexistent_tool' not found" in result.content
        assert result.error_type == "ToolNotFoundError"

    @pytest.mark.asyncio
    async def test_arun_registered_tool(self):
        """Test running a registered tool asynchronously."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        tool_call = ToolCall(
            id="call_123", name="test_tool", arguments={"param1": "async_run_value"}
        )

        result = await registry.arun(tool_call)

        assert result.status == "success"
        assert result.content == "Executed with async_run_value"
        assert result.tool_call_id == "call_123"

    def test_registry_iteration(self):
        """Test iterating over registered tools."""
        registry = ToolRegistry()
        tool1 = ConcreteTool()
        tool1._schema.name = "tool1"  # Change name to avoid duplicate error

        tool2 = ConcreteTool()
        tool2._schema.name = "tool2"

        registry.register(tool1)
        registry.register(tool2)

        tools = list(registry)

        assert len(tools) == 2
        assert tools[0].name in ["tool1", "tool2"]
        assert tools[1].name in ["tool1", "tool2"]
        assert tools[0].name != tools[1].name

    def test_repr(self):
        """Test string representation of registry."""
        registry = ToolRegistry()
        tool = ConcreteTool()
        registry.register(tool)

        repr_str = repr(registry)

        assert "ToolRegistry" in repr_str
        assert "test_tool" in repr_str


def test_tool_decorator():
    """Test the @tool decorator."""

    @tool(
        name="decorated_tool",
        description="A decorated tool",
        parameters=[
            ParameterSchema(name="param1", type="string", description="A parameter")
        ],
        category=ToolCategory.UTILITY,
    )
    def decorated_function(param1: str) -> str:
        return f"Result: {param1}"

    assert isinstance(decorated_function, BaseTool)
    assert decorated_function.name == "decorated_tool"
    assert decorated_function.description == "A decorated tool"
    assert len(decorated_function.schema.parameters) == 1
    assert decorated_function.schema.parameters[0].name == "param1"

    # Test execution
    result = decorated_function.execute(param1="test")
    assert result == "Result: test"
