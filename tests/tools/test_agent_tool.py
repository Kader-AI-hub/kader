"""Unit tests for AgentTool."""

from unittest.mock import MagicMock, patch

import pytest

from kader.tools.agent import AgentTool


class TestAgentToolInit:
    """Test AgentTool initialization."""

    def test_init_with_name(self):
        """Test basic initialization with name."""
        tool = AgentTool(name="test_agent")

        assert tool.name == "test_agent"
        assert tool.description == "Execute a specific task using an AI agent"

    def test_init_with_custom_description(self):
        """Test initialization with custom description."""
        tool = AgentTool(
            name="custom_agent",
            description="A specialized agent for data analysis",
        )

        assert tool.name == "custom_agent"
        assert tool.description == "A specialized agent for data analysis"

    def test_init_with_custom_model(self):
        """Test initialization with custom model."""
        tool = AgentTool(
            name="custom_model_agent",
            model_name="gpt-4",
        )

        assert tool._model_name == "gpt-4"

    def test_parameters(self):
        """Test that the tool has correct parameters."""
        tool = AgentTool(name="test_agent")

        assert len(tool.schema.parameters) == 1
        param = tool.schema.parameters[0]
        assert param.name == "task"
        assert param.type == "string"
        assert param.required is True


class TestAgentToolInterruptionMessage:
    """Test get_interruption_message method."""

    def test_short_task(self):
        """Test interruption message with short task."""
        tool = AgentTool(name="research_agent")
        msg = tool.get_interruption_message(task="Find the weather")

        assert msg == "execute research_agent: Find the weather"

    def test_long_task_truncation(self):
        """Test that long tasks are truncated in the interruption message."""
        tool = AgentTool(name="research_agent")
        long_task = "A" * 150  # 150 character task
        msg = tool.get_interruption_message(task=long_task)

        assert len(msg) < 150  # Task should be truncated
        assert "..." in msg
        assert msg.startswith("execute research_agent:")


class TestAgentToolExecution:
    """Test AgentTool execution with mocked agent."""

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_success(self, mock_registry, mock_react_agent):
        """Test successful task execution."""
        # Setup mocks
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Task completed successfully. Found relevant information."
        )
        mock_agent_instance.invoke.return_value = mock_response
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task")

        # Verify
        assert result == "Task completed successfully. Found relevant information."
        mock_agent_instance.invoke.assert_called_once_with("Test task")

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_with_dict_response(self, mock_registry, mock_react_agent):
        """Test execution when agent returns a dict."""
        # Setup mocks
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {"content": "Dict response"}
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task")

        # Verify
        assert result == "Dict response"

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_failure(self, mock_registry, mock_react_agent):
        """Test execution failure handling."""
        # Setup mocks to raise an exception
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.side_effect = Exception("LLM connection failed")
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task")

        # Verify error handling
        assert "Agent execution failed" in result
        assert "LLM connection failed" in result

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_agent_created_with_no_interrupt(self, mock_registry, mock_react_agent):
        """Test that sub-agent is created with interrupt_before_tool=True."""
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Done"
        mock_agent_instance.invoke.return_value = mock_response
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        tool = AgentTool(name="test_agent")
        tool.execute(task="Test task")

        # Verify the agent was created with interrupt_before_tool=True
        call_kwargs = mock_react_agent.call_args[1]
        assert call_kwargs["interrupt_before_tool"] is True


@pytest.mark.asyncio
class TestAgentToolAsyncExecution:
    """Test async execution of AgentTool."""

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    async def test_aexecute_success(self, mock_registry, mock_react_agent):
        """Test successful async execution."""
        # Setup mocks
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Async task completed"

        # Make ainvoke an async mock
        async def async_invoke(task):
            return mock_response

        mock_agent_instance.ainvoke = async_invoke
        mock_react_agent.return_value = mock_agent_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = await tool.aexecute(task="Async test task")

        # Verify
        assert result == "Async task completed"
