"""Unit tests for AgentTool."""

import os
import tempfile
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

        assert len(tool.schema.parameters) == 2

        # Check task param
        task_param = tool.schema.parameters[0]
        assert task_param.name == "task"
        assert task_param.type == "string"
        assert task_param.required is True

        # Check context param
        context_param = tool.schema.parameters[1]
        assert context_param.name == "context"
        assert context_param.type == "string"
        assert context_param.required is True


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

    def setup_method(self):
        self.temp_fd, self.checkpoint_path = tempfile.mkstemp()

    def teardown_method(self):
        os.close(self.temp_fd)
        os.remove(self.checkpoint_path)

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_success(
        self, mock_registry, mock_react_agent, mock_checkpointer, mock_aggregator
    ):
        """Test successful task execution."""
        # Setup mocks
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Task completed successfully."
        mock_agent_instance.invoke.return_value = mock_response
        mock_react_agent.return_value = mock_agent_instance

        # Setup Checkpointer to return our temp file
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("Task completed via checkpoint.")

        mock_checkpointer_instance = MagicMock()
        mock_checkpointer_instance.generate_checkpoint.return_value = (
            self.checkpoint_path
        )
        mock_checkpointer.return_value = mock_checkpointer_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task", context="Test context")

        # Verify
        # It should return the content of the checkpoint file
        assert (
            result
            == "Task completed via checkpoint.\n\nResponse:\nTask completed successfully."
        )
        mock_agent_instance.invoke.assert_called_once_with("Test task")

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_with_dict_response(
        self, mock_registry, mock_react_agent, mock_checkpointer, mock_aggregator
    ):
        """Test execution when agent returns a dict."""
        # Setup mocks
        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = {"content": "Dict response"}
        mock_react_agent.return_value = mock_agent_instance

        # Setup Checkpointer to return our temp file
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("Dict response via checkpoint.")

        mock_checkpointer_instance = MagicMock()
        mock_checkpointer_instance.generate_checkpoint.return_value = (
            self.checkpoint_path
        )
        mock_checkpointer.return_value = mock_checkpointer_instance

        # Execute
        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task", context="Test context")

        # Verify
        # Verify
        assert result == "Dict response via checkpoint.\n\nResponse:\nDict response"

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
        result = tool.execute(task="Test task", context="Test context")

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

        # We need to mock Checkpointer here too or ensure it doesn't fail
        # Or just assert on what happens before checkpointer
        # But execute() calls checkpointer.
        # If checkpointer fails (not mocked), we get execution failed or error.
        # But we handle exceptions in execute().

        # Actually, let's just mock checkpointer to avoid errors in logs
        with patch("kader.tools.agent.Checkpointer") as mock_cp:
            mock_cp_instance = MagicMock()
            mock_cp_instance.generate_checkpoint.side_effect = Exception(
                "Checkpoint failed"
            )
            mock_cp.return_value = mock_cp_instance

            # Execute
            tool = AgentTool(name="test_agent")
            tool.execute(task="Test task", context="Test context")

            # Verify the agent was created with interrupt_before_tool=True
            call_kwargs = mock_react_agent.call_args[1]
            assert call_kwargs["interrupt_before_tool"] is True


@pytest.mark.asyncio
class TestAgentToolAsyncExecution:
    """Test async execution of AgentTool."""

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    async def test_aexecute_success(
        self, mock_registry, mock_react_agent, mock_checkpointer, mock_aggregator
    ):
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

        # Create temp file
        fd, path = tempfile.mkstemp()
        with open(path, "w", encoding="utf-8") as f:
            f.write("Async task completed via checkpoint.")
        os.close(fd)

        try:
            mock_checkpointer_instance = MagicMock()

            # agenerate_checkpoint is async
            async def async_generate_checkpoint(file):
                return path

            mock_checkpointer_instance.agenerate_checkpoint.side_effect = (
                async_generate_checkpoint
            )
            mock_checkpointer.return_value = mock_checkpointer_instance

            # Mock aggregate (async)
            mock_aggregator_instance = MagicMock()
            mock_aggregator_instance.aaggregate = MagicMock()
            mock_aggregator.return_value = mock_aggregator_instance

            # Execute
            tool = AgentTool(name="test_agent")
            result = await tool.aexecute(
                task="Async test task", context="Async context"
            )

            # Verify
            # Verify
            assert (
                result
                == "Async task completed via checkpoint.\n\nResponse:\nAsync task completed"
            )

        finally:
            os.remove(path)


class TestToolExecutionRejected:
    """Test ToolExecutionRejected exception."""

    def test_exception_default_message(self):
        """Test that exception has a default message."""
        from kader.tools.base import ToolExecutionRejected

        exc = ToolExecutionRejected()
        assert exc.message == "Tool execution was rejected by the user."
        assert str(exc) == "Tool execution was rejected by the user."

    def test_exception_custom_message(self):
        """Test that exception accepts a custom message."""
        from kader.tools.base import ToolExecutionRejected

        exc = ToolExecutionRejected("Custom rejection reason")
        assert exc.message == "Custom rejection reason"
        assert str(exc) == "Custom rejection reason"


class TestAgentToolRejection:
    """Test AgentTool behavior when sub-agent raises ToolExecutionRejected."""

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_tool_rejected(self, mock_registry, mock_react_agent):
        """Test that execute returns [REJECTED] message when tool is rejected."""
        from kader.tools.agent import ToolExecutionRejected

        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.side_effect = ToolExecutionRejected(
            "User rejected the tool execution."
        )
        mock_react_agent.return_value = mock_agent_instance

        tool = AgentTool(name="test_agent")
        result = tool.execute(task="Test task", context="Test context")

        assert "[REJECTED]" in result
        assert "rejected the tool execution" in result
        assert "stopped" in result

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    async def test_aexecute_tool_rejected(self, mock_registry, mock_react_agent):
        """Test that aexecute returns [REJECTED] message when tool is rejected."""
        from kader.tools.agent import ToolExecutionRejected

        mock_registry.return_value = MagicMock()
        mock_agent_instance = MagicMock()

        async def async_invoke_rejected(task):
            raise ToolExecutionRejected("User rejected the tool execution.")

        mock_agent_instance.ainvoke = async_invoke_rejected
        mock_react_agent.return_value = mock_agent_instance

        tool = AgentTool(name="test_agent")
        result = await tool.aexecute(task="Test task", context="Test context")

        assert "[REJECTED]" in result
        assert "rejected the tool execution" in result
        assert "stopped" in result
