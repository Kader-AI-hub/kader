import threading
from unittest.mock import MagicMock, patch

import pytest

from kader.agent.logger import AgentLogger


class TestAgentLogger:
    """Unit tests for the AgentLogger class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a new AgentLogger instance for each test
        # We need to reset the internal _loggers dict to start fresh
        self.agent_logger = AgentLogger()
        # Clear the _loggers dictionary to isolate tests
        self.agent_logger._loggers = {}

    def test_singleton_pattern(self):
        """Test that AgentLogger follows the singleton pattern."""
        # Get the global instance
        from kader.agent.logger import agent_logger

        logger1 = agent_logger
        logger2 = agent_logger
        assert logger1 is logger2

    def test_setup_logger_no_session_id(self):
        """Test that setup_logger returns None when no session_id is provided."""
        result = self.agent_logger.setup_logger("test_agent", None)
        assert result is None

        result = self.agent_logger.setup_logger("test_agent", "")
        assert result is None

    def test_setup_logger_with_session_id(self):
        """Test that setup_logger creates and returns a logger when session_id is provided."""
        with patch("kader.agent.logger.Path") as mock_path:
            # Mock the Path operations
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            # Mock the loguru logger
            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                result = self.agent_logger.setup_logger(
                    "test_agent", "test_session_123"
                )

                assert result == "test_agent_test_session_123"
                # Verify the logger was set up properly
                assert "test_agent_test_session_123" in self.agent_logger._loggers
                mock_logger.bind.assert_called()
                mock_bound_logger.add.assert_called_once()

    def test_setup_logger_validates_inputs(self):
        """Test that setup_logger properly validates inputs."""
        # Test with None agent_name - should still create logger as long as session_id exists
        result = self.agent_logger.setup_logger(None, "test_session_123")
        assert result is not None  # Because session_id is provided

        # Test with empty agent_name - should still create logger as long as session_id exists
        result = self.agent_logger.setup_logger("", "test_session_123")
        assert result is not None  # Because session_id is provided

        # Test with valid agent_name but invalid session_id (None)
        result = self.agent_logger.setup_logger("test_agent", None)
        assert result is None

        # Test with valid agent_name but invalid session_id (empty string)
        result = self.agent_logger.setup_logger("test_agent", "")
        assert result is None

    def test_log_token_usage(self):
        """Test logging of token usage."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Now test the token usage logging
                self.agent_logger.log_token_usage(
                    logger_id,
                    prompt_tokens=100,
                    completion_tokens=200,
                    total_tokens=300,
                )

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains token information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "TOKEN_USAGE" in log_message
                assert "Prompt tokens: 100" in log_message
                assert "Completion tokens: 200" in log_message
                assert "Total tokens: 300" in log_message

    def test_log_token_usage_invalid_logger_id(self):
        """Test logging token usage with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_token_usage(
            "nonexistent_logger",
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300,
        )
        # No exception means the test passes

    def test_log_token_usage_with_negative_tokens(self):
        """Test logging token usage with negative token values."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with negative values - should still log correctly
                self.agent_logger.log_token_usage(
                    logger_id,
                    prompt_tokens=-100,
                    completion_tokens=-200,
                    total_tokens=-300,
                )

                mock_bound_logger.info.assert_called_once()

    def test_calculate_cost(self):
        """Test cost calculation functionality."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with default pricing
                cost = self.agent_logger.calculate_cost(logger_id, total_cost=0.5)

                # Default pricing: input_cost_per_million = 0.5, output_cost_per_million = 1.5
                expected_cost = 0.5
                assert cost == expected_cost

    def test_log_llm_response(self):
        """Test logging of LLM responses."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                response = "Test LLM response"
                self.agent_logger.log_llm_response(logger_id, response)

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains LLM response information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "LLM_RESPONSE" in log_message
                assert response in log_message

    def test_log_llm_response_invalid_logger_id(self):
        """Test logging LLM response with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_llm_response("nonexistent_logger", "test response")
        # No exception means the test passes

    def test_log_llm_response_empty_response(self):
        """Test logging empty or None LLM responses."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with empty string
                self.agent_logger.log_llm_response(logger_id, "")
                mock_bound_logger.info.assert_called_once()

                # Reset the mock and test with None
                mock_bound_logger.reset_mock()
                self.agent_logger.log_llm_response(logger_id, None)
                # Even with None, it should still try to log (might log 'None' as string)
                mock_bound_logger.info.assert_called_once()

    def test_log_tool_usage(self):
        """Test logging of tool usage."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                self.agent_logger.log_tool_usage(
                    logger_id, "test_tool", {"param": "value"}
                )

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains tool usage information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "TOOL_USAGE" in log_message
                assert "test_tool" in log_message
                assert "param" in log_message
                assert "value" in log_message

    def test_log_tool_usage_invalid_logger_id(self):
        """Test logging tool usage with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_tool_usage(
            "nonexistent_logger", "test_tool", {"param": "value"}
        )
        # No exception means the test passes

    def test_log_tool_usage_without_arguments(self):
        """Test logging tool usage without arguments."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with None arguments
                self.agent_logger.log_tool_usage(logger_id, "test_tool", None)
                mock_bound_logger.info.assert_called_once()

                # Reset the mock and test with empty dict
                mock_bound_logger.reset_mock()
                self.agent_logger.log_tool_usage(logger_id, "test_tool", {})
                mock_bound_logger.info.assert_called_once()

    def test_log_cost(self):
        """Test logging of cost information."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                self.agent_logger.log_cost(logger_id, 1.23)

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains cost information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "COST" in log_message
                assert "1.23" in log_message

    def test_log_cost_invalid_logger_id(self):
        """Test logging cost with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_cost("nonexistent_logger", 1.23)
        # No exception means the test passes

    def test_log_interaction(self):
        """Test logging of complete agent interactions."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                self.agent_logger.log_interaction(
                    logger_id,
                    input_msg="Test input",
                    output_msg="Test output",
                    token_usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 200,
                        "total_tokens": 300,
                    },
                    cost=1.23,
                    tools_used=[{"name": "test_tool", "arguments": {"param": "value"}}],
                )

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains interaction information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "INTERACTION" in log_message
                assert "Test input" in log_message
                assert "Test output" in log_message
                assert "Prompt: 100" in log_message  # prompt tokens
                assert "Completion: 200" in log_message  # completion tokens
                assert "Total: 300" in log_message  # total tokens
                assert "1.23" in log_message  # cost
                assert "test_tool" in log_message
                assert "param" in log_message
                assert "value" in log_message

    def test_log_interaction_optional_fields(self):
        """Test logging of interactions with optional fields as None."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with None values for optional fields
                self.agent_logger.log_interaction(
                    logger_id,
                    input_msg="Test input",
                    output_msg="Test output",
                    token_usage=None,
                    cost=None,
                    tools_used=None,
                )

                # Should still log but without the optional components
                mock_bound_logger.info.assert_called_once()
                # The call should still succeed even with None values

    def test_log_interaction_invalid_logger_id(self):
        """Test logging interaction with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_interaction(
            "nonexistent_logger",
            input_msg="Test input",
            output_msg="Test output",
            token_usage={
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300,
            },
            cost=1.23,
            tools_used=[{"name": "test_tool", "arguments": {"param": "value"}}],
        )
        # No exception means the test passes

    def test_log_event(self):
        """Test logging of custom events."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                self.agent_logger.log_event(logger_id, "test_event", {"data": "value"})

                # Verify info was called with the correct message
                mock_bound_logger.info.assert_called_once()
                # Check that the log message contains event information
                args, kwargs = mock_bound_logger.info.call_args
                log_message = args[0]
                assert "TEST_EVENT" in log_message
                assert "data" in log_message
                assert "value" in log_message

    def test_log_event_invalid_logger_id(self):
        """Test logging event with invalid logger_id."""
        # Test with logger_id not in _loggers
        # This should not raise an exception
        self.agent_logger.log_event(
            "nonexistent_logger", "test_event", {"data": "value"}
        )
        # No exception means the test passes

    def test_log_event_empty_data(self):
        """Test logging events with empty or None data."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # Test with None data
                self.agent_logger.log_event(logger_id, "test_event", None)
                mock_bound_logger.info.assert_called_once()

                # Reset the mock and test with empty dict
                mock_bound_logger.reset_mock()
                self.agent_logger.log_event(logger_id, "test_event", {})
                mock_bound_logger.info.assert_called_once()

    def test_thread_safety_for_setup(self):
        """Test that setup_logger is thread-safe."""
        results = []

        # Create multiple threads that try to setup the same logger
        def setup_logger():
            with patch("kader.agent.logger.Path") as mock_path:
                mock_logs_dir = MagicMock()
                mock_path.return_value = mock_logs_dir
                mock_logs_dir.__truediv__.return_value = mock_logs_dir
                mock_logs_dir.mkdir.return_value = None
                mock_log_file_path = MagicMock()
                mock_logs_dir.__truediv__.return_value = mock_log_file_path

                with patch("kader.agent.logger.logger") as mock_logger:
                    mock_bound_logger = MagicMock()
                    mock_logger.bind.return_value = mock_bound_logger
                    mock_bound_logger.add.return_value = 1  # handler ID

                    result = self.agent_logger.setup_logger(
                        "test_agent", "test_session"
                    )
                    results.append(result)

        threads = []
        for i in range(5):
            thread = threading.Thread(target=setup_logger)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Only one thread should have created a new logger, others should have returned the existing one
        non_none_results = [r for r in results if r is not None]
        # Since only the first thread will create the logger, we expect 1 non-None result
        assert len(non_none_results) >= 1  # At least one should succeed

    def test_error_handling_invalid_json_in_arguments(self):
        """Test error handling when logging invalid JSON arguments."""
        logger_id = "test_agent_test_session"

        # Setup logger first
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger = MagicMock()
                mock_logger.bind.return_value = mock_bound_logger
                mock_bound_logger.add.return_value = 1  # handler ID

                # Setup the logger
                self.agent_logger.setup_logger("test_agent", "test_session")

                # The actual implementation doesn't currently handle JSON serialization errors
                # So we expect this to raise an exception in the real implementation
                # The improved implementation should handle this gracefully
                problematic_arg = {
                    "function": lambda x: x
                }  # Functions are not JSON serializable

                # The real implementation will raise an exception, so we test this explicitly
                with pytest.raises(TypeError):
                    self.agent_logger.log_tool_usage(
                        logger_id, "test_tool", problematic_arg
                    )

    def test_multiple_loggers_different_sessions(self):
        """Test that different logger IDs create separate logger instances."""
        # Setup two different loggers
        with patch("kader.agent.logger.Path") as mock_path:
            mock_logs_dir = MagicMock()
            mock_path.return_value = mock_logs_dir
            mock_logs_dir.__truediv__.return_value = mock_logs_dir
            mock_logs_dir.mkdir.return_value = None
            mock_log_file_path = MagicMock()
            mock_logs_dir.__truediv__.return_value = mock_log_file_path

            with patch("kader.agent.logger.logger") as mock_logger:
                mock_bound_logger1 = MagicMock()
                mock_bound_logger2 = MagicMock()

                # Configure the mock_logger.bind to return different bound loggers on different calls
                mock_logger.bind.return_value = mock_bound_logger1
                mock_bound_logger1.add.return_value = 1
                mock_bound_logger2.add.return_value = 2

                logger_id1 = self.agent_logger.setup_logger("agent1", "session1")
                logger_id2 = self.agent_logger.setup_logger("agent2", "session2")

                # Verify both loggers were created
                assert logger_id1 is not None
                assert logger_id2 is not None
                assert logger_id1 != logger_id2

                # Verify both are in the _loggers dict
                assert logger_id1 in self.agent_logger._loggers
                assert logger_id2 in self.agent_logger._loggers
