from pathlib import Path

from kader.agent.logger import AgentLogger


class TestAgentLoggerIntegration:
    """Integration tests for the AgentLogger class using real file system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent_logger = AgentLogger()
        # Clear the _loggers dictionary to isolate tests
        self.agent_logger._loggers = {}
        # The global mock_kader_home fixture in conftest.py already handles
        # mocking Path.home() to a temp directory, so we don't need to do it here manually.
        # We can just rely on the fact that Path.home() returns a temp dir.

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove all logger handlers created during the test to release file locks
        for logger_id, (
            logger_instance,
            handler_id,
        ) in self.agent_logger._loggers.items():
            try:
                # Remove the handler from the specific logger instance
                logger_instance.remove(handler_id)
            except Exception:
                pass

        # Clear the loggers dictionary
        self.agent_logger._loggers = {}

    def test_real_logger_setup_and_usage(self):
        """Test the actual logger setup and usage functionality."""
        # Test that logger is created when session_id is provided
        logger_id = self.agent_logger.setup_logger("test_agent", "session123")
        assert logger_id == "test_agent_session123"
        assert logger_id in self.agent_logger._loggers

        # Verify it created the directory in the mocked home
        expected_log_file = (
            Path.home() / ".kader" / "logs" / "test_agent_session123.log"
        )
        assert expected_log_file.exists()

        # Test that logger is not created when no session_id is provided
        result = self.agent_logger.setup_logger("test_agent", None)
        assert result is None

        result = self.agent_logger.setup_logger("test_agent", "")
        assert result is None

    def test_real_log_token_usage(self):
        """Test actual token usage logging."""
        # Setup logger
        logger_id = self.agent_logger.setup_logger("test_agent", "session123")
        assert logger_id is not None

        # Log token usage
        self.agent_logger.log_token_usage(
            logger_id, prompt_tokens=100, completion_tokens=200, total_tokens=300
        )

        # The log should have been written successfully without errors
        # (We're not checking file contents here, just that no exception was raised)

    def test_real_log_interaction(self):
        """Test actual interaction logging."""
        # Setup logger
        logger_id = self.agent_logger.setup_logger("test_agent", "session123")
        assert logger_id is not None

        # Log interaction
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

        # The log should have been written successfully without errors
