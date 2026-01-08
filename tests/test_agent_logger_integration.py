import tempfile
from pathlib import Path

from kader.agent.logger import AgentLogger


class TestAgentLoggerIntegration:
    """Integration tests for the AgentLogger class using real file system."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent_logger = AgentLogger()
        # Clear the _loggers dictionary to isolate tests
        self.agent_logger._loggers = {}
        # Create a temporary directory for logs
        self.temp_logs_dir = tempfile.mkdtemp()
        # Patch the home directory temporarily for testing
        self.original_home = Path.home()

    def teardown_method(self):
        """Clean up after each test method."""
        # Clean up any created log files
        import shutil

        # Find and remove the .kader directory if it was created during the test
        home_kader_dir = self.original_home / ".kader"
        if home_kader_dir.exists():
            shutil.rmtree(home_kader_dir, ignore_errors=True)

    def test_real_logger_setup_and_usage(self):
        """Test the actual logger setup and usage functionality."""
        # Test that logger is created when session_id is provided
        logger_id = self.agent_logger.setup_logger("test_agent", "session123")
        assert logger_id == "test_agent_session123"
        assert logger_id in self.agent_logger._loggers

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
