"""
Logger module for Kader agents.

This module provides logging functionality for agents with memory sessions.
Logs are written to files in ~/.kader/logs without affecting agent performance.
Only agents with memory sessions will generate logs.
"""

import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


class AgentLogger:
    """
    Logger class for Kader agents that logs to files with thread-safe operations.
    Only agents with memory sessions will be logged.
    """

    def __init__(self):
        self._loggers = {}
        self._lock = Lock()

    def setup_logger(
        self, agent_name: str, session_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Set up logger for an agent with memory session.

        Args:
            agent_name: Name of the agent
            session_id: Session ID for the agent

        Returns:
            Logger ID if successful, None otherwise
        """
        if not session_id:
            # Only log agents with memory sessions
            return None

        # Create log file path
        logs_dir = Path.home() / ".kader" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_filename = f"{agent_name}_{session_id}.log"
        log_file_path = logs_dir / log_filename

        logger_id = f"{agent_name}_{session_id}"

        # Add file sink with thread-safe configuration
        with self._lock:
            if logger_id not in self._loggers:
                # Remove default handler to avoid console output
                # We don't remove default handlers globally as other parts of the app might use them
                # Instead, create a new logger instance for our file logging
                new_logger = logger.bind(name=logger_id)

                # Remove all existing handlers from this logger instance
                new_logger.remove()

                # Add file sink with rotation and compression - NO CONSOLE OUTPUT
                handler_id = new_logger.add(
                    log_file_path,
                    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                    level="INFO",
                    rotation="10 MB",
                    retention="7 days",
                    compression="zip",
                    enqueue=True,  # Enables thread-safe logging in a separate thread
                    serialize=False,
                    backtrace=True,
                    diagnose=True,
                )

                self._loggers[logger_id] = (new_logger, handler_id)
                return logger_id

        return logger_id

    def log_token_usage(
        self,
        logger_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
    ):
        """Log token usage information."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]
            logger_instance.info(
                f"TOKEN_USAGE | Prompt tokens: {prompt_tokens}, "
                f"Completion tokens: {completion_tokens}, "
                f"Total tokens: {total_tokens}"
            )

    def calculate_cost(
        self,
        logger_id: str,
        total_cost: float,
    ):
        """Calculate and log cost based on token usage."""

        self.log_cost(logger_id, total_cost)
        return total_cost

    def log_llm_response(self, logger_id: str, response: Any):
        """Log LLM response."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]
            logger_instance.info(f"LLM_RESPONSE | Response: {response}")

    def log_tool_usage(self, logger_id: str, tool_name: str, arguments: Dict[str, Any]):
        """Log tool usage with arguments."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]
            logger_instance.info(
                f"TOOL_USAGE | Tool: {tool_name}, Arguments: {json.dumps(arguments)}"
            )

    def log_cost(self, logger_id: str, cost: float):
        """Log cost information."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]
            logger_instance.info(f"COST | Cost: ${cost:.6f}")

    def log_interaction(
        self,
        logger_id: str,
        input_msg: str,
        output_msg: str,
        token_usage: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        tools_used: Optional[Dict[str, Any]] = None,
    ):
        """Log a complete agent interaction with all relevant information."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]

            log_parts = [f"INTERACTION | Input: {input_msg}"]

            if output_msg:
                log_parts.append(f"Output: {output_msg}")

            if token_usage:
                log_parts.append(
                    f"Tokens - Prompt: {token_usage.get('prompt_tokens', 0)}, "
                    f"Completion: {token_usage.get('completion_tokens', 0)}, "
                    f"Total: {token_usage.get('total_tokens', 0)}"
                )

            if cost is not None:
                log_parts.append(f"Cost: ${cost:.6f}")

            if tools_used:
                log_parts.append(f"Tools: {json.dumps(tools_used)}")

            logger_instance.info(" | ".join(log_parts))

    def log_event(self, logger_id: str, event_type: str, data: Dict[str, Any]):
        """Log a general event with custom data."""
        if logger_id in self._loggers:
            logger_instance, _ = self._loggers[logger_id]
            logger_instance.info(f"{event_type.upper()} | {json.dumps(data)}")


# Global logger instance
agent_logger = AgentLogger()
