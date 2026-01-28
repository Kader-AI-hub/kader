
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from kader.tools.agent import AgentTool


class TestAgentToolPersistence(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a temporary directory for tests
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Patch Path.home() to return our temp dir
        self.home_patcher = patch("pathlib.Path.home", return_value=self.test_dir)
        self.mock_home = self.home_patcher.start()

    def tearDown(self):
        self.home_patcher.stop()
        shutil.rmtree(self.test_dir)

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_creates_persistence_file_standalone(self, mock_get_registry, mock_agent_cls):
        # Setup mocks
        mock_registry = MagicMock()
        mock_registry.tools = []
        mock_get_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = MagicMock(content="Task completed")
        mock_agent_cls.return_value = mock_agent_instance

        # Create tool and execute (no session ID)
        tool = AgentTool(name="test_agent", interrupt_before_tool=False)
        result = tool.execute(task="Do something", context="Some context")

        # Verify result
        self.assertEqual(result, "Task completed")

        # Verify directory structure for standalone
        # Expected: .../executors/test_agent-<uuid>/conversation.json
        executors_dir = self.test_dir / ".kader" / "memory" / "sessions" / "standalone" / "executors"
        self.assertTrue(executors_dir.exists())

        # Find the specific agent execution directory
        agent_dirs = list(executors_dir.glob("test_agent-*"))
        self.assertEqual(len(agent_dirs), 1)
        agent_dir = agent_dirs[0]
        self.assertTrue(agent_dir.is_dir())

        # Verify conversation.json exists
        conversation_file = agent_dir / "conversation.json"
        self.assertTrue(conversation_file.exists())
        
        # Verify file content
        with open(conversation_file, "r") as f:
            data = json.load(f)
            self.assertIn("messages", data)
            self.assertTrue(len(data["messages"]) >= 1)
            self.assertEqual(data["messages"][0]["role"], "user")
            self.assertEqual(data["messages"][0]["content"], "Some context")

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    def test_execute_creates_persistence_file_with_session(self, mock_get_registry, mock_agent_cls):
        # Setup mocks
        mock_registry = MagicMock()
        mock_registry.tools = []
        mock_get_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        mock_agent_instance.invoke.return_value = MagicMock(content="Task completed")
        mock_agent_cls.return_value = mock_agent_instance

        # Create tool and set session ID
        tool = AgentTool(name="test_agent_session", interrupt_before_tool=False)
        tool.set_session_id("my-session-id")
        result = tool.execute(task="Do something", context="Some context")

        # Verify result
        self.assertEqual(result, "Task completed")

        # Verify directory structure for specific session
        executors_dir = self.test_dir / ".kader" / "memory" / "sessions" / "my-session-id" / "executors"
        self.assertTrue(executors_dir.exists())

        # Find the specific agent execution directory
        agent_dirs = list(executors_dir.glob("test_agent_session-*"))
        self.assertEqual(len(agent_dirs), 1)
        agent_dir = agent_dirs[0]
        self.assertTrue(agent_dir.is_dir())

        # Verify conversation.json exists
        conversation_file = agent_dir / "conversation.json"
        self.assertTrue(conversation_file.exists())

    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.get_default_registry")
    async def test_aexecute_creates_persistence_file_async(self, mock_get_registry, mock_agent_cls):
        # Setup mocks
        mock_registry = MagicMock()
        mock_registry.tools = []
        mock_get_registry.return_value = mock_registry

        mock_agent_instance = MagicMock()
        # Mock ainvoke for async
        mock_agent_instance.ainvoke = AsyncMock(return_value=MagicMock(content="Async Task completed"))
        mock_agent_cls.return_value = mock_agent_instance

        # Create tool and execute async
        tool = AgentTool(name="test_agent_async", interrupt_before_tool=False)
        result = await tool.aexecute(task="Do something async", context="Async context")

        # Verify result
        self.assertEqual(result, "Async Task completed")

        # Verify directory structure for standalone
        executors_dir = self.test_dir / ".kader" / "memory" / "sessions" / "standalone" / "executors"
        self.assertTrue(executors_dir.exists())

        # Find the specific agent execution directory
        agent_dirs = list(executors_dir.glob("test_agent_async-*"))
        self.assertEqual(len(agent_dirs), 1)
        agent_dir = agent_dirs[0]
        self.assertTrue(agent_dir.is_dir())

        # Verify conversation.json exists
        conversation_file = agent_dir / "conversation.json"
        self.assertTrue(conversation_file.exists())

if __name__ == "__main__":
    unittest.main()
