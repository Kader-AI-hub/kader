"""
Unit tests for the file-based Memory module.

Tests state management, session management, and conversation management.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from kader.memory import (
    AgentState,
    ConversationMessage,
    FileSessionManager,
    MemoryConfig,
    NullConversationManager,
    RequestState,
    SessionType,
    SlidingWindowConversationManager,
)
from kader.providers.base import Message


class TestAgentState(unittest.TestCase):
    """Tests for AgentState class."""

    def test_set_get(self):
        """Test setting and getting values."""
        state = AgentState(agent_id="test-agent")
        state.set("key1", "value1")
        state.set("key2", 42)

        self.assertEqual(state.get("key1"), "value1")
        self.assertEqual(state.get("key2"), 42)
        self.assertIsNone(state.get("nonexistent"))
        self.assertEqual(state.get("nonexistent", "default"), "default")

    def test_delete(self):
        """Test deleting values."""
        state = AgentState(agent_id="test-agent")
        state.set("key1", "value1")

        self.assertTrue(state.delete("key1"))
        self.assertFalse(state.delete("nonexistent"))
        self.assertIsNone(state.get("key1"))

    def test_get_all(self):
        """Test getting all state."""
        state = AgentState(agent_id="test-agent")
        state.set("a", 1)
        state.set("b", 2)

        all_state = state.get_all()
        self.assertEqual(all_state, {"a": 1, "b": 2})

    def test_clear(self):
        """Test clearing state."""
        state = AgentState(agent_id="test-agent")
        state.set("key1", "value1")
        state.clear()

        self.assertEqual(len(state), 0)

    def test_contains(self):
        """Test containment check."""
        state = AgentState(agent_id="test-agent")
        state.set("key1", "value1")

        self.assertIn("key1", state)
        self.assertNotIn("key2", state)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        state = AgentState(agent_id="test-agent")
        state.set("key1", "value1")

        data = state.to_dict()
        restored = AgentState.from_dict(data)

        self.assertEqual(restored.agent_id, "test-agent")
        self.assertEqual(restored.get("key1"), "value1")


class TestRequestState(unittest.TestCase):
    """Tests for RequestState class."""

    def test_ephemeral_state(self):
        """Test ephemeral request state."""
        state = RequestState(request_id="req-123")
        state.set("temp_data", {"x": 1})

        self.assertEqual(state.get("temp_data"), {"x": 1})
        state.clear()
        self.assertEqual(len(state), 0)


class TestFileSessionManager(unittest.TestCase):
    """Tests for FileSessionManager class."""

    def setUp(self):
        """Create a temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = MemoryConfig(memory_dir=Path(self.temp_dir))
        self.manager = FileSessionManager(config=self.config)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_create_session(self):
        """Test creating a session."""
        session = self.manager.create_session("agent-1")

        self.assertIsNotNone(session.session_id)
        self.assertEqual(session.agent_id, "agent-1")
        self.assertEqual(session.session_type, SessionType.AGENT)

    def test_get_session(self):
        """Test retrieving a session."""
        created = self.manager.create_session("agent-1")
        retrieved = self.manager.get_session(created.session_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.session_id, created.session_id)
        self.assertEqual(retrieved.agent_id, created.agent_id)

    def test_list_sessions(self):
        """Test listing sessions."""
        self.manager.create_session("agent-1")
        self.manager.create_session("agent-1")
        self.manager.create_session("agent-2")

        all_sessions = self.manager.list_sessions()
        self.assertEqual(len(all_sessions), 3)

        agent1_sessions = self.manager.list_sessions("agent-1")
        self.assertEqual(len(agent1_sessions), 2)

    def test_delete_session(self):
        """Test deleting a session."""
        session = self.manager.create_session("agent-1")

        self.assertTrue(self.manager.delete_session(session.session_id))
        self.assertFalse(self.manager.delete_session(session.session_id))
        self.assertIsNone(self.manager.get_session(session.session_id))

    def test_save_load_state(self):
        """Test saving and loading agent state."""
        session = self.manager.create_session("agent-1")

        state = AgentState(agent_id="agent-1")
        state.set("preference", "dark_mode")
        state.set("counter", 42)

        self.manager.save_agent_state(session.session_id, state)
        loaded = self.manager.load_agent_state(session.session_id, "agent-1")

        self.assertEqual(loaded.get("preference"), "dark_mode")
        self.assertEqual(loaded.get("counter"), 42)

    def test_save_load_conversation(self):
        """Test saving and loading conversation."""
        session = self.manager.create_session("agent-1")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        self.manager.save_conversation(session.session_id, messages)
        loaded = self.manager.load_conversation(session.session_id)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["content"], "Hello")


class TestConversationMessage(unittest.TestCase):
    """Tests for ConversationMessage class."""

    def test_from_dict(self):
        """Test creating from dict."""
        msg_dict = {"role": "user", "content": "Hello"}
        conv_msg = ConversationMessage.from_message(msg_dict, 0)

        self.assertEqual(conv_msg.role, "user")
        self.assertEqual(conv_msg.content, "Hello")

    def test_from_message_object(self):
        """Test creating from Message object."""
        msg = Message.user("Hello from user")
        conv_msg = ConversationMessage.from_message(msg, 0)

        self.assertEqual(conv_msg.role, "user")
        self.assertEqual(conv_msg.content, "Hello from user")

    def test_serialization(self):
        """Test to_dict and from_dict."""
        msg_dict = {"role": "assistant", "content": "Response"}
        conv_msg = ConversationMessage.from_message(msg_dict, 5)

        data = conv_msg.to_dict()
        restored = ConversationMessage.from_dict(data)

        self.assertEqual(restored.message_id, 5)
        self.assertEqual(restored.content, "Response")


class TestSlidingWindowConversationManager(unittest.TestCase):
    """Tests for SlidingWindowConversationManager class."""

    def test_add_message(self):
        """Test adding messages."""
        manager = SlidingWindowConversationManager(window_size=10)

        manager.add_message({"role": "user", "content": "Hello"})
        manager.add_message({"role": "assistant", "content": "Hi"})

        self.assertEqual(len(manager), 2)

    def test_add_messages(self):
        """Test adding multiple messages."""
        manager = SlidingWindowConversationManager()
        messages = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": "2"},
        ]

        manager.add_messages(messages)
        self.assertEqual(len(manager), 2)

    def test_sliding_window(self):
        """Test sliding window truncation."""
        manager = SlidingWindowConversationManager(window_size=2)

        # Add more than window_size * 2 messages
        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg{i}"})

        windowed = manager.apply_window()
        # Should keep at most window_size * 2 = 4 messages
        self.assertLessEqual(len(windowed), 4)

    def test_state_persistence(self):
        """Test state save and restore."""
        manager1 = SlidingWindowConversationManager(window_size=10)
        manager1.add_message({"role": "user", "content": "Hello"})

        state = manager1.get_state()

        manager2 = SlidingWindowConversationManager()
        manager2.set_state(state)

        self.assertEqual(len(manager2), 1)
        self.assertEqual(manager2.get_messages()[0].content, "Hello")

    def test_clear(self):
        """Test clearing conversation."""
        manager = SlidingWindowConversationManager()
        manager.add_message({"role": "user", "content": "Hello"})
        manager.clear()

        self.assertEqual(len(manager), 0)


class TestNullConversationManager(unittest.TestCase):
    """Tests for NullConversationManager class."""

    def test_no_storage(self):
        """Test that messages are not stored."""
        manager = NullConversationManager()
        manager.add_message({"role": "user", "content": "Hello"})

        self.assertEqual(len(manager.get_messages()), 0)

    def test_apply_window_empty(self):
        """Test that apply_window returns empty."""
        manager = NullConversationManager()
        manager.add_message({"role": "user", "content": "Hello"})

        self.assertEqual(manager.apply_window(), [])


if __name__ == "__main__":
    unittest.main()
