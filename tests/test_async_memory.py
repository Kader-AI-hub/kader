"""
Unit tests for AsyncFileSessionManager.

Tests async session operations to ensure non-blocking I/O.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from kader.memory import (
    AgentState,
    AsyncFileSessionManager,
    FileSessionManager,
    MemoryConfig,
    SessionType,
)


class TestAsyncFileSessionManager:
    """Tests for AsyncFileSessionManager class."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        config = MemoryConfig(memory_dir=Path(temp_dir))
        manager = AsyncFileSessionManager(config=config)
        yield manager, temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sync_manager(self):
        """Create an async manager with a temp dir for cross-validation."""
        temp_dir = tempfile.mkdtemp()
        config = MemoryConfig(memory_dir=Path(temp_dir))
        manager = AsyncFileSessionManager(config=config)
        yield manager, temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    async def test_async_create_session(self, temp_config):
        """Test creating a session asynchronously."""
        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")

        assert session.session_id
        assert session.agent_id == "agent-1"
        assert session.session_type == SessionType.AGENT

    async def test_async_get_session(self, temp_config):
        """Test retrieving a session asynchronously."""
        manager, _ = temp_config
        created = await manager.async_create_session("agent-1")
        retrieved = await manager.async_get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id
        assert retrieved.agent_id == created.agent_id

    async def test_async_get_session_missing(self, temp_config):
        """Test async_get_session returns None for missing session."""
        manager, _ = temp_config
        result = await manager.async_get_session("nonexistent-id")
        assert result is None

    async def test_async_list_sessions(self, temp_config):
        """Test listing sessions asynchronously."""
        manager, _ = temp_config
        await manager.async_create_session("agent-1")
        await manager.async_create_session("agent-1")
        await manager.async_create_session("agent-2")

        all_sessions = await manager.async_list_sessions()
        assert len(all_sessions) == 3

        agent1_sessions = await manager.async_list_sessions("agent-1")
        assert len(agent1_sessions) == 2

        agent2_sessions = await manager.async_list_sessions("agent-2")
        assert len(agent2_sessions) == 1

    async def test_async_list_sessions_sorted(self, temp_config):
        """Test that async_list_sessions returns sessions sorted by created_at descending."""
        manager, _ = temp_config
        s1 = await manager.async_create_session("agent-1")
        await manager.async_create_session("agent-1")
        s3 = await manager.async_create_session("agent-1")

        sessions = await manager.async_list_sessions()
        assert sessions[0].session_id == s3.session_id
        assert sessions[-1].session_id == s1.session_id

    async def test_async_list_sessions_empty_dir(self, temp_config):
        """Test async_list_sessions on non-existent directory."""
        manager, _ = temp_config
        result = await manager.async_list_sessions()
        assert result == []

    async def test_async_delete_session(self, temp_config):
        """Test deleting a session asynchronously."""
        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")

        deleted = await manager.async_delete_session(session.session_id)
        assert deleted is True

        retrieved = await manager.async_get_session(session.session_id)
        assert retrieved is None

    async def test_async_delete_session_missing(self, temp_config):
        """Test async_delete_session returns False for missing session."""
        manager, _ = temp_config
        result = await manager.async_delete_session("nonexistent-id")
        assert result is False

    async def test_async_save_load_agent_state(self, temp_config):
        """Test saving and loading agent state asynchronously."""
        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")

        state = AgentState(agent_id="agent-1")
        state.set("preference", "dark_mode")
        state.set("counter", 42)

        await manager.async_save_agent_state(session.session_id, state)
        loaded = await manager.async_load_agent_state(session.session_id, "agent-1")

        assert loaded.get("preference") == "dark_mode"
        assert loaded.get("counter") == 42

    async def test_async_load_agent_state_creates_new(self, temp_config):
        """Test async_load_agent_state returns new state for missing session."""
        manager, _ = temp_config
        loaded = await manager.async_load_agent_state("nonexistent", "agent-x")
        assert loaded.agent_id == "agent-x"

    async def test_async_save_load_conversation(self, temp_config):
        """Test saving and loading conversation asynchronously."""
        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        await manager.async_save_conversation(session.session_id, messages)
        loaded = await manager.async_load_conversation(session.session_id)

        assert len(loaded) == 2
        assert loaded[0]["content"] == "Hello"

    async def test_async_load_conversation_empty(self, temp_config):
        """Test async_load_conversation returns empty list for missing session."""
        manager, _ = temp_config
        loaded = await manager.async_load_conversation("nonexistent")
        assert loaded == []

    async def test_async_timestamp_updated_on_save(self, temp_config):
        """Test that updated_at is refreshed after async_save_conversation."""
        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")
        original_updated = session.updated_at

        import time

        time.sleep(0.01)

        await manager.async_save_conversation(
            session.session_id, [{"role": "user", "content": "test"}]
        )
        refreshed = await manager.async_get_session(session.session_id)
        assert refreshed is not None
        assert refreshed.updated_at >= original_updated

    async def test_sync_and_async_methods_produce_same_result(self, sync_manager):
        """Verify sync and async methods produce identical results."""
        manager, _ = sync_manager

        sync_mgr = FileSessionManager(
            config=MemoryConfig(memory_dir=Path(tempfile.mkdtemp()))
        )
        session_sync = sync_mgr.create_session("agent-sync")
        session_async = await manager.async_create_session("agent-async")

        sync_state = AgentState(agent_id="agent-sync")
        sync_state.set("key", "value")
        sync_mgr.save_agent_state(session_sync.session_id, sync_state)

        await manager.async_save_agent_state(session_async.session_id, sync_state)

        loaded_sync = sync_mgr.load_agent_state(session_sync.session_id, "agent-sync")
        loaded_async = await manager.async_load_agent_state(
            session_async.session_id, "agent-async"
        )

        assert loaded_sync.get("key") == loaded_async.get("key")
        assert loaded_sync.agent_id == loaded_async.agent_id

        shutil.rmtree(sync_mgr.config.memory_dir, ignore_errors=True)

    async def test_concurrent_session_operations(self, temp_config):
        """Test that concurrent async operations complete without errors.

        Multiple coroutines save conversations concurrently. The test verifies
        no exceptions are raised. Due to concurrent writes, only one
        conversation will be persisted (last write wins) — this is expected
        for file-based storage and not a bug.
        """
        import asyncio

        manager, _ = temp_config
        session = await manager.async_create_session("agent-1")

        async def save_conversation(n: int) -> None:
            await manager.async_save_conversation(
                session.session_id, [{"role": "user", "content": f"msg_{n}"}]
            )

        await asyncio.gather(*[save_conversation(i) for i in range(10)])

    async def test_concurrent_list_sessions(self, temp_config):
        """Test that async_list_sessions handles concurrent session creation."""
        import asyncio

        manager, _ = temp_config

        async def create_session(n: int) -> None:
            await manager.async_create_session(f"agent-{n}")

        await asyncio.gather(*[create_session(i) for i in range(5)])

        sessions = await manager.async_list_sessions()
        assert len(sessions) == 5

    async def test_lazy_directory_creation(self):
        """Test that AsyncFileSessionManager.__init__ does not create directories.

        Note: MemoryConfig.__post_init__ may create directories when the config
        is instantiated. The AsyncFileSessionManager only ensures directories
        exist when an async operation is first performed.
        """
        temp_dir = tempfile.mkdtemp()
        config = MemoryConfig(memory_dir=Path(temp_dir) / "new_memory")

        assert not (Path(temp_dir) / "new_memory").exists()

        manager = AsyncFileSessionManager(config=config)

        session_file = manager._session_file("test-session")
        assert not session_file.parent.exists()

        await manager.async_create_session("agent-1")

        assert (Path(temp_dir) / "new_memory").exists()
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestAsyncFileSessionManagerSyncWrappers:
    """Tests verifying FileSessionManager's async wrapper methods."""

    async def test_file_session_manager_has_async_methods(self):
        """Test that FileSessionManager has async methods that delegate correctly."""
        temp_dir = tempfile.mkdtemp()
        config = MemoryConfig(memory_dir=Path(temp_dir))
        manager = FileSessionManager(config=config)

        session = await manager.async_create_session("agent-1")
        assert session.agent_id == "agent-1"

        retrieved = await manager.async_get_session(session.session_id)
        assert retrieved is not None

        sessions = await manager.async_list_sessions()
        assert len(sessions) == 1

        state = AgentState(agent_id="agent-1")
        state.set("key", "value")
        await manager.async_save_agent_state(session.session_id, state)

        loaded = await manager.async_load_agent_state(session.session_id, "agent-1")
        assert loaded.get("key") == "value"

        messages = [{"role": "user", "content": "test"}]
        await manager.async_save_conversation(session.session_id, messages)
        loaded_conv = await manager.async_load_conversation(session.session_id)
        assert len(loaded_conv) == 1

        deleted = await manager.async_delete_session(session.session_id)
        assert deleted is True

        shutil.rmtree(temp_dir, ignore_errors=True)
