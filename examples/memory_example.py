"""
Memory Module Example

Demonstrates how to use the Kader memory module for:
- State management (AgentState, RequestState)
- Session management (FileSessionManager)
- Conversation management (SlidingWindowConversationManager)

Memory is persisted in $HOME/.kader/memory/sessions/
"""

import os
import sys

# Add project root to path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.memory import (
    AgentState,
    FileSessionManager,
    RequestState,
    SlidingWindowConversationManager,
)
from kader.providers.base import Message


def demo_agent_state():
    """Demonstrate AgentState for persistent key-value storage."""
    print("\n=== Agent State Demo ===")

    state = AgentState(agent_id="demo-agent")

    # Set values
    state.set("user_name", "Alice")
    state.set("preferences", {"theme": "dark", "language": "en"})
    state.set("interaction_count", 0)

    # Get values
    print(f"User: {state.get('user_name')}")
    print(f"Preferences: {state.get('preferences')}")
    print(f"Missing key with default: {state.get('missing', 'default_value')}")

    # Update value
    state.set("interaction_count", state.get("interaction_count") + 1)
    print(f"Interaction count: {state.get('interaction_count')}")

    # Check containment
    print(f"Has 'user_name': {'user_name' in state}")
    print(f"Total keys: {len(state)}")

    # Serialize for persistence
    data = state.to_dict()
    print(f"Serialized: {data}")

    # Restore from serialized data
    restored = AgentState.from_dict(data)
    print(f"Restored user: {restored.get('user_name')}")


def demo_request_state():
    """Demonstrate RequestState for ephemeral request context."""
    print("\n=== Request State Demo ===")

    # Create request-scoped state (not persisted)
    request_state = RequestState(request_id="req-12345")

    # Store temporary data
    request_state.set("start_time", "2025-01-01T12:00:00Z")
    request_state.set("intermediate_result", {"step": 1, "data": "processing"})

    print(f"Start time: {request_state.get('start_time')}")
    print(f"Intermediate: {request_state.get('intermediate_result')}")

    # Clear when done
    request_state.clear()
    print(f"After clear, keys: {len(request_state)}")


def demo_session_manager():
    """Demonstrate FileSessionManager for session persistence."""
    print("\n=== Session Manager Demo ===")

    # Create session manager (uses $HOME/.kader/memory by default)
    manager = FileSessionManager()

    # Create a new session
    session = manager.create_session("my-agent")
    print(f"Created session: {session.session_id}")
    print(f"Agent ID: {session.agent_id}")
    print(f"Created at: {session.created_at}")

    # Save agent state to session
    state = AgentState(agent_id="my-agent")
    state.set("counter", 42)
    state.set("last_topic", "memory examples")
    manager.save_agent_state(session.session_id, state)
    print("Saved agent state")

    # Save conversation to session
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "Tell me about memory management."},
    ]
    manager.save_conversation(session.session_id, messages)
    print(f"Saved {len(messages)} messages")

    # Load from session
    loaded_state = manager.load_agent_state(session.session_id, "my-agent")
    loaded_messages = manager.load_conversation(session.session_id)
    print(f"Loaded state counter: {loaded_state.get('counter')}")
    print(f"Loaded {len(loaded_messages)} messages")

    # List all sessions for this agent
    sessions = manager.list_sessions("my-agent")
    print(f"Total sessions for my-agent: {len(sessions)}")

    # Get session by ID
    retrieved = manager.get_session(session.session_id)
    print(f"Retrieved session: {retrieved.session_id}")

    # Clean up demo session
    manager.delete_session(session.session_id)
    print("Deleted demo session")


def demo_conversation_manager():
    """Demonstrate SlidingWindowConversationManager for context windowing."""
    print("\n=== Conversation Manager Demo ===")

    # Create conversation manager with window of 3 message pairs
    conv_manager = SlidingWindowConversationManager(window_size=3)

    # Add messages from Message objects
    conv_manager.add_message(Message.user("What's the weather?"))
    conv_manager.add_message(Message.assistant("It's sunny today!"))
    conv_manager.add_message(Message.user("How about tomorrow?"))
    conv_manager.add_message(Message.assistant("Rain is expected."))

    print(f"Total messages: {len(conv_manager)}")

    # Get windowed messages for LLM
    windowed = conv_manager.apply_window()
    print(f"After windowing: {len(windowed)} messages")
    for msg in windowed:
        print(f"  [{msg['role']}]: {msg['content'][:50]}...")

    # Persist and restore state
    state = conv_manager.get_state()

    new_manager = SlidingWindowConversationManager()
    new_manager.set_state(state)
    print(f"Restored {len(new_manager)} messages")


def demo_full_workflow():
    """Demonstrate a complete workflow with session + state + conversation."""
    print("\n=== Full Workflow Demo ===")

    # Initialize components
    session_manager = FileSessionManager()
    conv_manager = SlidingWindowConversationManager(window_size=20)

    # Create session
    session = session_manager.create_session("workflow-agent")
    state = AgentState(agent_id="workflow-agent")

    # Simulate conversation
    state.set("topic", "Python programming")

    conv_manager.add_message(Message.user("Teach me Python"))
    conv_manager.add_message(Message.assistant("Sure! Python is a great language."))
    conv_manager.add_message(Message.user("What about loops?"))
    conv_manager.add_message(Message.assistant("Python has for and while loops."))

    state.set("messages_processed", 4)

    # Persist everything
    session_manager.save_agent_state(session.session_id, state)
    messages = [msg.message for msg in conv_manager.get_messages()]
    session_manager.save_conversation(session.session_id, messages)

    print(f"Session: {session.session_id}")
    print(f"Topic: {state.get('topic')}")
    print(f"Messages: {state.get('messages_processed')}")

    # Clean up
    session_manager.delete_session(session.session_id)
    print("Workflow complete!")


if __name__ == "__main__":
    demo_agent_state()
    demo_request_state()
    demo_session_manager()
    demo_conversation_manager()
    demo_full_workflow()

    print("\n[OK] All demos completed successfully!")
