# Memory Management

Kader provides comprehensive memory management for agent conversations and state persistence.

## AgentState

Key-value storage for agent data:

```python
from kader.memory import AgentState

state = AgentState(agent_id="my-agent")

# Set values
state.set("user_name", "Alice")
state.set("preferences", {"theme": "dark", "language": "en"})

# Get values
print(state.get("user_name"))
print(state.get("preferences"))

# Check existence
if "user_name" in state:
    print("User name is set")
```

## Session Management

Persist sessions to disk:

```python
from kader.memory import FileSessionManager, AgentState

session_mgr = FileSessionManager()

# Create new session
session = session_mgr.create_session("my-agent")
print(f"Session ID: {session.session_id}")

# Save agent state
state = AgentState(agent_id="my-agent")
state.set("counter", 0)
session_mgr.save_agent_state(session.session_id, state)

# Load session later
loaded_state = session_mgr.load_agent_state(session.session_id)
```

### Session Storage Location

Sessions are saved to `~/.kader/memory/sessions/` with the following structure:

```
~/.kader/memory/sessions/<session-id>/
├── conversation.json       # Message history
├── agent_state.json       # Agent state
├── executor/             # Sub-agent contexts
│   └── <sub-agent-id>/
│       └── conversation.json
└── checkpoint.md         # Aggregated context
```

## Conversation Management

### SlidingWindowConversationManager

Maintain context within token limits:

```python
from kader.memory import SlidingWindowConversationManager
from kader.providers import Message

conv_mgr = SlidingWindowConversationManager(window_size=10)

# Add messages
conv_mgr.add_message(Message.user("Hello"))
conv_mgr.add_message(Message.assistant("Hi there!"))
conv_mgr.add_message(Message.user("How are you?"))

# Get messages (automatically manages window)
messages = conv_mgr.get_messages()

# Persist conversation
session_mgr.save_conversation(
    session_id,
    [msg.message for msg in conv_mgr.get_messages()]
)
```

### PersistentSlidingWindowConversationManager

Auto-saves sub-agent history:

```python
from kader.memory import PersistentSlidingWindowConversationManager
from kader.providers import Message

conv_mgr = PersistentSlidingWindowConversationManager(
    session_id="my-session",
    window_size=20,
    persistence_dir=Path("~/.kader/memory/sessions")
)
```

## Tool Output Compression

Compress tool outputs to save token space:

```python
from kader.memory import ToolOutputCompressor

compressor = ToolOutputCompressor(
    max_length=1000,
    summary_type="truncate",  # or "first_last"
)

compressed = compressor.compress(
    tool_name="read_file",
    output="Very long file content..."
)
```

### Compression Types

| Type | Description |
|------|-------------|
| `truncate` | Keep first N characters |
| `first_last` | Keep first and last N characters |

## Context Aggregation

Aggregate sub-agent contexts for the main session:

```python
from kader.utils.context_aggregator import ContextAggregator

aggregator = ContextAggregator()

# Add sub-agent context
aggregator.add_context(
    agent_id="research-agent",
    context="Research findings about..."
)

# Get aggregated context
summary = aggregator.get_summary()
```

## Checkpoint Generation

Generate markdown summaries of agent actions:

```python
from kader.utils.checkpointer import Checkpointer

checkpointer = Checkpointer()

# Record agent action
checkpointer.record_action(
    agent_name="PlannerAgent",
    action="Created todo list",
    details=["Step 1: Setup", "Step 2: Implement"]
)

# Generate checkpoint
checkpoint = checkpointer.generate_checkpoint()
```

## Memory Types Summary

| Class | Use Case |
|-------|----------|
| `AgentState` | Key-value storage for agent data |
| `FileSessionManager` | Persist sessions to disk |
| `SlidingWindowConversationManager` | Manage conversation history within token limits |
| `PersistentSlidingWindowConversationManager` | Auto-save sub-agent history |
| `ToolOutputCompressor` | Compress tool outputs |
| `ContextAggregator` | Aggregate sub-agent contexts |
| `Checkpointer` | Generate action summaries |
