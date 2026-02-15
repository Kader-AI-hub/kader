"""
Unit tests for HierarchicalConversationManager.

Tests extraction-based summarization, windowing behavior,
state persistence, and fallback behavior.
"""

import unittest

from kader.memory import (
    ConversationSummary,
    HierarchicalConversationManager,
    SlidingWindowConversationManager,
)


class TestConversationSummary(unittest.TestCase):
    """Tests for ConversationSummary dataclass."""

    def test_to_dict_roundtrip(self):
        """Test serialization and deserialization."""
        summary = ConversationSummary(
            summary_id="test-123",
            original_message_count=10,
            summary_text="Task involved editing files.",
            key_decisions=["Use extraction approach"],
            files_modified=["src/main.py"],
            errors_encountered=["FileNotFoundError: missing.py"],
            current_state="In progress",
            pending_tasks=["Write tests"],
        )

        data = summary.to_dict()
        restored = ConversationSummary.from_dict(data)

        self.assertEqual(restored.summary_id, "test-123")
        self.assertEqual(restored.original_message_count, 10)
        self.assertEqual(restored.summary_text, "Task involved editing files.")
        self.assertEqual(restored.key_decisions, ["Use extraction approach"])
        self.assertEqual(restored.files_modified, ["src/main.py"])
        self.assertEqual(
            restored.errors_encountered, ["FileNotFoundError: missing.py"]
        )
        self.assertEqual(restored.current_state, "In progress")
        self.assertEqual(restored.pending_tasks, ["Write tests"])

    def test_from_dict_defaults(self):
        """Test deserialization with missing fields uses defaults."""
        summary = ConversationSummary.from_dict({})

        self.assertEqual(summary.original_message_count, 0)
        self.assertEqual(summary.summary_text, "")
        self.assertEqual(summary.key_decisions, [])
        self.assertEqual(summary.files_modified, [])


class TestHierarchicalConversationManager(unittest.TestCase):
    """Tests for HierarchicalConversationManager."""

    def test_inherits_sliding_window(self):
        """Test that it is a subclass of SlidingWindowConversationManager."""
        manager = HierarchicalConversationManager()
        self.assertIsInstance(manager, SlidingWindowConversationManager)

    def test_under_threshold_returns_all(self):
        """Test that messages under summarization threshold are returned as-is."""
        manager = HierarchicalConversationManager(
            window_size=20, summarization_threshold=0.8
        )

        # Add a few messages (well under 80% of 20 pairs = 16 pairs = 32 messages)
        for i in range(6):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()
        self.assertEqual(len(result), 12)  # 6 pairs = 12 messages

    def test_over_threshold_triggers_summarization(self):
        """Test that exceeding the threshold triggers summarization."""
        manager = HierarchicalConversationManager(
            window_size=10,
            full_context_window=2,
            summarization_threshold=0.5,
        )

        # Add 10 pairs (= 20 messages), threshold is 5 pairs
        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()

        # Should have: 1 summary system message + 4 recent messages (2 pairs)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["role"], "system")
        self.assertIn("Previous Context", result[0]["content"])

    def test_recent_messages_preserved_verbatim(self):
        """Test that the most recent messages are kept exactly as-is."""
        manager = HierarchicalConversationManager(
            window_size=10,
            full_context_window=2,
            summarization_threshold=0.5,
        )

        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()

        # Last 4 messages (2 pairs) should be the most recent
        recent = result[1:]  # Skip summary message
        self.assertEqual(recent[0]["content"], "msg-8")
        self.assertEqual(recent[1]["content"], "reply-8")
        self.assertEqual(recent[2]["content"], "msg-9")
        self.assertEqual(recent[3]["content"], "reply-9")

    def test_extract_files_modified(self):
        """Test extraction of file paths from write_file tool calls."""
        manager = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
        )

        # Add an assistant message with a write_file tool call
        manager.add_message(
            {
                "role": "assistant",
                "content": "I'll write the file.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"file_path": "src/app.py", "content": "print()"}',
                        },
                    }
                ],
            }
        )
        manager.add_message(
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "File written successfully.",
            }
        )

        # Add more messages to cross the threshold
        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()
        summary_content = result[0]["content"]

        self.assertIn("src/app.py", summary_content)

    def test_extract_errors(self):
        """Test extraction of errors from tool results."""
        manager = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
        )

        # Add a tool result with an error
        manager.add_message(
            {
                "role": "assistant",
                "content": "Running...",
                "tool_calls": [
                    {
                        "id": "call_2",
                        "function": {"name": "command", "arguments": "{}"},
                    }
                ],
            }
        )
        manager.add_message(
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": "FileNotFoundError: No such file 'missing.py'",
            }
        )

        # Add more messages to cross the threshold
        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()
        summary_content = result[0]["content"]

        self.assertIn("Issues Encountered", summary_content)
        self.assertIn("FileNotFoundError", summary_content)

    def test_extract_key_decisions(self):
        """Test extraction of key decisions from user messages."""
        manager = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
        )

        # Add a user message with a decision
        manager.add_message(
            {"role": "user", "content": "Let's use the extraction approach for this."}
        )
        manager.add_message(
            {"role": "assistant", "content": "Got it, using extraction."}
        )

        # Add more messages to cross threshold
        for i in range(10):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()
        summary_content = result[0]["content"]

        self.assertIn("Key Decisions", summary_content)
        self.assertIn("extraction approach", summary_content)

    def test_empty_messages(self):
        """Test apply_window with no messages."""
        manager = HierarchicalConversationManager()
        self.assertEqual(manager.apply_window(), [])

    def test_state_persistence_roundtrip(self):
        """Test get_state/set_state preserves summaries."""
        manager1 = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
        )

        # Add enough messages to trigger summarization
        for i in range(8):
            manager1.add_message({"role": "user", "content": f"msg-{i}"})
            manager1.add_message({"role": "assistant", "content": f"reply-{i}"})

        # Trigger summarization by calling apply_window
        manager1.apply_window()

        # Save state
        state = manager1.get_state()

        # Restore into a new manager
        manager2 = HierarchicalConversationManager()
        manager2.set_state(state)

        self.assertEqual(manager2.full_context_window, 1)
        self.assertEqual(manager2.summarization_threshold, 0.5)
        self.assertEqual(len(manager2._summaries), len(manager1._summaries))

        # Verify summaries are restored
        if manager1._summaries:
            self.assertEqual(
                manager2._summaries[0].summary_id,
                manager1._summaries[0].summary_id,
            )

    def test_no_provider_fallback(self):
        """Test fallback behavior when no LLM provider is available."""
        manager = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
            provider=None,
        )

        for i in range(8):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()

        # Summary should use fallback text
        summary_content = result[0]["content"]
        self.assertIn("Previous Context", summary_content)
        self.assertIn("messages summarized", summary_content)

    def test_format_summary(self):
        """Test the _format_summary method produces valid output."""
        manager = HierarchicalConversationManager()

        summary = ConversationSummary(
            summary_id="test",
            original_message_count=20,
            summary_text="Worked on the parser module.",
            key_decisions=["Use regex for parsing"],
            files_modified=["parser.py", "tests/test_parser.py"],
            errors_encountered=["SyntaxError in first attempt"],
            current_state="Parser is working",
            pending_tasks=["Add edge case tests"],
        )

        formatted = manager._format_summary(summary)

        self.assertIn("20 messages summarized", formatted)
        self.assertIn("Worked on the parser module", formatted)
        self.assertIn("parser.py", formatted)
        self.assertIn("SyntaxError", formatted)
        self.assertIn("Add edge case tests", formatted)

    def test_clear(self):
        """Test clearing the manager."""
        manager = HierarchicalConversationManager()
        manager.add_message({"role": "user", "content": "Hello"})
        manager.clear()

        self.assertEqual(len(manager), 0)
        self.assertEqual(manager.apply_window(), [])

    def test_multiple_file_modifications_deduplicated(self):
        """Test that duplicate file paths are deduplicated in summary."""
        manager = HierarchicalConversationManager(
            window_size=5,
            full_context_window=1,
            summarization_threshold=0.5,
        )

        # Add two write_file calls to the same file
        for i in range(2):
            manager.add_message(
                {
                    "role": "assistant",
                    "content": f"Writing file attempt {i}",
                    "tool_calls": [
                        {
                            "id": f"call_{i}",
                            "function": {
                                "name": "write_file",
                                "arguments": f'{{"file_path": "main.py", "content": "v{i}"}}',
                            },
                        }
                    ],
                }
            )
            manager.add_message(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{i}",
                    "content": "OK",
                }
            )

        # Fill to exceed threshold
        for i in range(8):
            manager.add_message({"role": "user", "content": f"msg-{i}"})
            manager.add_message({"role": "assistant", "content": f"reply-{i}"})

        result = manager.apply_window()
        summary_content = result[0]["content"]

        # main.py should appear exactly once in the files modified section
        self.assertEqual(summary_content.count("main.py"), 1)


if __name__ == "__main__":
    unittest.main()
