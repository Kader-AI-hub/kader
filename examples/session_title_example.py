"""
Session Title Generator Example.

Demonstrates how to generate brief, descriptive titles for conversation
sessions using the session title generator utilities.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kader.providers import OllamaProvider
from kader.utils import generate_session_title


def main():
    print("=== Session Title Generator Demo ===\n")

    provider = OllamaProvider(model="minimax-m2.5:cloud")

    conversation_single = "How do I fix the authentication 401 error in my FastAPI app?"

    print("Example 1: Single string conversation")
    print(f"Conversation: {conversation_single}\n")
    title = generate_session_title(provider, conversation_single)
    print(f"Generated Title: {title}\n")


if __name__ == "__main__":
    main()
