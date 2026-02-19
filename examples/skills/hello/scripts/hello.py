#!/usr/bin/env python
"""Simple script to greet a user by name."""

import sys


def greet(name: str) -> None:
    """Print a greeting with the given name."""
    print(f"Hello, {name}! Welcome I am Kader! How can I help you today? 👋")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python hello.py <name>")
        print("Example: python hello.py World")
        sys.exit(1)

    name = sys.argv[1]
    greet(name)


if __name__ == "__main__":
    main()
