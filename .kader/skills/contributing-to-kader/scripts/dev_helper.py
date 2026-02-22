#!/usr/bin/env python3
"""
Kader Development Helper Script

This script provides common development commands for contributors.
Run with: uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py <command>
"""

import subprocess
import sys
from pathlib import Path

# Project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and print its output."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def sync_dependencies():
    """Install/sync project dependencies."""
    return run_command(["uv", "sync"], "Installing Dependencies")


def run_tests():
    """Run all tests."""
    return run_command(["uv", "run", "pytest", "-v"], "Running Tests")


def run_specific_test(test_path: str):
    """Run a specific test file or function."""
    return run_command(
        ["uv", "run", "pytest", test_path, "-v"], f"Running Test: {test_path}"
    )


def lint_code():
    """Run Ruff linter."""
    return run_command(["uv", "run", "ruff", "check", "."], "Linting Code")


def lint_fix():
    """Run Ruff with auto-fix."""
    return run_command(
        ["uv", "run", "ruff", "check", ".", "--fix"], "Fixing Lint Issues"
    )


def format_code():
    """Format code with Ruff."""
    return run_command(["uv", "run", "ruff", "format", "."], "Formatting Code")


def format_check():
    """Check code formatting without modifying."""
    return run_command(
        ["uv", "run", "ruff", "format", "--check", "."], "Checking Code Format"
    )


def run_cli():
    """Run the Kader CLI."""
    return run_command(["uv", "run", "python", "-m", "cli"], "Running CLI")


def dev_mode():
    """Run CLI in development mode with hot reload."""
    return run_command(
        ["uv", "run", "textual", "run", "--dev", "cli.app:KaderApp"],
        "Running CLI in Dev Mode",
    )


def all_checks():
    """Run all checks: lint, format check, and tests."""
    print("\n" + "=" * 60)
    print("RUNNING ALL CHECKS")
    print("=" * 60)

    commands = [
        (["uv", "run", "ruff", "check", ".", "--fix"], "Fixing Lint Issues"),
        (["uv", "run", "ruff", "format", "."], "Formatting Code"),
        (["uv", "run", "pytest", "-v"], "Running Tests"),
    ]

    for cmd, desc in commands:
        result = run_command(cmd, desc)
        if result != 0:
            print(f"\n[ERROR] {desc} failed with exit code {result}")
            return result

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED!")
    print("=" * 60)
    return 0


def show_help():
    """Show available commands."""
    help_text = """
Kader Development Helper

Usage: uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py <command>

Commands:
    sync           Install/sync project dependencies
    test           Run all tests
    test <path>    Run specific test file or function
    lint           Run Ruff linter
    lint-fix       Run Ruff with auto-fix
    format         Format code with Ruff
    format-check   Check code formatting
    cli            Run the Kader CLI
    dev            Run CLI in development mode
    all            Run all checks (lint-fix, format, test)
    help           Show this help message

Examples:
    uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py test
    uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py test tests/test_base_agent.py
    uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py test tests/test_base_agent.py::test_agent_mock_invocation
    uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py all
"""
    print(help_text)


def main():
    if len(sys.argv) < 2:
        show_help()
        return 0

    command = sys.argv[1].lower()

    # Map commands to functions
    commands = {
        "sync": sync_dependencies,
        "test": lambda: run_specific_test(sys.argv[2])
        if len(sys.argv) > 2
        else run_tests(),
        "lint": lint_code,
        "lint-fix": lint_fix,
        "format": format_code,
        "format-check": format_check,
        "cli": run_cli,
        "dev": dev_mode,
        "all": all_checks,
        "help": show_help,
    }

    if command not in commands:
        print(f"Unknown command: {command}")
        show_help()
        return 1

    return commands[command]()


if __name__ == "__main__":
    sys.exit(main())
