"""
Gitignore utility for filtering files based on .gitignore patterns.

This module provides functionality to read .gitignore files and filter
file/directory listings based on the patterns defined in them.
"""

import re
from pathlib import Path
from typing import Any


def _parse_gitignore_patterns(gitignore_path: Path) -> list[tuple[str, bool]]:
    """
    Parse a .gitignore file and return a list of (pattern, is_negation) tuples.

    Args:
        gitignore_path: Path to the .gitignore file

    Returns:
        List of tuples containing (pattern, is_negation)
    """
    patterns: list[tuple[str, bool]] = []

    if not gitignore_path.exists():
        return patterns

    with open(gitignore_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            is_negation = line.startswith("!")
            if is_negation:
                line = line[1:]

            if not line:
                continue

            patterns.append((line, is_negation))

    return patterns


def _match_gitignore_pattern(name: str, pattern: str, is_dir: bool = False) -> bool:
    """
    Check if a name matches a gitignore pattern.

    Args:
        name: The file or directory name to check
        pattern: The gitignore pattern
        is_dir: Whether the name is a directory

    Returns:
        True if the name matches the pattern
    """
    if not pattern:
        return False

    if pattern == "**":
        return True

    if pattern.startswith("**/"):
        pattern = pattern[3:]

    if pattern.endswith("/"):
        if is_dir:
            pattern = pattern[:-1]
        else:
            return False

    if "/" not in pattern and "*" not in pattern and "?" not in pattern:
        return name == pattern

    if "**" in pattern:
        parts = pattern.split("**")
        if len(parts) == 2:
            prefix, suffix = parts
            if prefix == "":
                return name.endswith(suffix) if suffix else True
            if suffix == "":
                return name.startswith(prefix)
            return name.startswith(prefix) and name.endswith(suffix)

    regex_pattern = pattern.replace(".", r"\.")
    regex_pattern = regex_pattern.replace("**", ".*")
    regex_pattern = regex_pattern.replace("*", "[^/]*")
    regex_pattern = regex_pattern.replace("?", ".")
    regex_pattern = f"^{regex_pattern}$"

    try:
        return bool(re.match(regex_pattern, name))
    except re.error:
        return False


def get_gitignore_filter(root_path: Path | str | None = None) -> set[str]:
    """
    Get all patterns from .gitignore files in the root_path and its parents.

    Args:
        root_path: The root directory to scan for .gitignore files.
                   Defaults to current working directory.

    Returns:
        Set of patterns to exclude based on .gitignore files
    """
    if root_path is None:
        root_path = Path.cwd()
    else:
        root_path = Path(root_path).resolve()

    all_patterns: list[tuple[str, bool]] = []

    current = root_path
    while True:
        gitignore_path = current / ".gitignore"
        if gitignore_path.exists():
            patterns = _parse_gitignore_patterns(gitignore_path)
            all_patterns.extend(patterns)

        parent = current.parent
        if parent == current:
            break
        current = parent

    exclusion_patterns: set[str] = set()

    for pattern, is_negation in all_patterns:
        if not is_negation:
            if pattern.endswith("/"):
                pattern = pattern[:-1]
            if pattern:
                exclusion_patterns.add(pattern)

    return exclusion_patterns


def filter_by_gitignore(
    items: list[dict[str, Any]],
    root_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """
    Filter a list of file/directory items based on .gitignore patterns.

    Args:
        items: List of file info dictionaries with 'path' and 'is_dir' keys
        root_path: Path to the root directory. Defaults to CWD.

    Returns:
        Filtered list with items matching .gitignore patterns removed
    """
    exclusion_patterns = get_gitignore_filter(root_path)

    filtered: list[dict[str, Any]] = []

    for item in items:
        full_path = item.get("path", "")
        is_dir = item.get("is_dir", False)

        name = Path(full_path).name

        if name in exclusion_patterns:
            continue

        matched = False
        for pattern in exclusion_patterns:
            if _match_gitignore_pattern(name, pattern, is_dir):
                matched = True
                break

        if matched:
            continue

        filtered.append(item)

    return filtered


def is_ignored(name: str, is_dir: bool, root_path: Path | str | None = None) -> bool:
    """
    Check if a file or directory should be ignored based on .gitignore patterns.

    Args:
        name: The file or directory name to check
        is_dir: Whether the name is a directory
        root_path: Path to the root directory. Defaults to CWD.

    Returns:
        True if the file/directory should be ignored
    """
    exclusion_patterns = get_gitignore_filter(root_path)

    if name in exclusion_patterns:
        return True

    for pattern in exclusion_patterns:
        if _match_gitignore_pattern(name, pattern, is_dir):
            return True

    return False
