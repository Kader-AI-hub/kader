"""
Tool Output Compression for Kader Agents.

Provides intelligent compression of tool outputs to reduce token usage
while preserving essential information. Inspired by SWE-agent's ACI design.

Features:
- Tool-specific compression rules
- Multiple compression strategies (truncate middle, truncate end, smart truncate)
- Configurable limits per tool type
- SWE-agent style feedback ("Command completed successfully")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompressionStrategy(Enum):
    """Compression strategies for different output types."""

    TRUNCATE_MIDDLE = "truncate_middle"
    TRUNCATE_END = "truncate_end"
    SUMMARIZE = "summarize"
    SMART_TRUNCATE = "smart_truncate"


@dataclass
class CompressionConfig:
    """Configuration for output compression.

    Attributes:
        max_lines: Maximum number of lines to keep
        max_chars: Maximum number of characters to keep
        preserve_patterns: Regex patterns for lines that should always be preserved
        strategy: The compression strategy to use
        enabled: Whether compression is enabled for this tool
    """

    max_lines: int = 100
    max_chars: int = 5000
    preserve_patterns: list[str] = field(default_factory=list)
    strategy: CompressionStrategy = CompressionStrategy.TRUNCATE_MIDDLE
    enabled: bool = True


class ToolOutputCompressor:
    """Compresses tool outputs intelligently based on tool type.

    Inspired by SWE-agent's ACI design principles:
    - Limit file reads to prevent context overflow
    - Show structured feedback instead of raw output
    - Preserve critical information (errors, file paths, etc.)

    Example:
        compressor = ToolOutputCompressor()
        compressed = compressor.compress("grep", long_grep_output)
        # Returns truncated output with indication of more results
    """

    DEFAULT_CONFIG = CompressionConfig()

    COMPRESSION_RULES: dict[str, CompressionConfig] = {
        "read_file": CompressionConfig(
            max_lines=100,
            max_chars=8000,
            strategy=CompressionStrategy.TRUNCATE_MIDDLE,
            preserve_patterns=[r"def\s+\w+", r"class\s+\w+", r"^\s*#", r"^\s*import"],
        ),
        "grep": CompressionConfig(
            max_lines=50,
            max_chars=3000,
            strategy=CompressionStrategy.TRUNCATE_END,
            preserve_patterns=[],
        ),
        "search_file": CompressionConfig(
            max_lines=50,
            max_chars=3000,
            strategy=CompressionStrategy.TRUNCATE_END,
            preserve_patterns=[],
        ),
        "search_dir": CompressionConfig(
            max_lines=50,
            max_chars=3000,
            strategy=CompressionStrategy.TRUNCATE_END,
            preserve_patterns=[],
        ),
        "ls": CompressionConfig(
            max_lines=200,
            max_chars=5000,
            strategy=CompressionStrategy.TRUNCATE_END,
            preserve_patterns=[],
        ),
        "command": CompressionConfig(
            max_lines=200,
            max_chars=10000,
            strategy=CompressionStrategy.TRUNCATE_MIDDLE,
            preserve_patterns=[
                r"error",
                r"fail",
                r"exception",
                r"success",
                r"Error:",
                r"Failed:",
            ],
        ),
        "execute_command": CompressionConfig(
            max_lines=200,
            max_chars=10000,
            strategy=CompressionStrategy.TRUNCATE_MIDDLE,
            preserve_patterns=[
                r"error",
                r"fail",
                r"exception",
                r"success",
                r"Error:",
                r"Failed:",
            ],
        ),
        "Bash": CompressionConfig(
            max_lines=200,
            max_chars=10000,
            strategy=CompressionStrategy.TRUNCATE_MIDDLE,
            preserve_patterns=[
                r"error",
                r"fail",
                r"exception",
                r"success",
                r"Error:",
                r"Failed:",
            ],
        ),
        "Glob": CompressionConfig(
            max_lines=100,
            max_chars=5000,
            strategy=CompressionStrategy.TRUNCATE_END,
            preserve_patterns=[],
        ),
        "file_read": CompressionConfig(
            max_lines=100,
            max_chars=8000,
            strategy=CompressionStrategy.TRUNCATE_MIDDLE,
            preserve_patterns=[r"def\s+\w+", r"class\s+\w+", r"^\s*#", r"^\s*import"],
        ),
    }

    def __init__(self, custom_rules: dict[str, CompressionConfig] | None = None):
        """Initialize the compressor with optional custom rules.

        Args:
            custom_rules: Optional custom compression rules to override defaults
        """
        self._rules = {**self.COMPRESSION_RULES}
        if custom_rules:
            self._rules.update(custom_rules)

    def compress(
        self, tool_name: str, output: str, context: dict[str, Any] | None = None
    ) -> str:
        """Compress tool output based on tool type.

        Args:
            tool_name: Name of the tool that produced the output
            output: Raw output string
            context: Additional context (e.g., line numbers for file reads)

        Returns:
            Compressed output string
        """
        config = self._rules.get(tool_name, self.DEFAULT_CONFIG)

        if not config.enabled:
            return output

        if not output:
            return "[Command completed successfully with no output]"

        if not output.strip():
            return "[Command completed successfully with no output]"

        lines = output.split("\n")
        total_lines = len(lines)

        if total_lines <= config.max_lines and len(output) <= config.max_chars:
            return output

        if config.strategy == CompressionStrategy.TRUNCATE_MIDDLE:
            return self._truncate_middle(lines, config, total_lines)
        elif config.strategy == CompressionStrategy.TRUNCATE_END:
            return self._truncate_end(lines, config)
        elif config.strategy == CompressionStrategy.SMART_TRUNCATE:
            return self._smart_truncate(lines, config, context)

        return output

    def _truncate_middle(
        self, lines: list[str], config: CompressionConfig, total: int
    ) -> str:
        """Keep first and last portions, truncate middle."""
        half_limit = config.max_lines // 2
        first_part = lines[:half_limit]
        last_part = lines[-half_limit:]
        skipped = total - config.max_lines

        return (
            f"[Output: {total} lines total]\n"
            + "\n".join(first_part)
            + f"\n... [{skipped} lines truncated] ...\n"
            + "\n".join(last_part)
        )

    def _truncate_end(self, lines: list[str], config: CompressionConfig) -> str:
        """Keep beginning only, indicate truncation."""
        kept = lines[: config.max_lines]
        skipped = len(lines) - config.max_lines

        result = "\n".join(kept)
        if skipped > 0:
            result += f"\n... and {skipped} more lines"

        return result

    def _smart_truncate(
        self,
        lines: list[str],
        config: CompressionConfig,
        context: dict[str, Any] | None,
    ) -> str:
        """Context-aware truncation preserving important sections.

        This method preserves lines matching preserve_patterns while
        truncating others to stay within limits.
        """
        import re

        if not config.preserve_patterns:
            return self._truncate_end(lines, config)

        preserved_lines: list[tuple[int, str]] = []
        other_lines: list[tuple[int, str]] = []

        for idx, line in enumerate(lines):
            is_preserved = False
            for pattern in config.preserve_patterns:
                if re.search(pattern, line):
                    is_preserved = True
                    break

            if is_preserved:
                preserved_lines.append((idx, line))
            else:
                other_lines.append((idx, line))

        available_slots = config.max_lines - len(preserved_lines)
        if available_slots <= 0:
            result = "\n".join(line for _, line in preserved_lines[: config.max_lines])
            result += f"\n[... {len(lines) - config.max_lines} lines truncated ...]"
            return result

        kept_other = other_lines[:available_slots]
        all_kept = sorted(preserved_lines + kept_other, key=lambda x: x[0])

        result = "\n".join(line for _, line in all_kept)

        total_skipped = len(lines) - len(all_kept)
        if total_skipped > 0:
            result += f"\n[... {total_skipped} lines truncated ...]"

        return result

    def get_config(self, tool_name: str) -> CompressionConfig:
        """Get compression config for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            CompressionConfig for the tool, or default config
        """
        return self._rules.get(tool_name, self.DEFAULT_CONFIG)

    def set_config(self, tool_name: str, config: CompressionConfig) -> None:
        """Set compression config for a tool.

        Args:
            tool_name: Name of the tool
            config: Compression configuration
        """
        self._rules[tool_name] = config

    def disable_compression(self, tool_name: str) -> None:
        """Disable compression for a specific tool.

        Args:
            tool_name: Name of the tool
        """
        self._rules[tool_name] = CompressionConfig(enabled=False)

    def enable_compression(self, tool_name: str) -> None:
        """Enable compression for a specific tool.

        Args:
            tool_name: Name of the tool
        """
        if tool_name in self.COMPRESSION_RULES:
            self._rules[tool_name] = self.COMPRESSION_RULES[tool_name]
        else:
            self._rules[tool_name] = self.DEFAULT_CONFIG


def compress_tool_output(
    tool_name: str, output: str, context: dict[str, Any] | None = None
) -> str:
    """Convenience function to compress tool output.

    Args:
        tool_name: Name of the tool that produced the output
        output: Raw output string
        context: Additional context

    Returns:
        Compressed output string
    """
    compressor = ToolOutputCompressor()
    return compressor.compress(tool_name, output, context)
