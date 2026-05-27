"""
Subagent Configuration and Loading.

Provides SubagentConfig dataclass and SubagentLoader for discovering
and loading subagent definitions from YAML files in user-level
(~/.kader/subagents/) and project-level (./.kader/subagents/) directories.
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SubagentConfig:
    """
    Configuration for a subagent loaded from YAML.

    Attributes:
        name: Unique identifier for the subagent (kebab-case).
        objective: Description to help the planner decide when to use this subagent.
                   Also used as the AgentTool description.
        system_prompt: The system prompt for the subagent.
        tools: List of tool name strings to resolve at AgentTool creation time.
               Defaults to empty list (all standard tools).
    """

    name: str
    objective: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)


class SubagentLoader:
    """
    Loads subagent definitions from YAML files in subagent directories.

    Follows the same discovery pattern as SkillLoader:
    - User-level (~/.kader/subagents/) takes priority over project-level on name collision
    - Supports both {name}.yaml flat files and {name}/template.yaml subdirectories
    - Project-level subagents are always enabled
    - User-level subagents are gated by settings.json (filtered externally)
    """

    def __init__(
        self,
        subagents_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the subagent loader.

        Args:
            subagents_dirs: List of directories to load subagents from.
                           If None, defaults to ~/.kader/subagents and ./.kader/subagents.
            priority_dir: Optional directory to check first (higher priority).
        """
        self._priority_dir = priority_dir

        if subagents_dirs is None:
            home_subagents = Path.home() / ".kader" / "subagents"
            cwd_subagents = Path.cwd() / ".kader" / "subagents"
            subagents_dirs = [home_subagents, cwd_subagents]

        if priority_dir is not None:
            subagents_dirs = [priority_dir] + subagents_dirs

        self.subagents_dirs = subagents_dirs

    def _find_subagent_file(self, name: str) -> tuple[Path, Path] | None:
        """
        Find the subagent YAML file in the subagent directories.

        Supports two structures:
        1. {dir}/{name}.yaml (flat file)
        2. {dir}/{name}/template.yaml (subdirectory)

        Args:
            name: Name of the subagent to find.

        Returns:
            Tuple of (parent_dir, yaml_file) if found, None otherwise.
        """
        for subagents_dir in self.subagents_dirs:
            if not subagents_dir.exists():
                continue

            flat_file = subagents_dir / f"{name}.yaml"
            if flat_file.exists():
                return (subagents_dir, flat_file)

            sub_dir = subagents_dir / name
            template_file = sub_dir / "template.yaml"
            if template_file.exists():
                return (subagents_dir, template_file)

        return None

    def load_subagent(self, name: str) -> SubagentConfig | None:
        """
        Load a subagent by name.

        Args:
            name: Name of the subagent to load.

        Returns:
            SubagentConfig if found, None otherwise.
        """
        result = self._find_subagent_file(name)
        if result is None:
            return None

        _, yaml_file = result
        try:
            with open(yaml_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError:
            return None

        if not data or not isinstance(data, dict):
            return None

        return SubagentConfig(
            name=data.get("name", name),
            objective=data.get("objective", ""),
            system_prompt=data.get("system_prompt", ""),
            tools=data.get("tools", []),
        )

    def list_subagents(self) -> list[SubagentConfig]:
        """
        List all available subagents from all directories.

        Subagents from ~/.kader/subagents take priority over subagents
        with the same name in other directories (matches SkillLoader behavior).

        Returns:
            List of all available SubagentConfig instances.
        """
        seen_names: set[str] = set()
        subagents: list[SubagentConfig] = []

        for subagents_dir in self.subagents_dirs:
            if not subagents_dir.exists():
                continue

            found_names: set[str] = set()

            for entry in sorted(subagents_dir.iterdir()):
                name: str | None = None

                if entry.is_file() and entry.suffix == ".yaml":
                    name = entry.stem
                elif entry.is_dir() and (entry / "template.yaml").exists():
                    name = entry.name

                if name is None or name in seen_names:
                    continue

                found_names.add(name)
                config = self.load_subagent(name)
                if config:
                    subagents.append(config)

            seen_names |= found_names

        return subagents

    def get_description(self) -> str:
        """
        Get a formatted description of all available subagents.

        Returns:
            Formatted string listing all subagents with their objectives.
        """
        subagents = self.list_subagents()

        if not subagents:
            return "No subagents available."

        listing = "\n".join(f"  - {s.name}: {s.objective}" for s in subagents)

        return listing


def load_subagents_from_settings(
    loader: SubagentLoader,
    enabled_subagents: list[dict[str, str]] | None = None,
) -> list[SubagentConfig]:
    """
    Filter subagents based on settings.json configuration.

    Args:
        loader: SubagentLoader instance with discovered directories.
        enabled_subagents: List of {"name": "...", "enabled": "true/false"} dicts
                          from settings.json. If None or empty, no filtering is
                          applied (all discovered subagents are returned).

    Returns:
        Filtered list of SubagentConfig instances.
    """
    all_configs = loader.list_subagents()

    if not enabled_subagents:
        return all_configs

    enabled_map = {
        item["name"]: item.get("enabled", "true") == "true"
        for item in enabled_subagents
    }

    filtered: list[SubagentConfig] = []
    for config in all_configs:
        # Determine which directory this subagent came from
        home_subagents = Path.home() / ".kader" / "subagents"
        result = loader._find_subagent_file(config.name)
        is_user_level = result is not None and result[0] == home_subagents

        if is_user_level:
            # User-level subagents are gated by settings
            if config.name not in enabled_map:
                # Not in settings at all - skip
                continue
            if not enabled_map[config.name]:
                # Disabled in settings - skip
                continue

        # Project-level subagents and enabled user-level subagents pass through
        filtered.append(config)

    return filtered
