"""CLI tools integration for Kader CLI.

Provides functionality to load custom tools from user and project directories.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from cli.settings.settings import KaderSettings
from cli.tools.loader import ToolLoader
from kader.tools.base import BaseTool


def _parse_agent_target(agent_str: str | None) -> list[str]:
    """
    Parse agent target string to list of target agents.

    Args:
        agent_str: Agent target string ("planner", "executor", "both")

    Returns:
        List of target agents ["planner", "executor"] or ["both"]
    """
    if agent_str is None:
        return ["both"]

    agent_str = agent_str.lower()
    if agent_str == "both":
        return ["both"]
    elif agent_str in ("planner", "executor"):
        return [agent_str]
    else:
        logger.warning(f"Unknown agent target '{agent_str}', defaulting to 'both'")
        return ["both"]


def _load_project_tool_agent_config(tool_dir: Path) -> list[str]:
    """
    Load agent config from tool directory's agent.json file.

    Args:
        tool_dir: Path to the tool directory

    Returns:
        List of target agents, defaults to ["both"] if not found
    """
    agent_config_file = tool_dir / "agent.json"
    if not agent_config_file.exists():
        return ["both"]

    try:
        config = json.loads(agent_config_file.read_text(encoding="utf-8"))
        agent_str = config.get("agent")
        return _parse_agent_target(agent_str)
    except Exception as e:
        logger.warning(f"Failed to load agent.json for {tool_dir.name}: {e}")
        return ["both"]


def load_tools_from_settings(
    settings: KaderSettings,
) -> tuple[list[BaseTool], list[BaseTool]]:
    """
    Load custom tools based on settings configuration.

    Loading order (user-level takes priority over project-level):
    1. Project-level tools from ./.kader/custom/tools (always enabled)
    2. User-level tools from settings.json (controlled by 'enabled' field)

    Agent targeting:
    - Project-level: Uses optional agent.json in tool directory
    - User-level: Uses 'agent' field in settings config

    Args:
        settings: KaderSettings containing tool configuration

    Returns:
        Tuple of (planner_tools, executor_tools) - lists of instantiated tool objects
    """
    planner_tools: list[BaseTool] = []
    executor_tools: list[BaseTool] = []

    project_tools_dir = Path.cwd() / ".kader" / "custom" / "tools"
    user_tools_dir = Path.home() / ".kader" / "custom" / "tools"

    project_loader = ToolLoader(tools_dirs=[project_tools_dir])
    user_loader = ToolLoader(tools_dirs=[user_tools_dir], priority_dir=user_tools_dir)

    project_tool_classes = project_loader.list_tools()
    project_tool_names: set[str] = set()

    for tool_class in project_tool_classes:
        try:
            tool_dir = project_tools_dir / tool_class.__name__
            agent_targets = _load_project_tool_agent_config(tool_dir)

            instance = tool_class()
            if not isinstance(instance, BaseTool):
                logger.warning(
                    f"Project tool {tool_class.__name__} is not a BaseTool instance"
                )
                continue

            if "both" in agent_targets:
                planner_tools.append(instance)
                executor_tools.append(instance)
            elif "planner" in agent_targets:
                planner_tools.append(instance)
            elif "executor" in agent_targets:
                executor_tools.append(instance)

            project_tool_names.add(tool_class.__name__.lower())
            logger.info(
                f"Loaded project tool: {tool_class.__name__} (agents: {agent_targets})"
            )
        except Exception as e:
            logger.warning(
                f"Failed to instantiate project tool {tool_class.__name__}: {e}"
            )

    tool_configs = settings.tools or []
    for config in tool_configs:
        if not isinstance(config, dict):
            continue

        name = config.get("name")
        enabled_str = config.get("enabled", "true")
        enabled = (
            enabled_str.lower() == "true"
            if isinstance(enabled_str, str)
            else bool(enabled_str)
        )

        if not name:
            continue

        if not enabled:
            logger.debug(f"Skipping disabled tool: {name}")
            continue

        tool_class = user_loader.load_tool(name)
        if tool_class is None:
            logger.debug(f"User tool not found: {name}")
            continue

        try:
            instance = tool_class()
            if not isinstance(instance, BaseTool):
                logger.warning(f"User tool {name} is not a BaseTool instance")
                continue

            agent_str = config.get("agent")
            agent_targets = _parse_agent_target(agent_str)

            if "both" in agent_targets:
                planner_tools.append(instance)
                executor_tools.append(instance)
            elif "planner" in agent_targets:
                planner_tools.append(instance)
            elif "executor" in agent_targets:
                executor_tools.append(instance)

            logger.info(f"Loaded user tool: {name} (agents: {agent_targets})")
        except Exception as e:
            logger.warning(f"Failed to instantiate user tool {name}: {e}")

    return planner_tools, executor_tools


__all__ = ["load_tools_from_settings"]
