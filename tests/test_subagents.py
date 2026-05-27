"""
Tests for the subagents feature.

Covers SubagentLoader, SubagentConfig, AgentTool.from_subagent_config,
PlannerExecutorWorkflow integration, settings migration, and prompt rendering.
"""

from pathlib import Path
from unittest.mock import patch

import yaml

from kader.workflows.subagents import (
    SubagentConfig,
    SubagentLoader,
    load_subagents_from_settings,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_subagent_yaml(
    base_dir: Path,
    name: str,
    objective: str,
    system_prompt: str,
    tools: list[str] | None = None,
) -> Path:
    """Create a flat subagent YAML file at ``{base_dir}/{name}.yaml``."""
    data: dict[str, object] = {
        "name": name,
        "objective": objective,
        "system_prompt": system_prompt,
    }
    if tools is not None:
        data["tools"] = tools
    yaml_path = base_dir / f"{name}.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    return yaml_path


def create_subagent_dir_yaml(
    base_dir: Path,
    name: str,
    objective: str,
    system_prompt: str,
    tools: list[str] | None = None,
) -> Path:
    """Create a subdirectory subagent: ``{base_dir}/{name}/template.yaml``."""
    sub_dir = base_dir / name
    sub_dir.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {
        "name": name,
        "objective": objective,
        "system_prompt": system_prompt,
    }
    if tools is not None:
        data["tools"] = tools
    yaml_path = sub_dir / "template.yaml"
    yaml_path.write_text(yaml.dump(data), encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# SubagentConfig
# ---------------------------------------------------------------------------


class TestSubagentConfig:
    def test_defaults(self):
        config = SubagentConfig(name="test", objective="", system_prompt="")
        assert config.name == "test"
        assert config.objective == ""
        assert config.system_prompt == ""
        assert config.tools == []

    def test_with_tools(self):
        config = SubagentConfig(
            name="reviewer",
            objective="Review code",
            system_prompt="You are a code reviewer.",
            tools=["read_file", "grep"],
        )
        assert config.tools == ["read_file", "grep"]


# ---------------------------------------------------------------------------
# SubagentLoader
# ---------------------------------------------------------------------------


class TestSubagentLoader:
    def test_load_flat_yaml(self, tmp_path):
        create_subagent_yaml(
            tmp_path,
            name="reviewer",
            objective="Review code",
            system_prompt="You are a code reviewer.",
            tools=["read_file"],
        )

        loader = SubagentLoader([tmp_path])
        config = loader.load_subagent("reviewer")

        assert config is not None
        assert config.name == "reviewer"
        assert config.objective == "Review code"
        assert config.system_prompt == "You are a code reviewer."
        assert config.tools == ["read_file"]

    def test_load_subdirectory_yaml(self, tmp_path):
        create_subagent_dir_yaml(
            tmp_path,
            name="builder",
            objective="Build projects",
            system_prompt="You are a project builder.",
            tools=["write_file"],
        )

        loader = SubagentLoader([tmp_path])
        config = loader.load_subagent("builder")

        assert config is not None
        assert config.name == "builder"
        assert config.objective == "Build projects"
        assert config.system_prompt == "You are a project builder."
        assert config.tools == ["write_file"]

    def test_load_nonexistent(self, tmp_path):
        loader = SubagentLoader([tmp_path])
        assert loader.load_subagent("nonexistent") is None

    def test_list_multiple_subagents(self, tmp_path):
        create_subagent_yaml(
            tmp_path, "a", objective="Agent A", system_prompt="Prompt A"
        )
        create_subagent_dir_yaml(
            tmp_path, "b", objective="Agent B", system_prompt="Prompt B"
        )

        loader = SubagentLoader([tmp_path])
        configs = loader.list_subagents()
        assert len(configs) == 2
        names = {c.name for c in configs}
        assert names == {"a", "b"}

    def test_list_empty(self, tmp_path):
        loader = SubagentLoader([tmp_path])
        assert loader.list_subagents() == []

    def test_name_from_yaml_overrides_filename(self, tmp_path):
        yaml_path = tmp_path / "my_agent.yaml"
        yaml_path.write_text(
            yaml.dump(
                {
                    "name": "custom-name",
                    "objective": "Override test",
                    "system_prompt": "Testing name from yaml",
                }
            ),
            encoding="utf-8",
        )

        loader = SubagentLoader([tmp_path])
        config = loader.load_subagent("my_agent")
        assert config is not None
        assert config.name == "custom-name"

    def test_priority_user_level_overrides_project(self, tmp_path):
        home_dir = tmp_path / "home"
        home_dir.mkdir()
        cwd_dir = tmp_path / "cwd"
        cwd_dir.mkdir()

        create_subagent_yaml(
            home_dir, "shared", objective="Home version", system_prompt="From home"
        )
        create_subagent_yaml(
            cwd_dir, "shared", objective="CWD version", system_prompt="From cwd"
        )

        loader = SubagentLoader([home_dir, cwd_dir])
        config = loader.load_subagent("shared")

        assert config is not None
        assert config.objective == "Home version"

    def test_get_description(self, tmp_path):
        create_subagent_yaml(tmp_path, "a", objective="First agent", system_prompt="")
        create_subagent_yaml(tmp_path, "b", objective="Second agent", system_prompt="")

        loader = SubagentLoader([tmp_path])
        desc = loader.get_description()

        assert "a: First agent" in desc
        assert "b: Second agent" in desc

    def test_get_description_empty(self, tmp_path):
        loader = SubagentLoader([tmp_path])
        assert loader.get_description() == "No subagents available."

    def test_no_provider_model_in_yaml(self, tmp_path):
        """Ensure that provider/model are NOT loaded from YAML (not present in SubagentConfig)."""
        create_subagent_yaml(
            tmp_path,
            name="no-prov",
            objective="No provider",
            system_prompt="No provider field",
        )

        loader = SubagentLoader([tmp_path])
        config = loader.load_subagent("no-prov")
        assert config is not None
        assert not hasattr(config, "provider")
        assert not hasattr(config, "model")

    def test_corrupt_yaml_returns_none(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("{{{ invalid yaml !!!", encoding="utf-8")

        loader = SubagentLoader([tmp_path])
        config = loader.load_subagent("bad")
        assert config is None

        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")

        config2 = loader.load_subagent("empty")
        assert config2 is None


# ---------------------------------------------------------------------------
# load_subagents_from_settings
# ---------------------------------------------------------------------------


class TestLoadSubagentsFromSettings:
    def test_no_settings_returns_all(self, tmp_path):
        create_subagent_yaml(tmp_path, "a", objective="A", system_prompt="")
        create_subagent_yaml(tmp_path, "b", objective="B", system_prompt="")

        loader = SubagentLoader([tmp_path])
        result = load_subagents_from_settings(loader, enabled_subagents=None)
        assert len(result) == 2

    def test_empty_settings_returns_all(self, tmp_path):
        create_subagent_yaml(tmp_path, "a", objective="A", system_prompt="")

        loader = SubagentLoader([tmp_path])
        result = load_subagents_from_settings(loader, enabled_subagents=[])
        assert len(result) == 1

    @patch("kader.workflows.subagents.Path.home")
    def test_user_level_filtered_by_enabled(self, mock_home, tmp_path):
        mock_home.return_value = tmp_path

        home_subagents = tmp_path / ".kader" / "subagents"
        home_subagents.mkdir(parents=True)
        cwd_subagents = tmp_path / "project" / ".kader" / "subagents"
        cwd_subagents.mkdir(parents=True)

        create_subagent_yaml(
            home_subagents,
            "enabled-one",
            objective="Enabled from home",
            system_prompt="",
        )
        create_subagent_yaml(
            home_subagents,
            "disabled-one",
            objective="Disabled from home",
            system_prompt="",
        )

        loader = SubagentLoader([home_subagents, cwd_subagents])
        result = load_subagents_from_settings(
            loader,
            enabled_subagents=[
                {"name": "enabled-one", "enabled": "true"},
                {"name": "disabled-one", "enabled": "false"},
            ],
        )

        names = {c.name for c in result}
        assert "enabled-one" in names
        assert "disabled-one" not in names

    @patch("kader.workflows.subagents.Path.home")
    def test_project_level_always_included(self, mock_home, tmp_path):
        mock_home.return_value = tmp_path

        home_subagents = tmp_path / ".kader" / "subagents"
        home_subagents.mkdir(parents=True)
        cwd_subagents = tmp_path / "project" / ".kader" / "subagents"
        cwd_subagents.mkdir(parents=True)

        create_subagent_yaml(
            home_subagents, "home-agent", objective="Home", system_prompt=""
        )
        create_subagent_yaml(
            cwd_subagents, "project-agent", objective="Project", system_prompt=""
        )

        loader = SubagentLoader([home_subagents, cwd_subagents])
        result = load_subagents_from_settings(
            loader,
            enabled_subagents=[
                {"name": "home-agent", "enabled": "false"},
            ],
        )

        names = {c.name for c in result}
        # home-agent is disabled
        assert "home-agent" not in names
        # project-agent is always included
        assert "project-agent" in names

    @patch("kader.workflows.subagents.Path.home")
    def test_user_agent_not_in_settings_is_skipped(self, mock_home, tmp_path):
        mock_home.return_value = tmp_path

        home_subagents = tmp_path / ".kader" / "subagents"
        home_subagents.mkdir(parents=True)
        cwd_subagents = tmp_path / "project" / ".kader" / "subagents"
        cwd_subagents.mkdir(parents=True)

        create_subagent_yaml(
            home_subagents, "secret-agent", objective="Secret", system_prompt=""
        )

        loader = SubagentLoader([home_subagents, cwd_subagents])
        result = load_subagents_from_settings(
            loader,
            enabled_subagents=[
                {"name": "other-agent", "enabled": "true"},
            ],
        )

        names = {c.name for c in result}
        assert "secret-agent" not in names


# ---------------------------------------------------------------------------
# AgentTool.from_subagent_config
# ---------------------------------------------------------------------------


class TestAgentToolFromSubagentConfig:
    def test_creates_agent_tool(self):
        from kader.tools.agent import AgentTool

        tool = AgentTool.from_subagent_config(
            name="test-tool",
            objective="Test objective",
            system_prompt="You are a test agent.",
            tool_names=["read_file"],
        )

        assert tool.name == "test-tool"
        assert tool.description == "Test objective"
        assert len(tool._custom_tools) == 1
        assert tool._custom_tools[0].name == "read_file"

    def test_skips_missing_tools(self):
        from kader.tools.agent import AgentTool

        tool = AgentTool.from_subagent_config(
            name="test-tool",
            objective="Test",
            system_prompt="",
            tool_names=["nonexistent_tool", "read_file"],
        )

        assert len(tool._custom_tools) == 1
        assert tool._custom_tools[0].name == "read_file"

    def test_empty_tool_names(self):
        from kader.tools.agent import AgentTool

        tool = AgentTool.from_subagent_config(
            name="test-tool",
            objective="Test",
            system_prompt="",
            tool_names=[],
        )

        assert tool._custom_tools == []


# ---------------------------------------------------------------------------
# PlannerExecutorWorkflow integration
# ---------------------------------------------------------------------------


class TestPlannerExecutorWorkflowSubagents:
    def test_no_subagents_still_has_executor(self):
        """Without any subagent dirs, the planner still registers 'executor'."""
        from kader.workflows.planner_executor import PlannerExecutorWorkflow

        workflow = PlannerExecutorWorkflow(
            name="test_no_subagents",
            interrupt_before_tool=False,
            use_persistence=False,
        )
        tool_names = workflow.planner.tools_map.keys()
        assert "executor" in tool_names

    def test_subagent_configs_registered(self, tmp_path):
        from kader.workflows.planner_executor import PlannerExecutorWorkflow
        from kader.workflows.subagents import SubagentConfig

        configs = [
            SubagentConfig(
                name="reviewer",
                objective="Review code",
                system_prompt="You review code.",
                tools=["read_file"],
            ),
        ]

        with patch(
            "kader.workflows.planner_executor.load_subagents_from_settings",
            return_value=configs,
        ):
            workflow = PlannerExecutorWorkflow(
                name="test_subagent_reg",
                interrupt_before_tool=False,
                use_persistence=False,
            )
            tool_names = workflow.planner.tools_map.keys()
            assert "executor" in tool_names
            assert "reviewer" in tool_names

    def test_subagent_infos_passed_to_prompt(self, tmp_path):
        from kader.prompts import KaderPlannerPrompt

        prompt = KaderPlannerPrompt(
            tools=[],
            subagents=[
                {"name": "reviewer", "objective": "Review code"},
                {"name": "builder", "objective": "Build projects"},
            ],
        )
        rendered = prompt.resolve_prompt()
        assert "reviewer" in rendered
        assert "Review code" in rendered
        assert "builder" in rendered
        assert "Build projects" in rendered

    def test_prompt_without_subagents(self):
        from kader.prompts import KaderPlannerPrompt

        prompt = KaderPlannerPrompt(
            tools=[],
            subagents=[],
        )
        rendered = prompt.resolve_prompt()
        assert "Available Sub-Agents" not in rendered

    def test_prompt_with_none_subagents(self):
        from kader.prompts import KaderPlannerPrompt

        prompt = KaderPlannerPrompt(
            tools=[],
            subagents=None,
        )
        rendered = prompt.resolve_prompt()
        assert "Available Sub-Agents" not in rendered


# ---------------------------------------------------------------------------
# KaderSettings migration
# ---------------------------------------------------------------------------


class TestKaderSettingsSubagentsMigration:
    def test_migrate_discovers_subagents(self, tmp_path):
        from cli.settings.settings import _migrate_user_subagents

        home = tmp_path / "home"
        home.mkdir()

        subagents_dir = home / ".kader" / "subagents"
        subagents_dir.mkdir(parents=True)

        create_subagent_yaml(
            subagents_dir,
            name="reviewer",
            objective="Review code",
            system_prompt="You review code.",
        )

        with patch("pathlib.Path.home", return_value=home):
            data = {"main-agent-provider": "ollama"}
            result = _migrate_user_subagents(data)

            assert "subagents" in result
            assert len(result["subagents"]) == 1
            assert result["subagents"][0]["name"] == "reviewer"
            assert result["subagents"][0]["enabled"] == "true"
            assert result["_subagents_migrated"] is True

    def test_migrate_discovers_subdirectory_subagents(self, tmp_path):
        from cli.settings.settings import _migrate_user_subagents

        home = tmp_path / "home"
        home.mkdir()

        subagents_dir = home / ".kader" / "subagents"
        subagents_dir.mkdir(parents=True)

        create_subagent_dir_yaml(
            subagents_dir,
            name="builder",
            objective="Build projects",
            system_prompt="You build.",
        )

        with patch("pathlib.Path.home", return_value=home):
            data = {}
            result = _migrate_user_subagents(data)

            assert len(result["subagents"]) == 1
            assert result["subagents"][0]["name"] == "builder"
            assert result["subagents"][0]["enabled"] == "true"

    def test_migrate_creates_directory_if_missing(self, tmp_path):
        from cli.settings.settings import _migrate_user_subagents

        home = tmp_path / "home"
        home.mkdir()

        with patch("pathlib.Path.home", return_value=home):
            data = {}
            result = _migrate_user_subagents(data)

            assert (home / ".kader" / "subagents").exists()
            # No subagents discovered since dir was just created
            assert "subagents" not in result or result.get("subagents", []) == []

    def test_migrate_skips_existing_names(self, tmp_path):
        from cli.settings.settings import _migrate_user_subagents

        home = tmp_path / "home"
        home.mkdir()

        subagents_dir = home / ".kader" / "subagents"
        subagents_dir.mkdir(parents=True)

        create_subagent_yaml(
            subagents_dir,
            name="existing",
            objective="Already present",
            system_prompt="",
        )

        with patch("pathlib.Path.home", return_value=home):
            data = {
                "subagents": [
                    {"name": "existing", "enabled": "false"},
                ],
            }
            result = _migrate_user_subagents(data)

            # existing should NOT be overwritten; it stays disabled
            existing = next(
                (s for s in result["subagents"] if s["name"] == "existing"),
                None,
            )
            assert existing is not None
            assert existing["enabled"] == "false"

    def test_serialize_deserialize_subagents(self):
        from cli.settings.settings import KaderSettings

        settings = KaderSettings(
            subagents=[
                {"name": "reviewer", "enabled": "true"},
                {"name": "builder", "enabled": "false"},
            ],
        )

        d = settings.to_dict()
        restored = KaderSettings.from_dict(d)

        assert len(restored.subagents) == 2
        names = {s["name"] for s in restored.subagents}
        assert names == {"reviewer", "builder"}
