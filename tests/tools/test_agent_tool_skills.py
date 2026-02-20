"""Unit tests for AgentTool skills integration."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kader.tools.agent import AgentTool


def _create_skill(skills_dir: Path, name: str, description: str, content: str) -> None:
    """Helper: create a skill directory with SKILL.md."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    frontmatter = f"---\nname: {name}\ndescription: {description}\n---\n\n{content}\n"
    (skill_dir / "SKILL.md").write_text(frontmatter, encoding="utf-8")


class TestAgentToolSkillsInit:
    """Test AgentTool initialisation with skills parameters."""

    def test_init_stores_skills_dirs(self):
        """skills_dirs is stored on the instance."""
        dirs = [Path("/tmp/skills")]
        tool = AgentTool(name="test_agent", skills_dirs=dirs)
        assert tool._skills_dirs == dirs

    def test_init_stores_priority_dir(self):
        """priority_dir is stored on the instance."""
        priority = Path("/tmp/priority")
        tool = AgentTool(name="test_agent", priority_dir=priority)
        assert tool._priority_dir == priority

    def test_init_defaults_to_none(self):
        """Both skills params default to None."""
        tool = AgentTool(name="test_agent")
        assert tool._skills_dirs is None
        assert tool._priority_dir is None


class TestAgentToolSkillsExecution:
    """Test that sub-agent registries include SkillsTool when skills are available."""

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.agent.get_cached_default_registry")
    def test_execute_adds_skills_tool_when_available(
        self,
        mock_registry_fn,
        mock_react_agent,
        mock_checkpointer,
        mock_aggregator,
    ):
        """Sub-agent registry contains skills_tool when skills are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            _create_skill(skills_dir, "my_skill", "A test skill", "Do things.")

            # Default registry returns a few fake tools
            fake_tool = MagicMock()
            fake_tool.name = "fake_tool"
            mock_base_registry = MagicMock()
            mock_base_registry.tools = [fake_tool]
            mock_registry_fn.return_value = mock_base_registry

            # Agent returns a simple response
            mock_agent_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "done"
            mock_agent_instance.invoke.return_value = mock_response
            mock_react_agent.return_value = mock_agent_instance

            # Checkpointer: raise so we fall back to raw response (simpler)
            mock_cp_instance = MagicMock()
            mock_cp_instance.generate_checkpoint.side_effect = Exception(
                "no checkpoint"
            )
            mock_checkpointer.return_value = mock_cp_instance

            tool = AgentTool(name="test_agent", skills_dirs=[skills_dir])
            tool.execute(task="Do something", context="ctx")

            # Inspect the ToolRegistry passed to ReActAgent
            call_kwargs = mock_react_agent.call_args[1]
            registry_passed = call_kwargs["tools"]

            tool_names = [t.name for t in registry_passed.tools]
            assert "skills_tool" in tool_names

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.agent.get_cached_default_registry")
    def test_execute_no_skills_tool_when_no_skills(
        self,
        mock_registry_fn,
        mock_react_agent,
        mock_checkpointer,
        mock_aggregator,
    ):
        """Sub-agent registry does NOT contain skills_tool when no skills exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_skills_dir = Path(tmpdir) / "skills"
            empty_skills_dir.mkdir()
            # No skill subdirectories → loader finds nothing

            fake_tool = MagicMock()
            fake_tool.name = "fake_tool"
            mock_base_registry = MagicMock()
            mock_base_registry.tools = [fake_tool]
            mock_registry_fn.return_value = mock_base_registry

            mock_agent_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "done"
            mock_agent_instance.invoke.return_value = mock_response
            mock_react_agent.return_value = mock_agent_instance

            mock_cp_instance = MagicMock()
            mock_cp_instance.generate_checkpoint.side_effect = Exception(
                "no checkpoint"
            )
            mock_checkpointer.return_value = mock_cp_instance

            tool = AgentTool(name="test_agent", skills_dirs=[empty_skills_dir])
            tool.execute(task="Do something", context="ctx")

            call_kwargs = mock_react_agent.call_args[1]
            registry_passed = call_kwargs["tools"]
            tool_names = [t.name for t in registry_passed.tools]
            assert "skills_tool" not in tool_names

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.agent.get_cached_default_registry")
    def test_cached_default_registry_not_mutated(
        self,
        mock_registry_fn,
        mock_react_agent,
        mock_checkpointer,
        mock_aggregator,
    ):
        """The original cached registry is not mutated when skills are injected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            _create_skill(skills_dir, "my_skill", "A test skill", "Do things.")

            original_tools = [MagicMock(name="fake_tool")]
            mock_base_registry = MagicMock()
            mock_base_registry.tools = original_tools
            mock_registry_fn.return_value = mock_base_registry

            mock_agent_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "done"
            mock_agent_instance.invoke.return_value = mock_response
            mock_react_agent.return_value = mock_agent_instance

            mock_cp_instance = MagicMock()
            mock_cp_instance.generate_checkpoint.side_effect = Exception(
                "no checkpoint"
            )
            mock_checkpointer.return_value = mock_cp_instance

            tool = AgentTool(name="test_agent", skills_dirs=[skills_dir])
            tool.execute(task="Task", context="ctx")

            # The original mock registry's tools list must not have been mutated
            assert mock_base_registry.tools == original_tools


@pytest.mark.asyncio
class TestAgentToolSkillsAsyncExecution:
    """Test async execution with skills integration."""

    @patch("kader.tools.agent.ContextAggregator")
    @patch("kader.tools.agent.Checkpointer")
    @patch("kader.agent.agents.ReActAgent")
    @patch("kader.tools.agent.get_cached_default_registry")
    async def test_aexecute_adds_skills_tool_when_available(
        self,
        mock_registry_fn,
        mock_react_agent,
        mock_checkpointer,
        mock_aggregator,
    ):
        """Async: sub-agent registry contains skills_tool when skills are available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()
            _create_skill(skills_dir, "my_skill", "A test skill", "Do things.")

            fake_tool = MagicMock()
            fake_tool.name = "fake_tool"
            mock_base_registry = MagicMock()
            mock_base_registry.tools = [fake_tool]
            mock_registry_fn.return_value = mock_base_registry

            mock_agent_instance = MagicMock()
            mock_response = MagicMock()
            mock_response.content = "done"

            async def async_invoke(task):
                return mock_response

            mock_agent_instance.ainvoke = async_invoke
            mock_react_agent.return_value = mock_agent_instance

            mock_cp_instance = MagicMock()
            mock_cp_instance.agenerate_checkpoint.side_effect = Exception(
                "no checkpoint"
            )
            mock_checkpointer.return_value = mock_cp_instance

            tool = AgentTool(name="test_agent", skills_dirs=[skills_dir])
            await tool.aexecute(task="Do something", context="ctx")

            call_kwargs = mock_react_agent.call_args[1]
            registry_passed = call_kwargs["tools"]
            tool_names = [t.name for t in registry_passed.tools]
            assert "skills_tool" in tool_names


class TestPlannerExecutorWorkflowSkills:
    """Test PlannerExecutorWorkflow skills integration."""

    @patch("kader.workflows.planner_executor.PlanningAgent")
    def test_workflow_propagates_skills_dirs_to_agent_tool(self, mock_planning_agent):
        """PlannerExecutorWorkflow stores skills_dirs and passes them to AgentTool."""
        from kader.workflows.planner_executor import PlannerExecutorWorkflow

        mock_planning_agent.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            workflow = PlannerExecutorWorkflow(
                name="test_workflow",
                skills_dirs=[skills_dir],
            )

            assert workflow.skills_dirs == [skills_dir]

            # Find the AgentTool(s) registered on the planner's registry
            # _build_planner registers tools on a ToolRegistry before creating PlanningAgent
            # Inspect via mock call args: PlanningAgent was called with tools=registry
            call_kwargs = mock_planning_agent.call_args[1]
            planner_registry = call_kwargs["tools"]

            # Planner registry should NOT have skills_tool
            planner_tool_names = [t.name for t in planner_registry.tools]
            assert "skills_tool" not in planner_tool_names

            # But the AgentTool inside should carry the skills_dirs
            agent_tools = [
                t for t in planner_registry.tools if isinstance(t, AgentTool)
            ]
            assert len(agent_tools) > 0
            for agent_tool in agent_tools:
                assert agent_tool._skills_dirs == [skills_dir]

    @patch("kader.workflows.planner_executor.PlanningAgent")
    def test_workflow_propagates_priority_dir_to_agent_tool(self, mock_planning_agent):
        """PlannerExecutorWorkflow stores priority_dir and passes it to AgentTool."""
        from kader.workflows.planner_executor import PlannerExecutorWorkflow

        mock_planning_agent.return_value = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            priority = Path(tmpdir) / "priority"
            priority.mkdir()

            workflow = PlannerExecutorWorkflow(
                name="test_workflow",
                priority_dir=priority,
            )

            assert workflow.priority_dir == priority

            call_kwargs = mock_planning_agent.call_args[1]
            planner_registry = call_kwargs["tools"]

            agent_tools = [
                t for t in planner_registry.tools if isinstance(t, AgentTool)
            ]
            for agent_tool in agent_tools:
                assert agent_tool._priority_dir == priority
