import tempfile
from pathlib import Path

import pytest

from kader.tools.skills import Skill, SkillLoader, SkillsTool


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir) / "skills"
        skills_dir.mkdir()
        yield skills_dir


@pytest.fixture
def temp_cwd_skills_dir():
    """Create a temporary .kader directory in cwd."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd_kader = Path(tmpdir) / ".kader"
        cwd_kader.mkdir()
        yield cwd_kader


def create_skill(skills_dir: Path, name: str, description: str, content: str) -> None:
    """Helper to create a skill directory with SKILL.md file."""
    skill_dir = skills_dir / name
    skill_dir.mkdir()

    frontmatter = f"""---
name: {name}
description: {description}
---

{content}
"""
    (skill_dir / "SKILL.md").write_text(frontmatter)


class TestSkillLoader:
    """Test cases for SkillLoader."""

    def test_load_skill_from_single_directory(self, temp_skills_dir):
        """Test loading a skill from a single directory."""
        create_skill(
            temp_skills_dir,
            "hello",
            "Skill for greeting users",
            "# Hello Skill\n\nGreet with warmth.",
        )

        loader = SkillLoader([temp_skills_dir])
        skill = loader.load_skill("hello")

        assert skill is not None
        assert skill.name == "hello"
        assert skill.description == "Skill for greeting users"
        assert "Greet with warmth" in skill.content

    def test_load_nonexistent_skill(self, temp_skills_dir):
        """Test loading a skill that doesn't exist."""
        loader = SkillLoader([temp_skills_dir])
        skill = loader.load_skill("nonexistent")

        assert skill is None

    def test_list_skills(self, temp_skills_dir):
        """Test listing all skills."""
        create_skill(temp_skills_dir, "hello", "Greeting skill", "Greet users.")
        create_skill(temp_skills_dir, "goodbye", "Farewell skill", "Say goodbye.")

        loader = SkillLoader([temp_skills_dir])
        skills = loader.list_skills()

        assert len(skills) == 2
        skill_names = [s.name for s in skills]
        assert "hello" in skill_names
        assert "goodbye" in skill_names

    def test_list_skills_empty_directory(self, temp_skills_dir):
        """Test listing skills when directory is empty."""
        loader = SkillLoader([temp_skills_dir])
        skills = loader.list_skills()

        assert skills == []

    def test_priority_home_over_cwd(self, temp_skills_dir, temp_cwd_skills_dir):
        """Test that ~/.kader/skills takes priority over ./.kader/."""
        create_skill(temp_cwd_skills_dir, "test_skill", "CWD version", "CWD content")
        create_skill(temp_skills_dir, "test_skill", "Home version", "Home content")

        loader = SkillLoader([temp_skills_dir, temp_cwd_skills_dir])
        skill = loader.load_skill("test_skill")

        assert skill is not None
        assert skill.description == "Home version"
        assert skill.content == "Home content"

    def test_get_description(self, temp_skills_dir):
        """Test getting formatted description of skills."""
        create_skill(temp_skills_dir, "hello", "Greeting skill", "Content")
        create_skill(temp_skills_dir, "search", "Web search skill", "Content")

        loader = SkillLoader([temp_skills_dir])
        description = loader.get_description()

        assert "hello: Greeting skill" in description
        assert "search: Web search skill" in description

    def test_get_description_empty(self, temp_skills_dir):
        """Test getting description when no skills exist."""
        loader = SkillLoader([temp_skills_dir])
        description = loader.get_description()

        assert description == "No skills available."

    def test_skills_directory_not_exists(self):
        """Test when skills directory doesn't exist."""
        loader = SkillLoader([Path("/nonexistent/path")])
        skills = loader.list_skills()

        assert skills == []


class TestSkillsTool:
    """Test cases for SkillsTool."""

    def test_skills_tool_execute_success(self, temp_skills_dir):
        """Test successfully loading a skill."""
        create_skill(
            temp_skills_dir,
            "hello",
            "Greeting skill",
            "# Hello\n\nAlways greet warmly.",
        )

        tool = SkillsTool([temp_skills_dir])
        result = tool.execute(name="hello")

        assert "error" not in result
        assert result["name"] == "hello"
        assert result["description"] == "Greeting skill"
        assert "Always greet warmly" in result["content"]

    def test_skills_tool_execute_not_found(self, temp_skills_dir):
        """Test loading a non-existent skill."""
        tool = SkillsTool([temp_skills_dir])
        result = tool.execute(name="nonexistent")

        assert "error" in result
        assert "not found" in result["error"]
        assert "available_skills" in result

    def test_skills_tool_execute_no_name(self, temp_skills_dir):
        """Test executing without providing a skill name."""
        tool = SkillsTool([temp_skills_dir])
        result = tool.execute(name="")

        assert "error" in result
        assert "required" in result["error"].lower()

    def test_skills_tool_get_interruption_message(self, temp_skills_dir):
        """Test getting interruption message."""
        tool = SkillsTool([temp_skills_dir])
        message = tool.get_interruption_message(name="hello")

        assert message == "execute skills_tool: hello"

    def test_skills_tool_schema(self, temp_skills_dir):
        """Test that the tool has correct schema."""
        tool = SkillsTool([temp_skills_dir])

        assert tool.name == "skills_tool"
        assert tool.schema.category.value == "utility"

        params = {p.name: p for p in tool.schema.parameters}
        assert "name" in params
        assert params["name"].type == "string"


@pytest.mark.asyncio
async def test_skills_tool_aexecute(temp_skills_dir):
    """Test async execution."""
    create_skill(temp_skills_dir, "test", "Test skill", "Test content")

    tool = SkillsTool([temp_skills_dir])
    result = await tool.aexecute(name="test")
    assert result["name"] == "test"


class TestSkill:
    """Test cases for Skill dataclass."""

    def test_skill_creation(self):
        """Test creating a Skill instance."""
        skill = Skill(
            name="test",
            description="Test description",
            content="Test content",
        )

        assert skill.name == "test"
        assert skill.description == "Test description"
        assert skill.content == "Test content"
