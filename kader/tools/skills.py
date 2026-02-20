from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter

from kader.tools.base import BaseTool, ParameterSchema, ToolCategory


@dataclass
class Skill:
    """Represents a skill with metadata and content."""

    name: str
    description: str
    content: str


class SkillLoader:
    """Loads skills from skill directories."""

    def __init__(
        self, skills_dirs: list[Path] | None = None, priority_dir: Path | None = None
    ) -> None:
        """
        Initialize the skill loader.

        Args:
            skills_dirs: List of directories to load skills from.
                        If None, defaults to ~/.kader/skills and ./.kader/
            priority_dir: Optional directory to check first (higher priority).
                         If provided, this directory is checked before others.
        """
        self._priority_dir = priority_dir

        if skills_dirs is None:
            home_skills = Path.home() / ".kader" / "skills"
            cwd_skills = Path.cwd() / ".kader"
            skills_dirs = [home_skills, cwd_skills]

        if priority_dir is not None:
            skills_dirs = [priority_dir] + skills_dirs

        self.skills_dirs = skills_dirs

    def _find_skill_file(self, name: str) -> tuple[Path, Path] | None:
        """
        Find the skill file in the skills directories.

        Args:
            name: Name of the skill to find

        Returns:
            Tuple of (skill_dir, skill_file) if found, None otherwise
        """
        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue
            skill_dir = skills_dir / name
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                return (skills_dir, skill_file)
        return None

    def load_skill(self, name: str) -> Skill | None:
        """
        Load a skill by name.

        Args:
            name: Name of the skill to load

        Returns:
            Skill object if found, None otherwise
        """
        result = self._find_skill_file(name)
        if result is None:
            return None

        _, skill_file = result
        parsed = frontmatter.load(str(skill_file))

        metadata = parsed.metadata or {}
        return Skill(
            name=str(metadata.get("name", name)),
            description=str(metadata.get("description", "")),
            content=parsed.content,
        )

    def list_skills(self) -> list[Skill]:
        """
        List all available skills from all directories.

        Skills from ~/.kader/skills take priority over skills
        with the same name in other directories.

        Returns:
            List of all available skills
        """
        seen_names: set[str] = set()
        skills: list[Skill] = []

        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue

            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue

                name = skill_dir.name
                if name in seen_names:
                    continue

                skill = self.load_skill(name)
                if skill:
                    skills.append(skill)
                    seen_names.add(name)

        return skills

    def get_description(self) -> str:
        """
        Get a formatted description of all available skills.

        Returns:
            Formatted string listing all skills
        """
        skills = self.list_skills()

        if not skills:
            return "No skills available."

        skills_listing = "\n".join(f"  - {s.name}: {s.description}" for s in skills)

        return skills_listing


class SkillsTool(BaseTool[dict[str, Any]]):
    """Tool for loading and managing agent skills."""

    def __init__(
        self,
        skills_dirs: list[Path] | None = None,
        priority_dir: Path | None = None,
    ) -> None:
        """
        Initialize the skills tool.

        Args:
            skills_dirs: Optional list of custom skill directories.
                        Defaults to ~/.kader/skills and ./.kader/
            priority_dir: Optional directory to check first (higher priority).
        """
        self._skill_loader = SkillLoader(skills_dirs, priority_dir)

        description = (
            "Load a skill to get specialized instructions for a task. "
            "Use this when you need specific domain knowledge or specialized procedures. "
            f"Available skills:\n{self._skill_loader.get_description()}"
        )

        super().__init__(
            name="skills_tool",
            description=description,
            category=ToolCategory.UTILITY,
            parameters=[
                ParameterSchema(
                    name="name",
                    type="string",
                    description="The exact name of the skill to load",
                ),
            ],
        )

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        """
        Execute the skills tool.

        Args:
            **kwargs: Must include 'name' parameter

        Returns:
            Dictionary with skill information
        """
        name = kwargs.get("name", "")
        if not name:
            return {
                "error": "Skill name is required",
                "available_skills": self._skill_loader.get_description(),
            }

        skill = self._skill_loader.load_skill(name)
        if skill is None:
            available = self._skill_loader.get_description()
            return {
                "error": f"Skill '{name}' not found",
                "available_skills": available,
            }

        return {
            "name": skill.name,
            "description": skill.description,
            "content": skill.content,
        }

    async def aexecute(self, **kwargs: Any) -> dict[str, Any]:
        """Asynchronous execution (delegates to synchronous)."""
        return self.execute(**kwargs)

    def get_interruption_message(self, **kwargs: Any) -> str:
        """
        Get interruption message for user confirmation.

        Args:
            **kwargs: Must include 'name' parameter

        Returns:
            Message describing the skill loading action
        """
        name = kwargs.get("name", "")
        return f"execute skill: {name}"
