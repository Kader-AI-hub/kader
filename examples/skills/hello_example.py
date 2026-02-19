"""
Example of how to use the SkillsTool.

This example demonstrates:
1. Loading skills from a directory
2. Using SkillLoader to list and load skills
3. Using SkillsTool as an agent tool
"""

from pathlib import Path

from kader.tools.skills import SkillLoader, SkillsTool


def main():
    # Define the skills directory
    skills_dir = Path(__file__).parent / "skills"

    # Create a SkillLoader
    loader = SkillLoader([skills_dir])

    # List all available skills
    print("=== Available Skills ===")
    print(loader.get_description())
    print()

    # Load a specific skill
    print("=== Loading 'hello' skill ===")
    skill = loader.load_skill("hello")
    if skill:
        print(f"Name: {skill.name}")
        print(f"Description: {skill.description}")
        print(f"Content:\n{skill.content}")
    print()

    # Use SkillsTool (for agent integration)
    print("=== Using SkillsTool ===")
    tool = SkillsTool([skills_dir])

    # The tool can be used directly
    result = tool.execute(name="hello")
    print(f"Tool result: {result}")


if __name__ == "__main__":
    main()
