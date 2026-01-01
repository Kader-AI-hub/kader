## General Instructions

- Current Operating System is Windows, Use cmd to run terminal commands.
- use uv as python dependency manager.
- check `pyproject.toml` file to search for dependencies first before add dependencies.
- use context7 mcp tools to search to get accurate docs if needed.
- to install new packages/dependecies use `uv add <package-name>`.
- to remove packages/dependecies use `uv remove <package-name>`.
- use `uv run <script-name>` to run python scripts.
- don't access secrets like api keys, tokens, etc. from environment variables, or files like `.env`.
