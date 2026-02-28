# Contributor Checklist

Use this checklist before submitting a pull request to ensure your contribution meets all requirements.

## Pre-Submission Checklist

### Code Quality
- [ ] Code follows the project's style guidelines (see KADER.md)
- [ ] Python 3.11+ features are used appropriately
- [ ] Type hints are present for all function signatures
- [ ] Naming conventions are followed (snake_case, PascalCase, SCREAMING_SNAKE_CASE)
- [ ] Import organization follows the standard (stdlib, third-party, local)

### Testing
- [ ] All tests pass locally (`uv run pytest -v`)
- [ ] New tests added for new functionality
- [ ] Tests use proper mocking (no real filesystem/network operations)
- [ ] Test files mirror the source directory structure
- [ ] Test functions follow naming convention (`test_<functionality>`)

### Linting & Formatting
- [ ] Ruff linting passes (`uv run ruff check .`)
- [ ] Ruff auto-fix applied (`uv run ruff check . --fix`)
- [ ] Code formatting applied (`uv run ruff format .`)
- [ ] No line exceeds 88 characters

### Documentation
- [ ] Code comments explain "why", not "what"
- [ ] Complex logic has docstrings
- [ ] README updated if adding new features
- [ ] API changes documented if applicable

### Commit & PR
- [ ] Commit messages are clear and descriptive
- [ ] Changes are in a feature branch (not main)
- [ ] PR description explains the changes and motivation
- [ ] Related issues referenced in PR (e.g., "Fixes #123")

## Common Issues to Avoid

### Don't
- Create real files in tests (use `tmp_path` fixture)
- Make real network requests in tests (mock them)
- Use `List`, `Dict` from typing (use built-in `list`, `dict`)
- Edit `uv.lock` manually
- Submit code with unresolved linting errors

### Do
- Use `uv` for all package management
- Mock filesystem operations in tests
- Run `uv run ruff check . --fix` before committing
- Test locally before submitting PR

## Quick Commands Reference

```bash
# Install dependencies
uv sync

# Run all checks (lint-fix, format, test)
uv run python .kader/skills/contributing-to-kader/scripts/dev_helper.py all

# Or run individually:
uv run ruff check . --fix    # Fix linting
uv run ruff format .         # Format code
uv run pytest -v             # Run tests
```

## Questions?

If you're unsure about any item on this checklist, please open a discussion issue before submitting your PR.
