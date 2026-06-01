# Subagents

Subagents extend the Planner-Executor workflow with specialized, domain-specific agents. They are defined as YAML files and automatically discovered — each subagent becomes a dedicated tool the planner can delegate tasks to.

## Overview

In the Planner-Executor workflow:

1. **Planner** (PlanningAgent) breaks complex tasks into a plan using TodoTool
2. **Executor** (default AgentTool) handles general sub-tasks
3. **Custom Subagents** handle specialized tasks (code review, research, testing, etc.)

Subagents provide:

- **Domain specialization**: Each subagent has its own system prompt tuned for a specific task
- **Tool isolation**: Subagents use only the tools they need
- **Separate memory**: Each subagent runs with its own isolated conversation context
- **Automatic context aggregation**: Subagent outputs are merged into the session checkpoint

## Subagent File Format

Subagents are loaded from two directories:

- `./.kader/subagents/` — Project-level subagents (always enabled)
- `~/.kader/subagents/` — User-level subagents (gated by `settings.json`)

### Directory Structure

Subagents can be defined in two formats:

**Option 1: Flat file**

```
~/.kader/subagents/
├── code_reviewer.yaml
├── research_agent.yaml
└── test_agent.yaml
```

**Option 2: Directory with template file**

```
~/.kader/subagents/
├── code_reviewer/
│   └── template.yaml
├── research_agent/
│   └── template.yaml
└── test_agent/
    └── template.yaml
```

### YAML Schema

```yaml
name: code-reviewer
objective: >
  Use for code review tasks including analyzing code quality,
  finding bugs, suggesting improvements, and reviewing pull requests
system_prompt: |
  You are an expert code reviewer specializing in thorough,
  constructive code reviews.

  ## Focus Areas
  - Code quality and readability
  - Bug detection and edge case analysis
  - Performance optimization opportunities
  - Security vulnerability identification
  - Test coverage assessment

  ## Approach
  1. Read and understand the code thoroughly before reviewing
  2. Check for logical errors and edge cases
  3. Evaluate naming conventions and code organization
  4. Assess error handling and defensive programming
  5. Look for potential performance bottlenecks
tools:
  - read_file
  - read_directory
  - grep
  - glob
```

### Field Reference

| Field           | Description                                                                                                |
| --------------- | ---------------------------------------------------------------------------------------------------------- |
| `name`          | Unique subagent identifier (kebab-case preferred). Used as the tool name the planner invokes.              |
| `objective`     | Description to help the planner decide when to use this subagent. Included in the planner's system prompt. |
| `system_prompt` | The system prompt defining the subagent's behavior, instructions, and constraints.                         |
| `tools`         | List of tool name strings to make available to the subagent. Empty list = all standard tools.              |

## How Subagents Work

### Discovery & Registration

1. On workflow creation, `SubagentLoader` scans the subagent directories
2. User-level subagents are filtered against `settings.json` (`subagents` field)
3. Each discovered subagent is converted to an `AgentTool` via `AgentTool.from_subagent_config()`
4. The `AgentTool` is registered in the planner's tool registry with the subagent's name and objective
5. The planner's system prompt includes a listing of all available subagents and their objectives

### Execution

1. The planner decides which subagent to invoke based on the task and the subagent's objective
2. The planner calls the subagent as a tool (e.g., `code-reviewer`)
3. The `AgentTool` spawns a `ReActAgent` with the subagent's system prompt, tools, and provider
4. The subagent executes its task with isolated memory
5. Output is compressed, checkpointed, and aggregated into the session context
6. The planner receives a summary and continues with the next steps

### User-Level Subagent Configuration

User-level subagents in `~/.kader/subagents/` are controlled via `~/.kader/settings.json`:

```json
{
  "subagents": [
    { "name": "code-reviewer", "enabled": "true" },
    { "name": "research-agent", "enabled": "false" }
  ]
}
```

- `name`: Matches the name field in the subagent YAML (or the file stem)
- `enabled`: `"true"` to enable, `"false"` to disable
- Subagents not listed in settings are **skipped** (not available to the planner)
- **Project-level** subagents in `./.kader/subagents/` are always enabled regardless of settings

## Example Subagents

### Code Reviewer

```yaml
name: code-reviewer
objective: "Use for code review tasks including analyzing code quality, finding bugs, suggesting improvements, and reviewing pull requests"
system_prompt: |
  You are an expert code reviewer. Focus on:
  - Code quality and readability
  - Bug detection and edge case analysis
  - Performance optimization opportunities
  - Security vulnerability identification
  - Adherence to best practices and design patterns
  - Test coverage and test quality assessment

  Always explain WHY something should change, not just WHAT to change.
  Provide specific code examples for each suggestion.
tools:
  - read_file
  - read_directory
  - grep
  - glob
```

### Research Agent

```yaml
name: research-agent
objective: "Use for web research tasks, information gathering, fact-checking, and staying up-to-date with latest technologies"
system_prompt: |
  You are a research specialist. Your task is to gather accurate,
  up-to-date information from the web.
  - Use web_search to find relevant sources
  - Use web_fetch to read full articles
  - Synthesize information from multiple sources
  - Always cite your sources
  - Highlight any conflicting information
tools:
  - web_search
  - web_fetch
```

## Programmatic Usage

### SubagentLoader

```python
from kader.workflows.subagents import SubagentLoader

loader = SubagentLoader()

# Custom directories
loader = SubagentLoader(
    subagents_dirs=[Path("./my_subagents")],
    priority_dir=Path("./project_subagents"),
)

# List all available subagents
all_configs = loader.list_subagents()

# Load a specific subagent
config = loader.load_subagent("code-reviewer")
print(config.name)          # "code-reviewer"
print(config.objective)     # Description for the planner
print(config.system_prompt) # The system prompt
print(config.tools)         # ["read_file", ...]

# Get formatted description for LLM prompts
description = loader.get_description()
```

### SubagentConfig

```python
from kader.workflows.subagents import SubagentConfig

config = SubagentConfig(
    name="test-agent",
    objective="Use for running tests and analyzing test results",
    system_prompt="You are a testing specialist...",
    tools=["read_file", "execute_command", "grep"],
)
```

### PlannerExecutorWorkflow with Subagents

```python
from pathlib import Path
from kader.workflows import PlannerExecutorWorkflow
from kader.providers import OllamaProvider

workflow = PlannerExecutorWorkflow(
    name="my_workflow",
    provider=OllamaProvider(model="llama3.2"),
    interrupt_before_tool=True,

    # Custom subagent directories
    subagents_dirs=[Path("./custom_subagents")],

    # Filter user-level subagents via settings
    enabled_subagents=[
        {"name": "code-reviewer", "enabled": "true"},
        {"name": "research-agent", "enabled": "false"},
    ],
)

# The planner will automatically discover and use subagents
result = workflow.run(
    "Review the code in src/ and research the latest testing frameworks"
)
```

## Subagent UI in CLI

When the planner delegates a task to a subagent, the CLI shows a distinct user interface indicating which agent is executing actions:

```
[Kader is thinking... spinner]

[^^] Executor Started
│ Entering subagent mode — actions are now executed by Executor │

[Executor is working... spinner]

  ⚡ [executor] read_file: Analyzing codebase structure...
  [+] [executor] read_file completed successfully

[^^] Executor — Tool Confirmation
│ execute execute_command: pytest
  Approve? (Y/n)

[✓] Executor finished
  [+] executor completed successfully
```

Key visual indicators:

- **Entry banner**: `[⚙] Executor Started` cyan panel when entering a subagent
- **Context prefix**: `⚡ [executor] read_file:` — tool messages are prefixed with the subagent name
- **Dynamic spinner**: Shows `Executor is working...` instead of `Kader is thinking...`
- **Result prefix**: `[+] [executor] read_file completed` — results are prefixed with the subagent name
- **Exit footer**: `[✓] Executor finished` when the subagent completes
- **Tool confirmation**: The panel title includes the subagent name: `[⚙] Executor — Tool Confirmation`

## Subagent Memory & Persistence

Each subagent execution creates isolated persistence files:

```
~/.kader/memory/sessions/<session-id>/executors/
├── <agent-name>-<uuid>/
│   ├── conversation.json   # Subagent's conversation history
│   └── checkpoint.md       # Subagent's action summary
└── checkpoint.md           # Aggregated context from all subagents
```

The `ContextAggregator` automatically merges individual subagent checkpoints into the session's aggregated checkpoint, so the planner always has access to previous subagent results.

## Subagents vs Special Commands

| Aspect        | Subagents                               | Special Commands                          |
| ------------- | --------------------------------------- | ----------------------------------------- |
| Invocation    | Called by the planner as a tool         | Called from CLI via `/<name>`             |
| Configuration | YAML file with system prompt + tools    | `CONTENT.md` file with plain instructions |
| When to use   | Part of automated workflow              | User-initiated ad-hoc tasks               |
| Visibility    | Invisible to user (internal delegation) | Explicit CLI invocation                   |
