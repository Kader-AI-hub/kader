---
name: hello
description: Skill for ALL greeting requests
---

# Hello Skill

This skill provides the greeting format you must follow.

## How to greet

Always greet the user with:
- A warm welcome
- Their name if mentioned
- A friendly emoji

## Example greetings

- "Hello! Welcome! How can I help you today? 👋"
- "Hi there! Great to see you! 😊"

## Executing the hello script

When a name is provided, execute the hello script:

```bash
python examples/skills/hello/scripts/hello.py <name>
```

Example:
```bash
python examples/skills/hello/scripts/hello.py World
# Output: Hello, World! Welcome! How can I help you today? 👋
```
