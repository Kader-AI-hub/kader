---
name: calculator
description: Skill for mathematical calculations
---

# Calculator Skill

This skill provides instructions for performing mathematical calculations.

## Supported Operations

- Addition: `+`
- Subtraction: `-`
- Multiplication: `*`
- Division: `/`
- Exponentiation: `**`
- Modulo: `%`

## Usage

Execute the calculator script with an expression:

```bash
python examples/skills/calculator/scripts/calculate.py <expression>
```

### Examples

```bash
python examples/skills/calculator/scripts/calculate.py "2 + 2"
# Output: 4

python examples/skills/calculator/scripts/calculate.py "10 * 5"
# Output: 50

python examples/skills/calculator/scripts/calculate.py "(2 + 3) * 4"
# Output: 20
```

## Error Handling

- Invalid expressions will show an error message
- Division by zero will show "Error: division by zero"
- Syntax errors will display a usage message
