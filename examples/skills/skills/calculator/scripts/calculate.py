#!/usr/bin/env python
"""Simple calculator script that evaluates mathematical expressions."""

import sys


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.

    Args:
        expression: A mathematical expression as a string

    Returns:
        The result as a string, or an error message
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero"
    except SyntaxError:
        return "Error: invalid expression syntax"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate.py <expression>")
        print("Example: python calculate.py '2 + 2'")
        print("Supported: +, -, *, /, **, %")
        sys.exit(1)

    expression = sys.argv[1]
    result = calculate(expression)
    print(result)


if __name__ == "__main__":
    main()
