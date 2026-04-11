# tests/fixtures/python_project/sample.py
import os
from pathlib import Path


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b


class Calculator:
    """A simple calculator."""

    def __init__(self, initial: int = 0) -> None:
        self.value = initial

    def multiply(self, factor: int) -> int:
        """Multiply current value by factor."""
        self.value *= factor
        return self.value

    def reset(self) -> None:
        """Reset to zero."""
        self.value = 0
