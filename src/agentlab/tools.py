"""Tool definitions and registry for the agent."""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    """A callable the agent can invoke.

    `parameters` is a JSON-schema object describing the arguments, used to
    advertise the tool to an LLM.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolError(Exception):
    """Raised when a tool cannot be found or fails to execute."""


class ToolRegistry:
    """Holds the tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        return self._tools[name]

    def specs(self) -> list[dict[str, Any]]:
        return [tool.to_openai_spec() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.get(name)
        try:
            return tool.handler(**arguments)
        except ToolError:
            raise
        except Exception as exc:  # surface tool failures to the agent loop
            raise ToolError(f"Tool '{name}' failed: {exc}") from exc

    def names(self) -> list[str]:
        return list(self._tools)


_ALLOWED_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate a numeric expression AST without using eval()."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_safe_eval(node.operand)
    raise ToolError("Unsupported expression")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (safe, no eval)."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression: {expression}") from exc
    result = _safe_eval(tree)
    if result.is_integer():
        return str(int(result))
    return str(result)


def echo(text: str) -> str:
    """Return the text unchanged. Useful as a trivial example tool."""
    return text


def default_registry() -> ToolRegistry:
    """A registry preloaded with the built-in example tools."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression, e.g. '2 * (3 + 4)'.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
            handler=calculator,
        )
    )
    registry.register(
        Tool(
            name="echo",
            description="Echo the provided text back verbatim.",
            parameters={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo."}
                },
                "required": ["text"],
            },
            handler=echo,
        )
    )
    return registry
