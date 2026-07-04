import pytest

from agentlab.tools import (
    Tool,
    ToolError,
    ToolRegistry,
    calculator,
    default_registry,
    echo,
)


def _dummy_tool(name: str = "noop") -> Tool:
    return Tool(
        name=name,
        description="d",
        parameters={"type": "object", "properties": {}},
        handler=lambda: "ok",
    )


def should_evaluate_expression_when_valid():
    assert calculator("2 * (3 + 4)") == "14"
    assert calculator("10 / 4") == "2.5"
    assert calculator("-3 + 5") == "2"


def should_raise_when_expression_is_invalid():
    with pytest.raises(ToolError):
        calculator("2 +")


def should_reject_non_arithmetic_expressions():
    with pytest.raises(ToolError):
        calculator("__import__('os').system('ls')")


def should_echo_text_verbatim():
    assert echo(text="hi there") == "hi there"


def should_register_and_execute_tool():
    registry = ToolRegistry()
    registry.register(_dummy_tool())
    assert registry.execute("noop", {}) == "ok"
    assert "noop" in registry.names()


def should_raise_when_registering_duplicate():
    registry = ToolRegistry()
    registry.register(_dummy_tool())
    with pytest.raises(ToolError):
        registry.register(_dummy_tool())


def should_raise_when_tool_unknown():
    registry = ToolRegistry()
    with pytest.raises(ToolError):
        registry.get("missing")


def should_wrap_handler_exceptions_as_tool_error():
    registry = default_registry()
    with pytest.raises(ToolError):
        registry.execute("calculator", {"expression": "1 +"})


def should_expose_openai_specs_for_all_tools():
    registry = default_registry()
    specs = registry.specs()
    names = {spec["function"]["name"] for spec in specs}
    assert {"calculator", "echo"} <= names
    for spec in specs:
        assert spec["type"] == "function"
