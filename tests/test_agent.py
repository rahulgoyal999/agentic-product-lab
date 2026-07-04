import pytest

from agentlab.agent import Agent
from agentlab.llm import LLMResponse, StubLLM, ToolCall, build_llm
from agentlab.config import Settings
from agentlab.tools import default_registry


def _agent(llm=None, max_steps=6) -> Agent:
    return Agent(
        llm=llm or StubLLM(),
        registry=default_registry(),
        max_steps=max_steps,
    )


def should_call_calculator_and_return_result_when_prompt_is_math():
    result = _agent().run("what is 2 * (3 + 4)?")
    assert "14" in result.answer
    assert result.steps == 2  # one tool call turn + one final turn
    assert any(m["role"] == "tool" for m in result.messages)


def should_return_direct_answer_when_no_tool_needed():
    result = _agent().run("hello there")
    assert "hello there" in result.answer
    assert result.steps == 1
    assert not any(m["role"] == "tool" for m in result.messages)


def should_stop_when_max_steps_reached():
    class LoopingLLM:
        def complete(self, messages, tools):
            return LLMResponse(
                tool_calls=[ToolCall(id="c", name="echo", arguments={"text": "x"})]
            )

    result = _agent(llm=LoopingLLM(), max_steps=3).run("go")
    assert "maximum number of steps" in result.answer
    assert result.steps == 3


def should_report_tool_error_without_crashing():
    class BadCallLLM:
        def __init__(self):
            self.called = False

        def complete(self, messages, tools):
            if not self.called:
                self.called = True
                return LLMResponse(
                    tool_calls=[
                        ToolCall(id="c", name="calculator", arguments={"expression": "1 +"})
                    ]
                )
            return LLMResponse(content="done")

    result = _agent(llm=BadCallLLM()).run("compute 1 +")
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert tool_msgs and tool_msgs[0]["content"].startswith("ERROR")


def should_reject_invalid_max_steps():
    with pytest.raises(ValueError):
        _agent(max_steps=0)


def should_build_stub_llm_by_default():
    assert isinstance(build_llm(Settings()), StubLLM)
