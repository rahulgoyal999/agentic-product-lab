"""The core agent loop: think -> (call tools) -> respond."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .llm import LLM, LLMResponse
from .tools import ToolError, ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful agent. Use the available tools when they help you "
    "answer accurately. Keep answers concise."
)


@dataclass
class AgentResult:
    answer: str
    messages: list[dict[str, Any]]
    steps: int


class Agent:
    """Runs a bounded tool-calling loop against an `LLM`."""

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self._llm = llm
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_steps = max_steps

    def run(self, user_input: str) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_input},
        ]
        specs = self._registry.specs()

        for step in range(1, self._max_steps + 1):
            response = self._llm.complete(messages, specs)

            if response.is_final:
                answer = response.content or ""
                messages.append({"role": "assistant", "content": answer})
                return AgentResult(answer=answer, messages=messages, steps=step)

            messages.append(self._assistant_tool_message(response))
            for call in response.tool_calls:
                messages.append(self._run_tool(call))

        # Ran out of steps without a final answer.
        fallback = "Stopped: reached the maximum number of steps."
        messages.append({"role": "assistant", "content": fallback})
        return AgentResult(answer=fallback, messages=messages, steps=self._max_steps)

    @staticmethod
    def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": response.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ],
        }

    def _run_tool(self, call: Any) -> dict[str, Any]:
        try:
            output = self._registry.execute(call.name, call.arguments)
        except ToolError as exc:
            output = f"ERROR: {exc}"
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": output,
        }
