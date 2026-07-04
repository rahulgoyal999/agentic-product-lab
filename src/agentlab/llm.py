"""LLM client abstraction.

Two implementations:
- StubLLM: deterministic, offline. Lets the whole agent loop run and be tested
  with no API key. It knows how to call the built-in calculator tool.
- OpenAILLM: thin wrapper over an OpenAI-compatible chat.completions endpoint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Either a final answer (`content`) or one or more `tool_calls`."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class LLM(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse: ...


_MATH_RE = re.compile(r"[-+*/%^().\d\s]{3,}")


class StubLLM:
    """Offline, deterministic LLM stand-in.

    Behaviour:
    - If the latest turn is a tool result, produce a final answer that reports it.
    - Else if the user's message contains an arithmetic expression, call the
      `calculator` tool.
    - Otherwise, echo a canned reply.
    """

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        last = messages[-1] if messages else {}

        if last.get("role") == "tool":
            return LLMResponse(content=f"Result: {last.get('content', '')}")

        user_text = self._last_user_text(messages)
        available = {t["function"]["name"] for t in tools}

        if "calculator" in available:
            expr = self._extract_expression(user_text)
            if expr is not None:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="calculator",
                            arguments={"expression": expr},
                        )
                    ]
                )

        return LLMResponse(content=f"(stub) You said: {user_text}")

    @staticmethod
    def _last_user_text(messages: list[dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    @staticmethod
    def _extract_expression(text: str) -> str | None:
        candidates = _MATH_RE.findall(text)
        for candidate in candidates:
            stripped = candidate.strip()
            if any(op in stripped for op in "+-*/%^") and any(c.isdigit() for c in stripped):
                return stripped
        return None


class OpenAILLM:
    """Wrapper over an OpenAI-compatible chat.completions endpoint."""

    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai provider")
        from openai import OpenAI

        self._client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = settings.openai_model

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(id=call.id, name=call.function.name, arguments=arguments)
            )

        return LLMResponse(content=message.content, tool_calls=tool_calls)


def build_llm(settings: Settings) -> LLM:
    """Factory: pick an LLM implementation from settings."""
    if settings.provider == "openai":
        return OpenAILLM(settings)
    return StubLLM()
