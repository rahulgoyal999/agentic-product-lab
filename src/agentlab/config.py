"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional at runtime
    pass


@dataclass(frozen=True)
class Settings:
    """Runtime settings, sourced from the environment.

    Defaults to the "stub" provider so the agent runs with no API key.
    """

    provider: str = "stub"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    max_steps: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("AGENT_PROVIDER", "stub").strip().lower() or "stub"
        base_url = os.getenv("OPENAI_BASE_URL") or None
        return cls(
            provider=provider,
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=base_url,
            max_steps=int(os.getenv("AGENT_MAX_STEPS", "6")),
        )
