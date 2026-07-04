"""Command-line entrypoint for the agent."""

from __future__ import annotations

import argparse
import sys

from .agent import Agent
from .config import Settings
from .llm import build_llm
from .tools import default_registry


def build_agent(settings: Settings | None = None) -> Agent:
    settings = settings or Settings.from_env()
    return Agent(
        llm=build_llm(settings),
        registry=default_registry(),
        max_steps=settings.max_steps,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the agentic-product-lab agent.")
    parser.add_argument("prompt", nargs="*", help="Prompt to send to the agent.")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    agent = build_agent(settings)

    prompt = " ".join(args.prompt).strip()
    if not prompt:
        print(f"[provider={settings.provider}] Enter a prompt (Ctrl-D to exit).")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            print(agent.run(line).answer)
        return 0

    print(agent.run(prompt).answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
