# Agentic Product Lab

Personal practice space for building an agentic product from scratch. Unrelated to any work project.

A minimal, provider-agnostic **tool-calling agent**: an LLM decides when to call
tools, the agent executes them, feeds results back, and loops until it produces a
final answer. It ships with a deterministic offline stub so you can run and test
the whole loop with **no API key**.

## Layout

```
src/agentlab/
  config.py   # settings from environment (.env supported)
  tools.py    # Tool, ToolRegistry, and built-in tools (calculator, echo)
  llm.py      # LLM protocol + StubLLM (offline) and OpenAILLM
  agent.py    # the bounded think -> call tools -> respond loop
  cli.py      # command-line entrypoint
tests/        # pytest suite (runs fully offline via StubLLM)
```

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Runs offline with the stub provider (no key needed):
python -m agentlab.cli "what is 2 * (3 + 4)?"
```

## Using a real model

```bash
cp .env.example .env
# In .env set:
#   AGENT_PROVIDER=openai
#   OPENAI_API_KEY=sk-...
#   OPENAI_MODEL=gpt-4o-mini
python -m agentlab.cli "plan a weekend trip to Lisbon"
```

`OPENAI_BASE_URL` lets you point at any OpenAI-compatible endpoint.

## Tests

```bash
pip install -r requirements.txt
pytest
```

## Extending

Add a tool by registering a `Tool` in `tools.py` (name, description, JSON-schema
parameters, and a handler). The agent advertises it to the model automatically.
