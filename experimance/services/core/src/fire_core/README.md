# fire_core

Core orchestration service for the **Feed the Fires** interactive art installation.

Listens for audience stories and live conversation transcripts, uses an LLM to infer environmental settings, then drives a panoramic image pipeline (base image → tiles → display) with smart interruption.

## Quick start

```bash
scripts/project fire
uv run -m fire_core

# Mock LLM (no API key needed)
uv run -m fire_core --llm-provider mock

# CLI test utility
uv run -m fire_core.cli --interactive
```

## Documentation

Full documentation is in [`services/core/docs/fire-core.md`](../../docs/fire-core.md).
