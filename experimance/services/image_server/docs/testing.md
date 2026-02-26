# Testing Guide

## Overview

The image server includes pytest suites and ZMQ utilities for validating message flows, generator integrations, and audio generation.

Environment assumptions:

- Python 3.11 with `uv`
- ZMQ ports available (see `DEFAULT_PORTS`)

When to use:

- You are validating ZMQ request/response paths
- You are debugging generator behavior

When not to use:

- You only need to smoke test the CLI; use `uv run -m image_server.cli` instead

Files touched:

- `services/image_server/tests/`

## Setup

```bash
uv sync --package image-server --extra dev
```

Audio generation tests also require:

```bash
uv sync --package image-server --extra audio_gen
```

## Usage

### Pytest suites

```bash
uv run -m pytest services/image_server/tests
```

### ZMQ test runner

```bash
uv run python services/image_server/tests/run_zmq_tests.py
```

### Validate ZMQ addresses

```bash
uv run python services/image_server/tests/validate_zmq_addresses.py
```

### Direct audio generator test

```bash
uv run python services/image_server/tests/test_audio_direct.py -i
```

## Troubleshooting

- Missing dependencies: install `--extra dev` and `--extra audio_gen` as needed.
- ZMQ port conflicts: ensure ports 5555, 5564, 5565 are free.

## Integrations

- ZMQ flow: `services/image_server/docs/zmq_messaging.md`
