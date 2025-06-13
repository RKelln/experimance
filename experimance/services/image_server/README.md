# Image Server Service

The Image Server Service is responsible for generating images in response to RenderRequest messages via ZeroMQ. It supports multiple generation strategies including mock images, local SDXL, FAL.AI, and OpenAI DALL-E.

## Overview

The Image Server Service:

- Listens for `RenderRequest` messages on the events channel
- Generates images based on text prompts, era, biome and optional depth maps
- Publishes `ImageReady` messages with references to the generated images

## Architecture

The Image Server Service follows the Experimance star topology pattern:
- Subscribes to the events channel for RenderRequest messages
- Publishes ImageReady messages to the images channel
- Uses a plugin-based generator architecture

## ZMQ Communication

### Addresses and Ports

The service uses the unified events channel for all communication:

- **Events Channel (Unified)**
  - Server subscribes to: `tcp://localhost:5555` (events)
  - Server publishes to: `tcp://*:5555` (events)
  - All message types use the unified channel with message type filtering

### Message Types

- **RenderRequest**: Sent by clients to request an image
  ```json
  {
    "type": "RenderRequest",
    "request_id": "uuid-string", 
    "era": "wilderness", 
    "biome": "forest",
    "prompt": "A forest scene with tall trees",
    "depth_map_png": "optional-base64-encoded-png"
  }
  ```

- **ImageReady**: Published by server when an image is ready
  ```json
  {
    "type": "ImageReady",
    "request_id": "uuid-string",
    "image_id": "uuid-string",
    "uri": "file:///path/to/image.png", 
    "width": 512,
    "height": 512
  }
  ```

## CLI Tool

The CLI utility allows testing the Image Server Service interactively or via command line:

```bash
# Interactive mode
./src/image_server/cli.py -i

# Command line mode
./src/image_server/cli.py --prompt "A forest scene" --era wilderness --biome forest
```

## Testing ZMQ Communication

The service comes with several test utilities to verify ZeroMQ communication:

### Test Runner

The main test runner executes all ZMQ communication tests:

```bash
# Run all tests
./tests/run_zmq_tests.py

# Run specific test
./tests/run_zmq_tests.py --test basic
```

Available test options:
- `config`: Validates ZMQ address configuration
- `basic`: Basic ZMQ message test
- `render`: Comprehensive render request test
- `cli`: Tests CLI functionality
- `all`: Runs all tests (default)

### Individual Test Scripts

You can also run individual test scripts:

```bash
# Validate ZMQ addressing
./tests/validate_zmq_addresses.py

# Test basic ZMQ messaging
./tests/test_zmq_messaging.py

# Test render request messaging
./tests/test_zmq_render_request.py
```

## Troubleshooting

If you encounter communication issues:

1. Verify the image server service is running
2. Check the address configuration with `./tests/validate_zmq_addresses.py`
3. Ensure there are no port conflicts
4. Look for ZMQ connection errors in the logs
5. Run the test suite to diagnose specific issues

Common issues:
- Incorrect port numbers in client code
- Using incorrect address format (`tcp://*:PORT` for binding vs `tcp://localhost:PORT` for connecting) 
- ZMQ slow joiner syndrome (fixed by adding delays after socket creation)
- Mismatched subscription topics (use empty string "" to receive all messages)
