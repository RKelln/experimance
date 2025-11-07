# Image Server Tests

This directory contains test tools for the Image Server service.

## Direct Audio Generation Testing

The `test_audio_direct.py` script provides direct testing of the TangoFlux audio generator without going through the ZMQ service layer. This is useful for:

- Debugging audio generation issues
- Testing audio generator configuration
- Development and experimentation

### Usage

```bash
# Interactive mode
uv run python services/image_server/tests/test_audio_direct.py -i

# Command line with custom prompt
uv run python services/image_server/tests/test_audio_direct.py --prompt "gentle rain" --duration 5

# Use a sample prompt
uv run python services/image_server/tests/test_audio_direct.py --sample-prompt forest_ambience

# List available sample prompts
uv run python services/image_server/tests/test_audio_direct.py --list-prompts
```

### Prerequisites

Make sure you have the audio generation dependencies installed:

```bash
uv sync --package image-server --extra audio_gen
```

### Output

Audio files are saved to `/tmp/experimance_audio_direct_test/` with normalized volume and proper file naming.

## ZMQ Testing

For testing the Image Server service over ZMQ (including both image and audio generation), use the main CLI:

```bash
# Image generation
uv run -m image_server.cli -i

# Audio generation via ZMQ (when AudioRenderRequest schema is available)
uv run -m image_server.cli --audio --audio-prompt "gentle rain"
```
