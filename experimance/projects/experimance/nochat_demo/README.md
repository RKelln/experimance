# Experimance NoChat Demo Variant

Simplified Experimance installation for quick demos in noisy environments.

## Services

- **core**: Sand interaction and state machine (unchanged)
- **display**: Projected satellite imagery (unchanged)
- **image_server**: Vast.ai-powered image generation
- **audio**: Environmental audio only (no music)
- **health**: Service monitoring

## No Agent

This variant **excludes the speech-to-speech chatbot** service. This is ideal for:

- Noisy venues where speech recognition won't work reliably
- Quick setup without needing speech/audio infrastructure
- Focusing on the core sand interaction → image generation loop
- Demo-only installations

## Configuration Differences

| Service | Difference |
|---------|-----------|
| **audio** | `music_volume = 0.0`, stereo output, `auto_start_jack = false` |
| **image_server** | `strategy = "vastai"` (cloud generation) |
| **health** | Monitors only core/display/audio/image_server (no agent) |
| **core** | Same as base project |
| **display** | Same as base project |

## Setup

With the variant support system, you have two approaches:

**Recommended: Use the variant system**
```bash
# Set the variant once
uv run scripts/set_project.py experimance/nochat_demo

# Then all services automatically use nochat_demo configs
uv run -m experimance_audio
uv run -m image_server
uv run -m experimance_health
uv run -m experimance_core
uv run -m experimance_display
```

**Alternative: Pass config paths explicitly**
```bash
uv run -m experimance_audio   --config projects/experimance/nochat_demo/audio.toml
uv run -m image_server        --config projects/experimance/nochat_demo/image_server.toml
uv run -m experimance_health  --config projects/experimance/nochat_demo/health.toml
uv run -m experimance_core    --config projects/experimance/nochat_demo/core.toml
uv run -m experimance_display --config projects/experimance/nochat_demo/display.toml
```

## Customization

Adjust these settings in the variant configs:

- **audio.toml**: device name, master_volume, environment_volume for your venue
- **image_server.toml**: vastai model settings, timeout
- **core.toml**: camera resolution, interaction thresholds for your sand table
- **display.toml**: for your projector setup

See the base project configs in `projects/experimance/` for more detailed options.
