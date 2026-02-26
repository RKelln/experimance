# test_reolink_camera.py

Python client for Reolink IP cameras via their HTTP API. Used for presence detection testing and camera control during development of the `ReolinkDetector` service.

See `scripts/test_reolink_camera.py`.

## Quick Start

```bash
# Continuous presence monitoring (person/vehicle/pet)
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD

# Check camera status and capabilities
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD --status

# Explore all supported API commands
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD --explore
```

## Camera Control (Stealth Mode)

Turn off all visible indicators so the camera is less obtrusive:

```bash
# Disable all LEDs and IR (stealth mode)
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD --camera-off

# Restore all defaults
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD --camera-on

# Individual controls
uv run python scripts/test_reolink_camera.py ... --ir-lights off
uv run python scripts/test_reolink_camera.py ... --ir-lights on
uv run python scripts/test_reolink_camera.py ... --power-led off
uv run python scripts/test_reolink_camera.py ... --power-led on
```

## Debug Mode

```bash
uv run python scripts/test_reolink_camera.py \
    --host 192.168.2.229 --user admin --password YOUR_PASSWORD --debug
```

Shows raw AI detection state data alongside the formatted presence summary.

## All Flags

| Flag | Description |
|---|---|
| `--host IP` | Camera IP address (required) |
| `--user NAME` | Camera username (default: `admin`) |
| `--password PASS` | Camera password (required) |
| `--http` | Use HTTP instead of HTTPS |
| `--poll-interval N` | Seconds between presence polls (default: 1.0) |
| `--status` | Print camera status and exit |
| `--explore` | Probe all known API commands and print results |
| `--camera-off` | Enable stealth mode (IR off, LEDs off) |
| `--camera-on` | Disable stealth mode (restore IR and LEDs) |
| `--ir-lights on\|off` | Control IR night-vision LEDs |
| `--power-led on\|off` | Control power/status LED |
| `--debug` | Show raw AI detection data |

## Camera Setup

1. Enable AI detection in the camera's web UI (Person / Vehicle / Pet detection)
2. Create a dedicated user with minimal permissions (or use admin)
3. Note the camera's LAN IP address
4. SSL warnings are suppressed automatically (Reolink uses self-signed certs)

## Security Notes

- Use a dedicated camera user with read-only permissions when possible
- Prefer HTTPS (default) even with self-signed certificates
- Do not expose the camera HTTP API to untrusted networks

## Tested Hardware

- Reolink RLC-820A, firmware v3.1.0.2368_23062508

## Requirements

- `requests`, `urllib3` (not included in the standard experimance venv — install if missing)
- Network access to the camera on the local LAN

## Relation to Production Code

This script serves as the reference implementation for `ReolinkDetector` in `services/agent/src/agent/vision/`. When testing new camera features or debugging detection issues, run this script directly rather than going through the full agent stack.
