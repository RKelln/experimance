# dev

Multi-service development launcher. Starts one or more Experimance services with proper environment setup, log routing, and graceful shutdown.

See `scripts/dev`.

## Quick Start

```bash
# Start a single service in the foreground
./scripts/dev display
./scripts/dev core
./scripts/dev image        # shortcut for image_server

# Start multiple services (background, with logging)
./scripts/dev health core

# Start all services
./scripts/dev all

# Start everything except the agent
./scripts/dev no_agent

# Use a different project environment
PROJECT_ENV=fire ./scripts/dev all
```

## Services

Available services are discovered dynamically from `services/` (via `infra/scripts/get_project_services.py`):

| Name | Module (default) | Module (fire) |
|---|---|---|
| `core` | `experimance_core` | `fire_core` |
| `agent` | `experimance_agent` | `fire_agent` |
| `display` | `experimance_display` | `experimance_display` |
| `audio` | `experimance_audio` | `experimance_audio` |
| `health` | `experimance_health` | `experimance_health` |
| `image` / `image_server` | `image_server` | `image_server` |
| `transition` | `experimance_transition` | `experimance_transition` |

`PROJECT_ENV` (default: `experimance`) controls which project's core and agent modules are loaded.

## Single vs Multi-Service Mode

**Single service** — runs in the foreground, output goes directly to the terminal. Ctrl+C stops it cleanly.

**Multiple services** — each service runs as a background subprocess. Output is prefixed with `[service_name]` and tee'd to `logs/dev/<service>.log`. Ctrl+C triggers graceful shutdown of all services (SIGINT → 15s wait → SIGKILL).

```bash
# Tail logs for a specific service while running multi-service mode
tail -f logs/dev/core.log
tail -f logs/dev/agent.log
```

## Display Service Handling

The display service needs access to an X11/Wayland display. `dev` detects the environment automatically:

- **Local session with display** — uses the existing `$DISPLAY` and `$XAUTHORITY`
- **Xwayland** — detects `:0` and the mutter auth socket automatically
- **Remote (SSH / VS Code Remote)** — reads display environment from the running GNOME shell process, then launches via `systemd-run --user --scope` so the process has proper desktop session context
- **No display found** — sets `EXPERIMANCE_DISPLAY_HEADLESS=true` as fallback

## Shutdown Behavior

On Ctrl+C (or SIGTERM) the script:
1. Sends SIGINT to all service Python processes (graceful shutdown)
2. Waits up to 15 seconds for services to exit cleanly
3. Force-kills (SIGKILL) any stubborn processes

Pre-existing instances of the same services are cleaned up before starting new ones.

## Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `PROJECT_ENV` | `experimance` | Selects project module variants (e.g. `fire_core`) |
| `EXPERIMANCE_ENV` | `development` | Set by the script automatically |
| `NOTIFICATIONS_DRY_RUN` | `true` | Disables real notifications during dev |
| `EXPERIMANCE_DISPLAY_HEADLESS` | (unset) | Set to `true` if no display is found |

## Log Files

Multi-service logs are written to `logs/dev/<service>.log` (falls back to `./dev-logs/` if that directory can't be created).
