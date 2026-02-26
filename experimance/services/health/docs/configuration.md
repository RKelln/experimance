# Health Service Configuration

Configuration is loaded from `services/health/config.toml` by default. Pass a different path with
`--config <path>` at runtime. Command-line flags override file values.

## Quick Reference

```toml
# services/health/config.toml

service_name = "experimance_health"

# --- Monitoring ---
check_interval          = 30.0    # seconds between health checks
startup_grace_period    = 10.0    # seconds before checking other services after startup
service_timeout         = 120.0   # seconds before a stale health file becomes ERROR
max_health_file_age     = 86400.0 # seconds before a health file is deleted (24 h)
cleanup_interval        = 3600.0  # seconds between cleanup runs (1 h)

# --- Health file locations ---
health_dir            = "/var/cache/experimance/health"  # resolved at runtime
dev_health_dir        = "logs/health"
production_health_dir = "/var/cache/experimance/health"

# --- Services to monitor ---
expected_services = [
    "experimance_core",
    "experimance_display",
    "experimance_agent",
    "experimance_audio",
    "image_server",
]

# --- Notifications ---
enable_notifications      = true
notification_level        = "warning"  # "error" | "warning" | "info"
notification_on_startup   = true
notification_on_shutdown  = true
notify_on_healthy         = false
notify_on_unknown         = true
notification_cooldown     = 300.0  # seconds between repeated same-status alerts

# --- Buffering ---
enable_buffering = true
buffer_time      = 10.0  # seconds to collect before sending

# --- ntfy push notifications (optional) ---
# ntfy_topic  = "my-topic"
# ntfy_server = "ntfy.sh"

# --- Webhook notifications (optional) ---
# webhook_url         = "https://example.com/hook"
# webhook_auth_header = "Bearer <token>"
```

## Key Details

### Environment-Aware Health Directory

`get_effective_health_dir()` selects the health directory based on environment:

- **Production** (any of: running as root, `EXPERIMANCE_ENV=production`, or `/etc/experimance`
  exists): uses `production_health_dir`
- **Development** (all other cases): uses `dev_health_dir`

The `health_dir` key is the fallback when neither override is set.

### `notification_level`

Controls the minimum severity that triggers a notification:

| Level | Statuses that trigger notifications |
|---|---|
| `"error"` | `ERROR`, `FATAL` |
| `"warning"` | `WARNING`, `ERROR`, `FATAL` |
| `"info"` | All statuses |

`WARNING` status does **not** trigger notifications by default (user preference) even at
`notification_level = "warning"`. Set `notification_level = "info"` to include warnings.

### `notify_on_healthy` and `notify_on_unknown`

These independently gate notifications when a service transitions to `HEALTHY` or `UNKNOWN`.
Both respect `notification_cooldown`.

### `startup_grace_period`

Note: the default in `config.py` is 60 s, but `config.toml` sets it to 10 s. The file value
wins at runtime. Adjust this to match how long your slowest service takes to start.

### `expected_services`

The list of service type strings the health service watches. Use the same identifiers that
services report in their health files (`service_name` field). Current defaults match the names
used in `experimance_common.SERVICE_TYPES`.
