# Experimance Health Service

A standalone service that monitors the health of all other services in the Experimance installation.

## Overview

The health service monitors other services by:
1. Reading health status files written by each service
2. Detecting stale or missing health data
3. Sending notifications when services become unhealthy
4. Providing a centralized view of system health

## Configuration

The service is configured via the main config file under the `health_service` section:

```toml
[health_service]
service_name = "health"
health_dir = "/var/cache/experimance/health"
check_interval = 30  # seconds
service_timeout = 120  # seconds
notification_cooldown = 300  # seconds
expected_services = [
    "core",
    "display", 
    "agent",
    "audio",
    "image_server"
]
```

## Running

The health service is designed to run as a systemd service:

```bash
# Start the health service
systemctl start health@experimance

# Check status
systemctl status health@experimance

# View logs
journalctl -u health@experimance -f
```

## Health File Format

Services write their health status to JSON files in the health directory:

```json
{
    "service_name": "core",
    "overall_status": "healthy",
    "checks": [
        {
            "name": "periodic_check",
            "status": "healthy",
            "message": "Service is responsive",
            "timestamp": "2025-11-18T14:30:00.123456",
            "metadata": {}
        }
    ],
    "last_updated": "2025-11-18T14:30:00.123456",
    "uptime": 1234.5,
    "restart_count": 0,
    "error_count": 0
}
```

## Notifications

The health service sends notifications when:
- Services become unhealthy (error, fatal)
- Services go missing (unknown)
- Services show warnings (with cooldown)

Notifications are sent via the same system as other services (ntfy, logs, etc.).
