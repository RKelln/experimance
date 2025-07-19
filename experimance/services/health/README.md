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
service_name = "experimance_health"
health_dir = "/var/cache/experimance/health"
check_interval = 30  # seconds
service_timeout = 120  # seconds
notification_cooldown = 300  # seconds
expected_services = [
    "experimance_core",
    "experimance_display", 
    "experimance_agent",
    "experimance_audio",
    "image_server"
]
```

## Running

The health service is designed to run as a systemd service:

```bash
# Start the health service
systemctl start experimance-health@experimance

# Check status
systemctl status experimance-health@experimance

# View logs
journalctl -u experimance-health@experimance -f
```

## Health File Format

Services write their health status to JSON files in the health directory:

```json
{
    "service_name": "experimance_core",
    "overall_status": "HEALTHY",
    "message": "Service is running normally",
    "last_check": "2025-07-18T14:30:00.000000",
    "uptime": 1234.5,
    "error_count": 0,
    "restart_count": 0,
    "checks": [
        {
            "name": "periodic_check",
            "status": "HEALTHY",
            "message": "Service is responsive",
            "timestamp": "2025-07-18T14:30:00.000000"
        }
    ]
}
```

## Notifications

The health service sends notifications when:
- Services become unhealthy (ERROR, FATAL)
- Services go missing (UNKNOWN)
- Services show warnings (with cooldown)

Notifications are sent via the same system as other services (ntfy, logs, etc.).
