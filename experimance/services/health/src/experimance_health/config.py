#!/usr/bin/env python3
"""
Configuration schema for the Experimance Health Service.

This module defines Pydantic models for validating and accessing
health service configuration in a type-safe way.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from pydantic import Field

from experimance_common.config import BaseConfig
from experimance_common.constants import (
    DEFAULT_PORTS, ZMQ_TCP_BIND_PREFIX, ZMQ_TCP_CONNECT_PREFIX,
    get_project_config_path, SERVICE_TYPES
)

logger = logging.getLogger(__name__)

# Define the default configuration path with project-aware fallback
# TODO: Fix HEALTH_SERVICE_DIR import or define it locally
from experimance_common.constants import SERVICES_DIR
HEALTH_SERVICE_DIR = SERVICES_DIR / "health"
DEFAULT_CONFIG_PATH = get_project_config_path("health", HEALTH_SERVICE_DIR)


class HealthServiceConfig(BaseConfig):
    """Health service configuration."""
    
    service_name: str = "health"

    health_dir: Path = Field(
        default=Path("/var/cache/experimance/health"),
        description="Directory where service health files are stored"
    )
    
    check_interval: float = Field(
        default=30.0,
        description="Interval between health checks in seconds"
    )
    
    startup_grace_period: float = Field(
        default=120.0,
        description="Time to wait after health service startup before checking other services (seconds)"
    )
    
    service_timeout: float = Field(
        default=120.0,
        description="Maximum age of health data before considering service stale (seconds)"
    )
    
    notification_cooldown: float = Field(
        default=300.0,
        description="Minimum time between notifications for the same service (seconds)"
    )
    
    startup_grace_period: float = Field(
        default=60.0,
        description="Time to wait after health service startup before checking other services (seconds)"
    )
    
    expected_services: List[str] = Field(
        default_factory=lambda: SERVICE_TYPES,
        description="List of service types expected to be monitored"
    )
    
    # Environment-specific overrides
    dev_health_dir: Optional[Path] = Field(
        default=None,
        description="Health directory for development environment (overrides health_dir if set)"
    )
    
    production_health_dir: Optional[Path] = Field(
        default=None,
        description="Health directory for production environment (overrides health_dir if set)"
    )
    
    # Notification settings
    enable_notifications: bool = Field(
        default=True,
        description="Enable health notifications"
    )
    
    notification_level: str = Field(
        default="warning",
        description="Minimum notification level: 'error' (ERROR/FATAL only), 'warning' (WARNING+), 'info' (all statuses)"
    )
    
    notification_on_startup: bool = Field(
        default=True,
        description="Send notification when health service starts"
    )
    
    notification_on_shutdown: bool = Field(
        default=True,
        description="Send notification when health service shuts down"
    )
    
    # Suppress notifications for healthy services
    notify_on_healthy: bool = Field(
        default=False,
        description="Send notifications when services become healthy (usually not needed)"
    )
    
    # Suppress notifications for unknown services (missing health files)
    notify_on_unknown: bool = Field(
        default=True,
        description="Send notifications for unknown service status (missing health files)"
    )
    
    # Notification buffering
    buffer_time: float = Field(
        default=10.0,
        description="Time to buffer notifications before sending (seconds)"
    )
    
    enable_buffering: bool = Field(
        default=True,
        description="Enable notification buffering to reduce spam"
    )
    
    # External notification services
    ntfy_topic: Optional[str] = Field(
        default=None,
        description="ntfy.sh topic for push notifications"
    )
    
    ntfy_server: str = Field(
        default="ntfy.sh",
        description="ntfy server hostname or URL"
    )
    
    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for health notifications"
    )
    
    webhook_auth_header: Optional[str] = Field(
        default=None,
        description="Authorization header for webhook requests"
    )
    
    # Advanced settings
    max_health_file_age: float = Field(
        default=86400.0,  # 24 hours
        description="Maximum age of health files before they are cleaned up (seconds)"
    )
    
    cleanup_interval: float = Field(
        default=3600.0,  # 1 hour
        description="Interval between cleanup tasks (seconds)"
    )
    
    def get_effective_health_dir(self) -> Path:
        """Get the effective health directory based on environment."""
        import os
        from pathlib import Path
        
        # Use the same production detection logic as the logger
        is_production = (
            os.geteuid() == 0 or  # Running as root
            os.environ.get("EXPERIMANCE_ENV") == "production" or
            Path("/etc/experimance").exists()  # Production marker
        )
        
        # Check for environment-specific overrides
        if is_production and self.production_health_dir:
            return self.production_health_dir
        elif not is_production and self.dev_health_dir:
            return self.dev_health_dir
        else:
            # Use default based on environment detection
            if is_production:
                return Path("/var/cache/experimance/health")
            else:
                return Path("logs/health")
    
    def get_expected_services_for_project(self, project_name: str) -> List[str]:
        """Get expected service types for a specific project."""
        # Use the dynamic project-aware service detection if available
        try:
            from experimance_common.config import get_project_services
            return get_project_services(project_name)
        except Exception as e:
            # Fallback to default configured service types
            logger.warning(f"Failed to get project services for {project_name}, using default: {e}")
            return self.expected_services
