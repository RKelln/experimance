"""
Health management system for Experimance services.

This module provides a unified health checking and notification system that replaces
the old heartbeat system with comprehensive health monitoring and smart notifications.

Health data is written to files for inter-process communication, with a dedicated
health service monitoring all services by reading these files.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

class HealthStatus(str, Enum):
    """Standardized health status across all services and monitoring systems."""
    HEALTHY = "healthy"      # Service is operating normally
    WARNING = "warning"      # Service has issues but is still functional
    ERROR = "error"         # Service has significant issues affecting functionality
    FATAL = "fatal"         # Service is non-functional and requires intervention
    UNKNOWN = "unknown"     # Health status cannot be determined

@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class ServiceHealth:
    """Overall service health status."""
    service_name: str
    overall_status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    uptime: Optional[float] = None
    restart_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "service_name": self.service_name,
            "overall_status": self.overall_status.value,
            "checks": [check.to_dict() for check in self.checks],
            "last_updated": self.last_updated.isoformat(),
            "uptime": self.uptime,
            "restart_count": self.restart_count,
            "error_count": self.error_count
        }
    
    def add_check(self, check: HealthCheck):
        """Add a health check and update overall status."""
        self.checks.append(check)
        self.last_updated = datetime.now()
        
        # Keep only last 50 checks to prevent memory growth
        if len(self.checks) > 50:
            self.checks = self.checks[-50:]
        
        # Calculate overall status based on all checks
        self._calculate_overall_status()
    
    def _calculate_overall_status(self):
        """Calculate overall status based on individual checks."""
        if not self.checks:
            self.overall_status = HealthStatus.UNKNOWN
            return
            
        statuses = [check.status for check in self.checks]
        
        # Priority order: FATAL > ERROR > WARNING > HEALTHY > UNKNOWN
        if HealthStatus.FATAL in statuses:
            self.overall_status = HealthStatus.FATAL
        elif HealthStatus.ERROR in statuses:
            self.overall_status = HealthStatus.ERROR
        elif HealthStatus.WARNING in statuses:
            self.overall_status = HealthStatus.WARNING
        elif HealthStatus.HEALTHY in statuses:
            self.overall_status = HealthStatus.HEALTHY
        else:
            self.overall_status = HealthStatus.UNKNOWN

class HealthReporter:
    """Centralized health reporting and notification system."""
    
    def __init__(self, service_name: str, service_type: str, health_dir: Optional[str] = None):
        self.service_name = service_name
        self.service_type = service_type
        self.health = ServiceHealth(service_name, HealthStatus.HEALTHY)
        self.start_time = time.time()
        self.notification_handlers: List[Callable[[ServiceHealth, bool], None]] = []
        self.last_notification_time = {}
        self.notification_cooldown = 300  # 5 minutes
        
        # Health file location
        if health_dir is None:
            # Use environment-aware default
            if os.getenv("EXPERIMANCE_ENV") == "production":
                health_dir = "/var/cache/experimance/health"
            else:
                health_dir = f"{os.getcwd()}/logs/health"
        
        self.health_dir = Path(health_dir)
        self.health_dir.mkdir(parents=True, exist_ok=True)
        # FIXME: currently only 1 service per type allowed, TODO: handle names per type
        # Use service_type for health file name for consistent monitoring
        self.health_file = self.health_dir / f"{service_type}.json"
        
    def add_notification_handler(self, handler):
        """Add a notification handler for health status changes."""
        # Wrap the handler to call the correct method with flush parameter
        def wrapped_handler(service_health, flush=False):
            if hasattr(handler, 'send_notification'):
                handler.send_notification(service_health, flush=flush)
            else:
                handler(service_health, flush=flush)
        
        self.notification_handlers.append(wrapped_handler)
    
    def record_health_check(self, name: str, status: HealthStatus, 
                          message: Optional[str] = None, 
                          metadata: Optional[Dict[str, Any]] = None,
                          flush: bool = False):
        """Record a health check result."""
        check = HealthCheck(
            name=name,
            status=status,
            message=message,
            metadata=metadata or {}
        )
        
        previous_status = self.health.overall_status
        self.health.add_check(check)
        self.health.uptime = time.time() - self.start_time
        
        # Log the health check
        level = logging.DEBUG
        if status in [HealthStatus.ERROR, HealthStatus.FATAL]:
            level = logging.ERROR
        elif status == HealthStatus.WARNING:
            level = logging.WARNING
            
        logger.log(level, f"Health check '{name}': {status.value} - {message}")
        
        # Write health status to file
        self._write_health_file()
        
        # Send notifications if status changed or is critical
        if (previous_status != self.health.overall_status or 
            status in [HealthStatus.ERROR, HealthStatus.FATAL]):
            self._send_notifications(flush=flush)
    
    def record_error(self, error: Exception, is_fatal: bool = False):
        """Record an error and update health status."""
        self.health.error_count += 1
        
        status = HealthStatus.FATAL if is_fatal else HealthStatus.ERROR
        self.record_health_check(
            name="service_error",
            status=status,
            message=str(error),
            metadata={
                "error_type": type(error).__name__,
                "is_fatal": is_fatal,
                "error_count": self.health.error_count
            }
        )
    
    def record_restart(self):
        """Record a service restart."""
        self.health.restart_count += 1
        self.record_health_check(
            name="service_restart",
            status=HealthStatus.WARNING,
            message=f"Service restarted (count: {self.health.restart_count})"
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get a summary of current health status."""
        health_data = self.health.to_dict()
        
        # Add service statistics if available
        if hasattr(self, '_service_stats'):
            health_data.update(self._service_stats)
        
        return health_data
    
    def update_service_stats(self, messages_sent: int, messages_received: int, errors: int):
        """Update service statistics for health reporting."""
        self._service_stats = {
            "messages_sent": messages_sent,
            "messages_received": messages_received,
            "error_count": errors  # This will override the health object's error_count
        }
    
    def _write_health_file(self):
        """Write current health status to file for monitoring."""
        try:
            health_data = self.get_health_summary()
            health_data["last_check"] = datetime.now().isoformat()
            
            # Write atomically by writing to temp file first
            temp_file = self.health_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(health_data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(self.health_file)
            
        except Exception as e:
            logger.error(f"Error writing health file {self.health_file}: {e}")
    
    def _send_notifications(self, flush: bool = False):
        """Send notifications to all registered handlers."""
        now = datetime.now()
        status_key = self.health.overall_status.value
        
        # Check cooldown
        if status_key in self.last_notification_time:
            last_time = self.last_notification_time[status_key]
            if (now - last_time).total_seconds() < self.notification_cooldown:
                return
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(self.health, flush)
            except Exception as e:
                logger.error(f"Error sending health notification: {e}")
        
        self.last_notification_time[status_key] = now


def create_health_reporter(service_name: str, service_type: str, health_dir: Optional[str] = None) -> HealthReporter:
    """Create a health reporter with appropriate notification handlers."""
    reporter = HealthReporter(service_name, service_type, health_dir)
    
    # Add notification handlers based on environment
    from .notifications import create_notification_handlers
    handlers = create_notification_handlers()
    
    for handler in handlers:
        reporter.add_notification_handler(handler)
    
    return reporter
