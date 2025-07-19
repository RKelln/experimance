#!/usr/bin/env python3
"""
Experimance Health Service

A standalone service that monitors the health of all other services
in the Experimance installation via file-based health monitoring.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from experimance_common.base_service import BaseService
from experimance_common.config import load_config_with_overrides
from experimance_common.health import HealthStatus
from experimance_common.notifications import create_notification_handlers
from .config import HealthServiceConfig, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)


class HealthService(BaseService):
    """
    Health monitoring service that tracks the health of all other services
    by reading their health status files.
    """
    
    def __init__(self, config: HealthServiceConfig):
        """Initialize the health service."""
        # Initialize BaseService first
        super().__init__(service_name=config.service_name)
        
        self.config = config
        self.health_dir = config.get_effective_health_dir()
        self.last_notifications = {}  # Track last notification time per service
        self.notification_handlers = []
        self.startup_time = datetime.now()
        
        # Initialize notification handlers
        self._initialize_notification_handlers()
    
    
    def _initialize_notification_handlers(self):
        """Initialize notification handlers based on configuration."""
        if self.config.enable_notifications:
            # Create notification handlers using configuration instead of env vars
            from experimance_common.notifications import create_notification_handlers_from_config
            
            # Check for dry-run mode from environment (this is OK as it's a debug/test setting)
            import os
            dry_run = os.environ.get("NOTIFICATIONS_DRY_RUN", "false").lower() == "true"
            
            # Use configuration for notification settings, with env var fallbacks
            ntfy_topic = self.config.ntfy_topic or os.environ.get("NTFY_TOPIC")
            ntfy_server = self.config.ntfy_server or os.environ.get("NTFY_SERVER", "ntfy.sh")
            webhook_url = self.config.webhook_url or os.environ.get("HEALTH_WEBHOOK_URL")
            
            # Build webhook headers
            webhook_headers = {}
            if self.config.webhook_auth_header:
                webhook_headers["Authorization"] = self.config.webhook_auth_header
            elif os.environ.get("HEALTH_WEBHOOK_AUTH"):
                webhook_headers["Authorization"] = os.environ.get("HEALTH_WEBHOOK_AUTH")
            
            # Create handlers with configuration
            health_log_path = self.health_dir / "health.log"
            self.notification_handlers = create_notification_handlers_from_config(
                log_file=str(health_log_path),
                ntfy_topic=ntfy_topic,
                ntfy_server=ntfy_server,
                webhook_url=webhook_url,
                webhook_headers=webhook_headers,
                enable_buffering=self.config.enable_buffering,
                buffer_time=self.config.buffer_time,
                dry_run=dry_run
            )
        else:
            logger.info("Notifications disabled by configuration")
    
    async def start(self):
        """Start the health monitoring service."""
        logger.info(f"Starting health service monitoring {len(self.config.expected_services)} services")
        logger.info(f"Health directory: {self.health_dir}")
        
        # Send startup notification
        if self.config.notification_on_startup:
            await self._send_startup_notification()
        
        # Add health monitoring task
        self.add_task(self._health_monitoring_loop())
        
        # Add cleanup task
        self.add_task(self._cleanup_loop())
        
        await super().start()
    
    async def stop(self):
        """Stop the health monitoring service."""
        logger.info("Stopping health service")
        
        # Flush any buffered notifications immediately
        self._flush_notification_buffers()
        
        # Send shutdown notification
        if self.config.notification_on_shutdown:
            await self._send_shutdown_notification()
        
        # Flush again after shutdown notification
        self._flush_notification_buffers()
        
        await super().stop()
    
    def _flush_notification_buffers(self):
        """Flush all buffered notification handlers immediately."""
        for handler in self.notification_handlers:
            # Check if this is a BufferedNotificationHandler
            if hasattr(handler, 'flush_immediately') and callable(getattr(handler, 'flush_immediately')):
                try:
                    handler.flush_immediately()  # type: ignore
                except Exception as e:
                    logger.error(f"Error flushing notification buffer: {e}")
    
    async def _send_startup_notification(self):
        """Send notification that health service is starting."""
        for handler in self.notification_handlers:
            try:
                from experimance_common.health import ServiceHealth, HealthCheck
                mock_health = ServiceHealth(
                    service_name=self.config.service_name,
                    overall_status=HealthStatus.HEALTHY
                )
                
                check = HealthCheck(
                    name="startup",
                    status=HealthStatus.HEALTHY,
                    message=f"Health service started, monitoring {len(self.config.expected_services)} services"
                )
                mock_health.add_check(check)
                
                handler.send_notification(mock_health)
            except Exception as e:
                logger.error(f"Error sending startup notification: {e}")
    
    async def _send_shutdown_notification(self):
        """Send notification that health service is shutting down."""
        for handler in self.notification_handlers:
            try:
                from experimance_common.health import ServiceHealth, HealthCheck
                mock_health = ServiceHealth(
                    service_name=self.config.service_name,
                    overall_status=HealthStatus.WARNING
                )
                
                check = HealthCheck(
                    name="shutdown",
                    status=HealthStatus.WARNING,
                    message="Health service is shutting down"
                )
                mock_health.add_check(check)
                
                handler.send_notification(mock_health)
            except Exception as e:
                logger.error(f"Error sending shutdown notification: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old health files periodically."""
        while self.running:
            try:
                await self._cleanup_old_health_files()
                await asyncio.sleep(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)  # Short delay on error
    
    async def _cleanup_old_health_files(self):
        """Remove old health files."""
        current_time = datetime.now()
        max_age = timedelta(seconds=self.config.max_health_file_age)
        
        for health_file in self.health_dir.glob("*.json"):
            try:
                file_age = datetime.fromtimestamp(health_file.stat().st_mtime)
                if current_time - file_age > max_age:
                    health_file.unlink()
                    logger.info(f"Cleaned up old health file: {health_file}")
            except Exception as e:
                logger.warning(f"Error cleaning up {health_file}: {e}")
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(5)  # Short delay on error
    
    async def _check_all_services(self):
        """Check the health of all expected services."""
        current_time = datetime.now()
        
        # Check if we're still in startup grace period
        time_since_startup = (current_time - self.startup_time).total_seconds()
        if time_since_startup < self.config.startup_grace_period:
            logger.debug(f"Still in startup grace period ({time_since_startup:.1f}s/{self.config.startup_grace_period}s), skipping health checks")
            return
        
        overall_healthy = True
        service_statuses = {}
        
        for service_name in self.config.expected_services:
            status = await self._check_service_health(service_name, current_time)
            service_statuses[service_name] = status
            
            # Check if service is unhealthy (case-insensitive comparison)
            if status["status"].upper() not in ["HEALTHY", "WARNING"]:
                overall_healthy = False
        
        # Log overall system health
        if overall_healthy:
            logger.debug("All services are healthy")
        else:
            unhealthy_services = [
                name for name, status in service_statuses.items()
                if status["status"].upper() not in ["HEALTHY", "WARNING"]
            ]
            logger.warning(f"Unhealthy services: {unhealthy_services}")
        
        # Send notifications if needed
        await self._send_notifications(service_statuses, overall_healthy)
    
    async def _check_service_health(self, service_name: str, current_time: datetime) -> Dict:
        """Check the health of a single service."""
        health_file = self.health_dir / f"{service_name}.json"
        
        try:
            if not health_file.exists():
                return {
                    "status": "UNKNOWN",
                    "message": f"Health file not found: {health_file}",
                    "last_check": None,
                    "uptime": 0
                }
            
            # Read health file
            with open(health_file, 'r') as f:
                health_data = json.load(f)
            
            # Check if health data is stale
            last_check = datetime.fromisoformat(health_data.get("last_check", ""))
            time_since_check = (current_time - last_check).total_seconds()
            
            if time_since_check > self.config.service_timeout:
                return {
                    "status": "ERROR",
                    "message": f"Service health data is stale ({time_since_check:.1f}s old)",
                    "last_check": health_data.get("last_check"),
                    "uptime": health_data.get("uptime", 0)
                }
            
            # Return the service's reported health
            return {
                "status": health_data.get("overall_status", "UNKNOWN"),
                "message": health_data.get("message", ""),
                "last_check": health_data.get("last_check"),
                "uptime": health_data.get("uptime", 0),
                "error_count": health_data.get("error_count", 0),
                "restart_count": health_data.get("restart_count", 0)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error reading health file: {e}",
                "last_check": None,
                "uptime": 0
            }
    
    async def _send_notifications(self, service_statuses: Dict, overall_healthy: bool):
        """Send notifications for health status changes."""
        notifications_sent = 0
        
        for service_name, status in service_statuses.items():
            service_status = status["status"]
            
            # Check if we should send a notification
            if self._should_notify(service_name, service_status):
                notifications_sent += 1
                
                # Send to all notification handlers
                for handler in self.notification_handlers:
                    try:
                        # Create a mock ServiceHealth object for the notification
                        from experimance_common.health import ServiceHealth, HealthCheck
                        
                        # Convert string status to HealthStatus enum
                        try:
                            health_status = HealthStatus(service_status.lower())
                        except ValueError:
                            # If status string doesn't match enum, default to UNKNOWN
                            health_status = HealthStatus.UNKNOWN
                        
                        mock_health = ServiceHealth(
                            service_name=service_name,
                            overall_status=health_status
                        )
                        mock_health.uptime = status.get("uptime", 0)
                        mock_health.error_count = status.get("error_count", 0)
                        mock_health.restart_count = status.get("restart_count", 0)
                        
                        # Add a health check with the message
                        message = status.get("message", "")
                        if message:
                            check = HealthCheck(
                                name="health_service_check",
                                status=health_status,
                                message=message
                            )
                            mock_health.add_check(check)
                        
                        handler.send_notification(mock_health)
                    except Exception as e:
                        logger.error(f"Error sending notification for {service_name}: {e}")
                
                # Update last notification time
                self.last_notifications[service_name] = datetime.now()
        
        if notifications_sent > 0:
            logger.info(f"Sent {notifications_sent} health notifications")
        
        # If using buffered notifications, also send system summary
        if self.config.enable_buffering and service_statuses:
            await self._send_system_summary(service_statuses, overall_healthy)
    
    async def _send_system_summary(self, service_statuses: Dict, overall_healthy: bool):
        """Send a system-wide health summary for buffered notifications."""
        # Only send summary if there are issues or if configured to notify on healthy
        if overall_healthy and not self.config.notify_on_healthy:
            return
        
        # Find any buffered notification handlers and send system summary
        for handler in self.notification_handlers:
            if hasattr(handler, 'send_system_notification'):
                try:
                    # Convert raw status data to ServiceHealth objects
                    from experimance_common.health import ServiceHealth, HealthCheck
                    
                    service_health_objects = {}
                    for service_name, status in service_statuses.items():
                        # Convert string status to HealthStatus enum
                        try:
                            health_status = HealthStatus(status["status"].lower())
                        except ValueError:
                            health_status = HealthStatus.UNKNOWN
                        
                        service_health = ServiceHealth(
                            service_name=service_name,
                            overall_status=health_status
                        )
                        service_health.uptime = status.get("uptime", 0)
                        service_health.error_count = status.get("error_count", 0)
                        service_health.restart_count = status.get("restart_count", 0)
                        
                        # Add health check with message if available
                        message = status.get("message", "")
                        if message:
                            check = HealthCheck(
                                name="health_service_check",
                                status=health_status,
                                message=message
                            )
                            service_health.add_check(check)
                        
                        service_health_objects[service_name] = service_health
                    
                    handler.send_system_notification(service_health_objects)
                except Exception as e:
                    logger.error(f"Error sending system summary: {e}")
    
    def _should_notify(self, service_name: str, status: str) -> bool:
        """Check if we should send a notification for this service."""
        # Check if notifications are enabled
        if not self.config.enable_notifications:
            return False
        
        # Normalize status to uppercase for comparison
        status_upper = status.upper()
        
        # Apply notification level filtering
        if self.config.notification_level == "error":
            # Only notify for ERROR and FATAL
            if status_upper not in ["ERROR", "FATAL"]:
                return False
        elif self.config.notification_level == "warning":
            # Notify for WARNING, ERROR, and FATAL
            if status_upper not in ["WARNING", "ERROR", "FATAL"]:
                return False
        # For "info" level, notify for all statuses (handled below)
        
        # Always notify for ERROR and FATAL (immediate problems)
        if status_upper in ["ERROR", "FATAL"]:
            return True
        
        # Check configuration for HEALTHY notifications
        if status_upper == "HEALTHY":
            return self.config.notify_on_healthy
        
        # Check configuration for UNKNOWN notifications
        if status_upper == "UNKNOWN":
            if not self.config.notify_on_unknown:
                return False
        
        # For WARNING and UNKNOWN, only notify if we haven't notified recently
        # This prevents spam while still alerting about ongoing issues
        if status_upper in ["WARNING", "UNKNOWN"]:
            if service_name in self.last_notifications:
                time_since_last = (datetime.now() - self.last_notifications[service_name]).total_seconds()
                # Only notify if cooldown has passed
                return time_since_last >= self.config.notification_cooldown
            else:
                # First time seeing this status, notify
                return True
        
        # Default: don't notify for unknown statuses
        return False
    
    def _format_notification_message(self, service_name: str, status: Dict) -> str:
        """Format a notification message for a service."""
        service_status = status["status"]
        message = status.get("message", "")
        uptime = status.get("uptime", 0)
        
        base_msg = f"Service {service_name}: {service_status}"
        if message:
            base_msg += f" - {message}"
        
        if uptime > 0:
            base_msg += f" (uptime: {uptime:.1f}s)"
        
        return base_msg

async def run_health_service(
    config_path: str | Path = DEFAULT_CONFIG_PATH, 
    args:Optional[argparse.Namespace] = None
):
    """
    Run the Experimance Core Service.
    
    Args:
        config_path: Path to configuration file
        args: CLI arguments from argparse (if using new CLI system)
    """
    # Create config with CLI overrides
    config = HealthServiceConfig.from_overrides(
        config_file=config_path,
        args=args
    )
    
    service = HealthService(config=config)
    
    await service.start()
    await service.run()