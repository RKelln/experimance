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

from experimance_common.logger import setup_logging

SERVICE_TYPE = "health"

logger = setup_logging(__name__, log_filename=f"{SERVICE_TYPE}.log")


class HealthService(BaseService):
    """
    Health monitoring service that tracks the health of all other services
    by reading their health status files.
    """
    
    def __init__(self, config: HealthServiceConfig):
        """Initialize the health service."""
        # Initialize BaseService first
        super().__init__(service_name=config.service_name, service_type=SERVICE_TYPE)

        self.config = config
        self.health_dir = config.get_effective_health_dir()
        self.last_notifications = {}  # Track last notification time per service
        self.last_service_status = {}  # Track last known status per service
        self.notification_handlers = []
        self.startup_time = datetime.now()
        self.initial_system_notification_sent = False  # Track if we've sent the post-grace-period notification
        
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
        
        # Add main health monitoring task
        self.add_task(self._health_monitoring_loop())
        
        # Add cleanup task
        self.add_task(self._cleanup_loop())
        
        # Add stats collection task
        self.add_task(self._stats_collection_loop())
        
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
            # All handlers now have built-in flush capability
            if hasattr(handler, 'flush_immediately') and callable(getattr(handler, 'flush_immediately')):
                try:
                    handler.flush_immediately()
                except Exception as e:
                    logger.error(f"Error flushing notification buffer for {handler.name}: {e}")
    
    async def _send_startup_notification(self):
        """Send notification that health service is starting."""
        # Only send startup notification if configured to do so and if we notify on healthy
        if not self.config.notification_on_startup:
            return
        
        # For healthy startup notifications, respect the notify_on_healthy setting
        if not self.config.notify_on_healthy:
            logger.debug("Skipping startup notification - notify_on_healthy is False")
            return
            
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
        """Send system notification that health service is shutting down."""
        logger.info("Sending system shutdown notification")
        
        # Get current system state before shutdown
        service_statuses = {}
        for service_type in self.config.expected_services:
            status = await self._collect_service_stats(service_type, datetime.now())
            service_statuses[service_type] = {
                "status": status.get("status", "UNKNOWN"),
                "message": status.get("message", ""),
                "uptime": 0,  # We don't have uptime in stats format
                "error_count": status.get("errors", 0),
                "restart_count": 0
            }
        
        # Convert to ServiceHealth objects
        from experimance_common.health import ServiceHealth, HealthCheck
        
        service_health_objects = {}
        for service_type, status in service_statuses.items():
            try:
                health_status = HealthStatus(status["status"].lower())
            except ValueError:
                health_status = HealthStatus.UNKNOWN
            
            service_health = ServiceHealth(
                service_name=service_type,  # Using service_type as service_name for consistency
                overall_status=health_status
            )
            service_health.uptime = status.get("uptime", 0)
            service_health.error_count = status.get("error_count", 0)
            service_health.restart_count = status.get("restart_count", 0)
            
            service_health_objects[service_type] = service_health
        
        # Add the health service itself as shutting down
        health_service_health = ServiceHealth(
            service_name=self.config.service_name,
            overall_status=HealthStatus.WARNING
        )
        
        check = HealthCheck(
            name="shutdown",
            status=HealthStatus.WARNING,
            message="Health service is shutting down"
        )
        health_service_health.add_check(check)
        service_health_objects[self.config.service_name] = health_service_health
        
        # Send system shutdown notification to all handlers
        for handler in self.notification_handlers:
            try:
                handler.send_system_notification(service_health_objects)
            except Exception as e:
                logger.error(f"Error sending shutdown system notification to {handler.name}: {e}")
    
    async def _cleanup_loop(self):
        """Cleanup old health files periodically."""
        while self.running:
            try:
                await self._cleanup_old_health_files()
                await self._sleep_if_running(self.config.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await self._sleep_if_running(60)
    
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
    
    async def _stats_collection_loop(self):
        """Collect and display global service statistics."""
        while self.running:
            try:
                await self._collect_and_display_stats()
                await self._sleep_if_running(10)  # Display stats every 10 seconds, same as BaseService
            except Exception as e:
                logger.error(f"Error in stats collection loop: {e}")
                await self._sleep_if_running(10)
    
    async def _collect_and_display_stats(self):
        """Collect statistics from all services and display global summary."""
        current_time = datetime.now()
        overall_stats = {
            "timestamp": current_time.isoformat(),
            "services": {},
            "global_summary": {
                "total_services": len(self.config.expected_services),
                "healthy_services": 0,
                "warning_services": 0,
                "error_services": 0,
                "unknown_services": 0,
                "total_messages_sent": 0,
                "total_messages_received": 0,
                "total_errors": 0
            }
        }
        
        for service_type in self.config.expected_services:
            stats = await self._collect_service_stats(service_type, current_time)
            overall_stats["services"][service_type] = stats
            
            # Update global summary
            status = stats.get("status", "UNKNOWN").upper()
            if status == "HEALTHY":
                overall_stats["global_summary"]["healthy_services"] += 1
            elif status == "WARNING":
                overall_stats["global_summary"]["warning_services"] += 1
            elif status == "ERROR":
                overall_stats["global_summary"]["error_services"] += 1
            else:
                overall_stats["global_summary"]["unknown_services"] += 1
            
            # Aggregate message counts and errors
            overall_stats["global_summary"]["total_messages_sent"] += stats.get("messages_sent", 0)
            overall_stats["global_summary"]["total_messages_received"] += stats.get("messages_received", 0)
            overall_stats["global_summary"]["total_errors"] += stats.get("errors", 0)
        
        # Log the global summary in a clean format
        summary = overall_stats["global_summary"]
        healthy_count = summary["healthy_services"]
        total_count = summary["total_services"]
        
        logger.info(f"Global Stats: {healthy_count}/{total_count} services healthy | "
                   f"Messages: {summary['total_messages_sent']} sent, {summary['total_messages_received']} received | "
                   f"Total errors: {summary['total_errors']}")
        
        # Log individual service stats at debug level to reduce noise
        for service_name, stats in overall_stats["services"].items():
            if stats.get("status") != "UNKNOWN":  # Only log services that are reporting
                logger.debug(f"{service_name}: {stats.get('status')} | "
                           f"Uptime: {stats.get('uptime', 'N/A')} | "
                           f"Msgs: {stats.get('messages_sent', 0)}↑/{stats.get('messages_received', 0)}↓ | "
                           f"Errors: {stats.get('errors', 0)}")
    
    async def _collect_service_stats(self, service_type: str, current_time: datetime) -> dict:
        """Collect statistics for a single service type from its health file."""
        health_file = self.health_dir / f"{service_type}.json"
        
        try:
            if not health_file.exists():
                return {
                    "status": "UNKNOWN",
                    "message": "Health file not found",
                    "uptime": "N/A",
                    "messages_sent": 0,
                    "messages_received": 0,
                    "errors": 0
                }
            
            with open(health_file, 'r') as f:
                health_data = json.load(f)
            
            # Check if health data is fresh
            last_check_str = health_data.get("last_check")
            if last_check_str:
                last_check = datetime.fromisoformat(last_check_str.replace('Z', '+00:00'))
                time_since_check = (current_time - last_check.replace(tzinfo=None)).total_seconds()
                
                # During startup grace period, don't mark stale data as error
                time_since_startup = (current_time - self.startup_time).total_seconds()
                in_grace_period = time_since_startup < self.config.startup_grace_period
                
                if time_since_check > self.config.service_timeout and not in_grace_period:
                    return {
                        "status": "ERROR",
                        "message": f"Service health data is stale ({time_since_check:.1f}s old)",
                        "uptime": health_data.get("uptime", "N/A"),
                        "messages_sent": health_data.get("messages_sent", 0),
                        "messages_received": health_data.get("messages_received", 0),
                        "errors": health_data.get("error_count", 0)
                    }
                elif time_since_check > self.config.service_timeout and in_grace_period:
                    # During grace period, just mark as unknown instead of error
                    return {
                        "status": "UNKNOWN",
                        "message": f"Stale data from previous run ({time_since_check:.1f}s old, in grace period)",
                        "uptime": health_data.get("uptime", "N/A"),
                        "messages_sent": health_data.get("messages_sent", 0),
                        "messages_received": health_data.get("messages_received", 0),
                        "errors": health_data.get("error_count", 0)
                    }
            
            # Extract stats from health data
            uptime_seconds = health_data.get("uptime", 0)
            if isinstance(uptime_seconds, (int, float)) and uptime_seconds > 0:
                hours, remainder = divmod(int(uptime_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                uptime_str = f"{hours:02}:{minutes:02}:{seconds:02}"
            else:
                uptime_str = "N/A"
            
            return {
                "status": health_data.get("overall_status", "UNKNOWN"),
                "message": health_data.get("message", ""),
                "uptime": uptime_str,
                "messages_sent": health_data.get("messages_sent", 0),
                "messages_received": health_data.get("messages_received", 0),
                "errors": health_data.get("error_count", 0)
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Error reading stats: {e}",
                "uptime": "N/A",
                "messages_sent": 0,
                "messages_received": 0,
                "errors": 0
            }
    
    async def _health_monitoring_loop(self):
        """Main health monitoring loop."""
        while self.running:
            try:
                await self._check_all_services()
                await self._sleep_if_running(self.config.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await self._sleep_if_running(5)  # Short delay on error
    
    async def _check_all_services(self):
        """Check the health of all expected services."""
        current_time = datetime.now()
        
        # Debug logging to track duplicate calls
        logger.debug(f"_check_all_services called at {current_time.isoformat()}")
        
        # Check if we're still in startup grace period
        time_since_startup = (current_time - self.startup_time).total_seconds()
        if time_since_startup < self.config.startup_grace_period:
            logger.debug(f"Still in startup grace period ({time_since_startup:.1f}s/{self.config.startup_grace_period}s), skipping health checks")
            return
        
        overall_healthy = True
        service_statuses = {}
        
        for service_type in self.config.expected_services:
            status = await self._check_service_health(service_type, current_time)
            service_statuses[service_type] = status
            
            # Check if service is unhealthy (case-insensitive comparison)
            if status["status"].upper() not in ["HEALTHY", "WARNING"]:
                overall_healthy = False
        
        # Send initial system notification after grace period (once)
        if not self.initial_system_notification_sent:
            await self._send_initial_system_notification(service_statuses, overall_healthy)
            self.initial_system_notification_sent = True
        
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
    
    async def _check_service_health(self, service_type: str, current_time: datetime) -> Dict:
        """Check the health of a single service type."""
        health_file = self.health_dir / f"{service_type}.json"
        
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
                logger.debug(f"Sending notification for {service_name}: {service_status}")
                
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
            else:
                logger.debug(f"Skipping notification for {service_name}: {service_status} (no change or cooldown active)")
        
        if notifications_sent > 0:
            logger.info(f"Sent {notifications_sent} health notifications")
        
        # Only send system notifications for significant events, not routine health checks
        # Individual service notifications (buffered) handle the routine monitoring
    
    async def _send_initial_system_notification(self, service_statuses: Dict, overall_healthy: bool):
        """Send initial system notification after startup grace period."""
        logger.info("Sending initial system health notification after startup grace period")
        
        # Convert raw status data to ServiceHealth objects
        from experimance_common.health import ServiceHealth, HealthCheck
        
        service_health_objects = {}
        for service_name, status in service_statuses.items():
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
                    name="initial_system_check",
                    status=health_status,
                    message=message
                )
                service_health.add_check(check)
            
            service_health_objects[service_name] = service_health
        
        # Send to all handlers - this is a significant system event
        for handler in self.notification_handlers:
            try:
                handler.send_system_notification(service_health_objects)
            except Exception as e:
                logger.error(f"Error sending initial system notification to {handler.name}: {e}")
    
    async def _send_system_summary(self, service_statuses: Dict, overall_healthy: bool):
        """Send a system-wide health summary."""
        # Only send summary if there are issues or if configured to notify on healthy
        if overall_healthy and not self.config.notify_on_healthy:
            return
        
        # Convert raw status data to ServiceHealth objects once
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
        
        # Send system summary only to the log handler (non-buffered immediate logging)
        # Other handlers will get system notifications when their buffers flush
        for handler in self.notification_handlers:
            # Only send immediate system notifications to handlers that don't buffer
            if not handler.enable_buffering:
                try:
                    handler.send_system_notification(service_health_objects)
                except Exception as e:
                    logger.error(f"Error sending system summary to {handler.name}: {e}")
    
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
        
        # Check if this is a status change
        last_status = self.last_service_status.get(service_name)
        status_changed = last_status != status_upper
        
        # Update tracked status
        self.last_service_status[service_name] = status_upper
        
        # Always notify on status changes (but still respect other filters)
        if status_changed:
            logger.debug(f"Status change detected for {service_name}: {last_status} -> {status_upper}")
            
            # For HEALTHY notifications, check configuration
            if status_upper == "HEALTHY":
                return self.config.notify_on_healthy
            
            # For UNKNOWN notifications, check configuration
            if status_upper == "UNKNOWN":
                return self.config.notify_on_unknown
            
            # For WARNING - don't notify, just log (user preference)
            if status_upper == "WARNING":
                return False
            
            # For ERROR, FATAL - always notify on status change
            if status_upper in ["ERROR", "FATAL"]:
                return True
        
        # If status hasn't changed, apply cooldown logic for ongoing issues
        if service_name in self.last_notifications:
            time_since_last = (datetime.now() - self.last_notifications[service_name]).total_seconds()
            
            # Different cooldown for different severity levels
            if status_upper in ["ERROR", "FATAL"]:
                # For critical errors, use a longer cooldown to reduce spam
                # but still remind periodically
                cooldown_period = max(self.config.notification_cooldown, 300)  # At least 5 minutes
            elif status_upper in ["UNKNOWN"]:
                # Use configured cooldown for unknown (WARNING excluded per user preference)
                cooldown_period = self.config.notification_cooldown
            else:
                # For healthy status, use configured cooldown
                cooldown_period = self.config.notification_cooldown
            
            # Only notify if cooldown has passed
            if time_since_last >= cooldown_period:
                # Apply same configuration filters as for status changes
                if status_upper == "HEALTHY":
                    return self.config.notify_on_healthy
                elif status_upper == "UNKNOWN":
                    return self.config.notify_on_unknown
                elif status_upper == "WARNING":
                    return False  # Don't notify on WARNING per user preference
                elif status_upper in ["ERROR", "FATAL"]:
                    return True
        
        # Default: don't notify
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