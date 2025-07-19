"""
Notification handlers for health status changes.

This module provides notification handlers that can be used with the unified health system
to send alerts via various channels (ntfy, email, webhooks, etc.).
"""

import asyncio
import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Timer

import aiohttp
import requests

from experimance_common.health import HealthStatus, ServiceHealth

logger = logging.getLogger(__name__)

class NotificationHandler:
    """Base class for notification handlers."""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        
    def send_notification(self, service_health: ServiceHealth):
        """Send a notification for a service health change."""
        raise NotImplementedError("Subclasses must implement send_notification")
        
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Send a notification for system-wide health changes."""
        raise NotImplementedError("Subclasses must implement send_system_notification")


class BufferedNotificationHandler(NotificationHandler):
    """
    A wrapper that buffers notifications for a configurable time period
    and sends aggregated summaries instead of individual notifications.
    """
    
    def __init__(self, wrapped_handler: NotificationHandler, buffer_time: float = 10.0):
        super().__init__(f"buffered_{wrapped_handler.name}")
        self.wrapped_handler = wrapped_handler
        self.buffer_time = buffer_time
        self.buffer: Dict[str, ServiceHealth] = {}
        self.buffer_timer: Optional[Timer] = None
        
    def send_notification(self, service_health: ServiceHealth):
        """Buffer individual service notifications."""
        # Add to buffer
        self.buffer[service_health.service_name] = service_health
        
        # Reset/start timer
        if self.buffer_timer:
            self.buffer_timer.cancel()
        
        self.buffer_timer = Timer(self.buffer_time, self._flush_buffer)
        self.buffer_timer.start()
    
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Forward system notifications immediately (these are already aggregated)."""
        self.wrapped_handler.send_system_notification(services)
    
    def _flush_buffer(self):
        """Send aggregated notification for all buffered services."""
        if not self.buffer:
            return
        
        try:
            # Create aggregated notification
            if len(self.buffer) == 1:
                # Single service - send as individual notification
                service_health = next(iter(self.buffer.values()))
                self.wrapped_handler.send_notification(service_health)
            else:
                # Multiple services - send as system notification
                self.wrapped_handler.send_system_notification(self.buffer.copy())
            
            # Clear buffer
            self.buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing notification buffer: {e}")
            self.buffer.clear()
    
    def flush_immediately(self):
        """Force immediate flush of the buffer."""
        if self.buffer_timer:
            self.buffer_timer.cancel()
        self._flush_buffer()

class NtfyHandler(NotificationHandler):
    """Send notifications via ntfy.sh service."""
    
    def __init__(self, topic: str, server: str = "ntfy.sh", priority: str = "default", dryrun: bool = False):
        """ Initialize the ntfy handler. """
        super().__init__("ntfy")
        self.topic = topic
        self.server = server
        self.priority = priority
        # Fix URL format - ensure proper protocol
        if not server.startswith(('http://', 'https://')):
            server = f"https://{server}"
        self.url = f"{server}/{topic}"
        self.dryrun = dryrun
        
    def send_notification(self, service_health: ServiceHealth):
        """Send ntfy notification for service health change."""
        try:
            # Map health status to ntfy priority
            priority_map = {
                HealthStatus.HEALTHY: "low",
                HealthStatus.WARNING: "default", 
                HealthStatus.ERROR: "high",
                HealthStatus.FATAL: "urgent",
                HealthStatus.UNKNOWN: "low"
            }
            
            # Create message
            title = f"🔧 {service_health.service_name} Health Alert"
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.ERROR: "❌",
                HealthStatus.FATAL: "🚨",
                HealthStatus.UNKNOWN: "❓"
            }
            
            message = f"{status_emoji[service_health.overall_status]} Status: {service_health.overall_status.value.upper()}"
            
            # Add recent checks
            if service_health.checks:
                recent_checks = service_health.checks[-3:]  # Last 3 checks
                message += "\n\nRecent checks:"
                for check in recent_checks:
                    message += f"\n• {check.name}: {check.status.value}"
                    if check.message:
                        message += f" - {check.message}"
            
            # Add service info
            if service_health.uptime:
                uptime_hours = service_health.uptime / 3600
                message += f"\n\nUptime: {uptime_hours:.1f}h"
            
            if service_health.error_count > 0:
                message += f"\nErrors: {service_health.error_count}"
                
            if service_health.restart_count > 0:
                message += f"\nRestarts: {service_health.restart_count}"
            
            # Send notification
            status_value = service_health.overall_status.value
            
            # Create comprehensive tags for filtering
            tags = [
                "experimance",
                "health",
                status_value,
                service_health.service_name,
                f"priority-{priority_map[service_health.overall_status]}"
            ]
            
            # Add severity tags for easy filtering
            if service_health.overall_status in [HealthStatus.ERROR, HealthStatus.FATAL]:
                tags.append("critical")
            elif service_health.overall_status == HealthStatus.WARNING:
                tags.append("warning")
            elif service_health.overall_status == HealthStatus.HEALTHY:
                tags.append("info")
            
            data = {
                "title": title,
                "message": message,
                "priority": priority_map[service_health.overall_status],
                "tags": tags
            }
            
            if self.dryrun:
                logger.debug(f"[DRY RUN] Would send ntfy notification: {json.dumps(data, indent=2)}")
            else:
                response = requests.post(self.url, json=data, timeout=10)
                response.raise_for_status()
            
            logger.info(f"Sent ntfy notification for {service_health.service_name} ({service_health.overall_status})")
            
        except Exception as e:
            logger.error(f"Failed to send ntfy notification: {e}")
            
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Send ntfy notification for system-wide health changes."""
        try:
            # Calculate overall system status
            all_statuses = [svc.overall_status for svc in services.values()]
            
            if HealthStatus.FATAL in all_statuses:
                overall_status = HealthStatus.FATAL
            elif HealthStatus.ERROR in all_statuses:
                overall_status = HealthStatus.ERROR
            elif HealthStatus.WARNING in all_statuses:
                overall_status = HealthStatus.WARNING
            elif HealthStatus.HEALTHY in all_statuses:
                overall_status = HealthStatus.HEALTHY
            else:
                overall_status = HealthStatus.UNKNOWN
            
            # Create message
            title = f"🏗️ Experimance System Health Update"
            status_emoji = {
                HealthStatus.HEALTHY: "✅",
                HealthStatus.WARNING: "⚠️",
                HealthStatus.ERROR: "❌",
                HealthStatus.FATAL: "🚨",
                HealthStatus.UNKNOWN: "❓"
            }
            
            message = f"{status_emoji[overall_status]} System Status: {overall_status.value.upper()}"
            
            # Group services by status
            status_groups = defaultdict(list)
            for svc in services.values():
                status_groups[svc.overall_status].append(svc)
            
            # Add summary by status
            message += f"\n\n📊 Service Summary ({len(services)} total):"
            for status in [HealthStatus.FATAL, HealthStatus.ERROR, HealthStatus.WARNING, HealthStatus.HEALTHY, HealthStatus.UNKNOWN]:
                if status in status_groups:
                    count = len(status_groups[status])
                    emoji = status_emoji[status]
                    message += f"\n{emoji} {status.value.title()}: {count}"
            
            # Add details for problematic services
            critical_services = [
                svc for svc in services.values() 
                if svc.overall_status in [HealthStatus.ERROR, HealthStatus.FATAL]
            ]
            
            if critical_services:
                message += f"\n\n🔴 Critical Issues ({len(critical_services)}):"
                for svc in critical_services[:5]:  # Limit to first 5
                    message += f"\n• {svc.service_name}: {svc.overall_status.value}"
                    if svc.checks:
                        latest_check = svc.checks[-1]
                        if latest_check.message:
                            # Truncate long messages
                            msg = latest_check.message[:50] + "..." if len(latest_check.message) > 50 else latest_check.message
                            message += f" - {msg}"
                
                if len(critical_services) > 5:
                    message += f"\n... and {len(critical_services) - 5} more"
            
            # Add warning services if no critical ones
            elif status_groups.get(HealthStatus.WARNING):
                warning_services = status_groups[HealthStatus.WARNING]
                message += f"\n\n🟡 Warnings ({len(warning_services)}):"
                for svc in warning_services[:3]:  # Limit to first 3
                    message += f"\n• {svc.service_name}"
                    if svc.checks:
                        latest_check = svc.checks[-1]
                        if latest_check.message:
                            msg = latest_check.message[:30] + "..." if len(latest_check.message) > 30 else latest_check.message
                            message += f" - {msg}"
                
                if len(warning_services) > 3:
                    message += f"\n... and {len(warning_services) - 3} more"
            
            # Send notification
            priority_map = {
                HealthStatus.HEALTHY: "low",
                HealthStatus.WARNING: "default",
                HealthStatus.ERROR: "high", 
                HealthStatus.FATAL: "urgent",
                HealthStatus.UNKNOWN: "low"
            }
            
            # Create comprehensive tags
            tags = [
                "experimance",
                "system",
                "health",
                overall_status.value,
                f"priority-{priority_map[overall_status]}",
                f"services-{len(services)}"
            ]
            
            # Add severity tags
            if critical_services:
                tags.append("critical")
                tags.append(f"critical-{len(critical_services)}")
            elif status_groups.get(HealthStatus.WARNING):
                tags.append("warning")
                tags.append(f"warnings-{len(status_groups[HealthStatus.WARNING])}")
            else:
                tags.append("info")
            
            data = {
                "title": title,
                "message": message,
                "priority": priority_map[overall_status],
                "tags": tags
            }
            
            if self.dryrun:
                logger.info(f"[DRY RUN] Would send system ntfy notification: {json.dumps(data, indent=2)}")
            else:
                response = requests.post(self.url, json=data, timeout=10)
                response.raise_for_status()
            
            logger.info(f"Sent system ntfy notification ({overall_status})")
            
        except Exception as e:
            logger.error(f"Failed to send system ntfy notification: {e}")

class LogHandler(NotificationHandler):
    """Log notifications to file."""
    
    def __init__(self, log_file: Optional[str] = None):
        super().__init__("log")
        self.log_file = log_file
        
    def send_notification(self, service_health: ServiceHealth):
        """Log service health notification."""
        message = f"[{datetime.now().isoformat()}] Service {service_health.service_name} health: {service_health.overall_status.value}"
        
        if service_health.checks:
            latest_check = service_health.checks[-1]
            message += f" - Latest: {latest_check.name} ({latest_check.status.value})"
            if latest_check.message:
                message += f": {latest_check.message}"
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                logger.error(f"Failed to write health log: {e}")
        
        logger.info(message)
        
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Log system health notification."""
        message = f"[{datetime.now().isoformat()}] System health update: {len(services)} services"
        
        status_counts = {}
        for svc in services.values():
            status_counts[svc.overall_status] = status_counts.get(svc.overall_status, 0) + 1
            
        for status, count in status_counts.items():
            message += f" | {status.value}: {count}"
        
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                logger.error(f"Failed to write system health log: {e}")
        
        logger.info(message)

class WebhookHandler(NotificationHandler):
    """Send notifications via webhook."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None, dryrun: bool = False):
        super().__init__("webhook")
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.dryrun = dryrun
        
    def send_notification(self, service_health: ServiceHealth):
        """Send webhook notification for service health change."""
        try:
            payload = {
                "type": "service_health",
                "timestamp": datetime.now().isoformat(),
                "service": service_health.to_dict()
            }
            
            if not self.dryrun:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
            
            logger.info(f"Sent webhook notification for {service_health.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Send webhook notification for system-wide health changes."""
        try:
            payload = {
                "type": "system_health",
                "timestamp": datetime.now().isoformat(),
                "services": {name: svc.to_dict() for name, svc in services.items()}
            }
            
            if not self.dryrun:
                response = requests.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
            
            logger.info("Sent system webhook notification")
            
        except Exception as e:
            logger.error(f"Failed to send system webhook notification: {e}")

def create_notification_handlers() -> List[NotificationHandler]:
    """Create notification handlers based on environment configuration."""
    handlers = []
    
    # Check for dry-run mode
    dry_run = os.environ.get("NOTIFICATIONS_DRY_RUN", "false").lower() == "true"
    
    # Check for buffering configuration
    buffer_time = float(os.environ.get("NOTIFICATION_BUFFER_TIME", "10.0"))
    enable_buffering = os.environ.get("NOTIFICATION_BUFFERING", "true").lower() == "true"
    
    # ntfy handler
    ntfy_topic = os.environ.get("NTFY_TOPIC")
    if ntfy_topic:
        ntfy_server = os.environ.get("NTFY_SERVER", "ntfy.sh")
        ntfy_handler = NtfyHandler(ntfy_topic, ntfy_server, dryrun=dry_run)
        
        # Wrap with buffered handler if enabled
        if enable_buffering:
            ntfy_handler = BufferedNotificationHandler(ntfy_handler, buffer_time)
        
        handlers.append(ntfy_handler)
    
    # Log handler (always include, even in dry-run)
    log_file = os.environ.get("HEALTH_LOG_FILE")
    log_handler = LogHandler(log_file)
    
    # Don't buffer log handlers - we want immediate logging
    handlers.append(log_handler)
    
    # Webhook handler
    webhook_url = os.environ.get("HEALTH_WEBHOOK_URL")
    if webhook_url:
        webhook_headers = {}
        if os.environ.get("HEALTH_WEBHOOK_AUTH"):
            webhook_headers["Authorization"] = os.environ.get("HEALTH_WEBHOOK_AUTH")
        webhook_handler = WebhookHandler(webhook_url, webhook_headers, dryrun=dry_run)
        
        # Wrap with buffered handler if enabled
        if enable_buffering:
            webhook_handler = BufferedNotificationHandler(webhook_handler, buffer_time)
        
        handlers.append(webhook_handler)
    
    if dry_run:
        logger.info("Notifications in dry-run mode - only logging enabled")
    
    if enable_buffering:
        logger.info(f"Notification buffering enabled with {buffer_time}s buffer time")
    
    return handlers


def create_notification_handlers_from_config(
    log_file: Optional[str] = None,
    ntfy_topic: Optional[str] = None,
    ntfy_server: str = "ntfy.sh",
    webhook_url: Optional[str] = None,
    webhook_headers: Optional[Dict[str, str]] = None,
    enable_buffering: bool = True,
    buffer_time: float = 10.0,
    dry_run: bool = False
) -> List[NotificationHandler]:
    """Create notification handlers based on direct configuration parameters."""
    handlers = []
    
    # Log handler (always include)
    log_handler = LogHandler(log_file)
    handlers.append(log_handler)
    
    # ntfy handler
    if ntfy_topic:
        ntfy_handler = NtfyHandler(ntfy_topic, ntfy_server, dryrun=dry_run)
        handlers.append(ntfy_handler)
    
    # Webhook handler
    if webhook_url:
        webhook_handler = WebhookHandler(webhook_url, webhook_headers or {}, dryrun=dry_run)
        handlers.append(webhook_handler)
    
    # Wrap with buffered handlers if enabled
    if enable_buffering:
        buffered_handlers = []
        for handler in handlers:
            # Don't buffer log handlers - we want immediate logging
            if isinstance(handler, LogHandler):
                buffered_handlers.append(handler)
            else:
                buffered_handlers.append(
                    BufferedNotificationHandler(handler, buffer_time)
                )
        handlers = buffered_handlers
        logger.info(f"Initialized {len(handlers)} notification handlers with {buffer_time}s buffering")
    else:
        logger.info(f"Initialized {len(handlers)} notification handlers")
    
    if dry_run:
        logger.info("Notifications in dry-run mode - only logging enabled")
    
    return handlers
