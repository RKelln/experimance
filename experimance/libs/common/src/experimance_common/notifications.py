"""
Notification handlers for health status changes.

This module provides notification handlers that can be used with the unified health system
to send alerts via various channels (ntfy, email, webhooks, etc.).

Design principles:
- System notifications are always sent immediately (no buffering)
- Individual service notifications can be buffered to reduce spam
- Each handler manages its own buffering internally
- Simple base class with optional buffering capability

Test using:
NOTIFICATIONS_DRY_RUN=true NTFY_TOPIC=test uv run ...
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
    """
    Base class for notification handlers with optional built-in buffering.
    
    System notifications are always sent immediately.
    Service notifications can be buffered based on enable_buffering parameter.
    """
    
    def __init__(self, name: str, enable_buffering: bool = False, buffer_time: float = 10.0):
        self.name = name
        self.enabled = True
        self.enable_buffering = enable_buffering
        self.buffer_time = buffer_time
        
        # Buffering state
        self.service_buffer: Dict[str, ServiceHealth] = {}
        self.buffer_timer: Optional[Timer] = None
        
    def send_notification(self, service_health: ServiceHealth, flush: bool = False):
        """Send a notification for a service health change."""
        if self.enable_buffering:
            self._buffer_service_notification(service_health)
            if flush:
                self.flush_immediately()
        else:
            self._send_service_notification(service_health)
    
    def send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Send a notification for system-wide health changes (always immediate)."""
        self._send_system_notification(services)
    
    def _buffer_service_notification(self, service_health: ServiceHealth):
        """Buffer a service notification for later sending."""
        # Add to buffer
        self.service_buffer[service_health.service_name] = service_health
        
        # Reset/start timer
        if self.buffer_timer:
            self.buffer_timer.cancel()
        
        self.buffer_timer = Timer(self.buffer_time, self._flush_service_buffer)
        self.buffer_timer.start()
    
    def _flush_service_buffer(self):
        """Send all buffered service notifications."""
        if not self.service_buffer:
            return
        
        try:
            if len(self.service_buffer) == 1:
                # Single service - send as individual notification
                service_health = next(iter(self.service_buffer.values()))
                self._send_service_notification(service_health)
            else:
                # Multiple services - send as aggregated system notification
                self._send_system_notification(self.service_buffer.copy())
            
            # Clear buffer
            self.service_buffer.clear()
            
        except Exception as e:
            logger.error(f"Error flushing {self.name} notification buffer: {e}")
            self.service_buffer.clear()
    
    def flush_immediately(self):
        """Force immediate flush of any buffered notifications."""
        if self.buffer_timer:
            self.buffer_timer.cancel()
        self._flush_service_buffer()
    
    def _send_service_notification(self, service_health: ServiceHealth):
        """Actually send a service notification - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _send_service_notification")
        
    def _send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Actually send a system notification - implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _send_system_notification")



class NtfyHandler(NotificationHandler):
    """Send notifications via ntfy.sh service."""
    
    def __init__(self, topic: str, server: str = "ntfy.sh", priority: str = "default", 
                 dryrun: bool = False, enable_buffering: bool = False, buffer_time: float = 10.0):
        """Initialize the ntfy handler."""
        super().__init__("ntfy", enable_buffering, buffer_time)
        self.topic = topic
        self.server = server
        self.priority = priority
        # Fix URL format - ensure proper protocol
        if not server.startswith(('http://', 'https://')):
            server = f"https://{server}"
        self.url = f"{server}/{topic}"
        self.dryrun = dryrun
        
    def _send_service_notification(self, service_health: ServiceHealth):
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
            
            self._send_ntfy_message(title, message, service_health.overall_status, 
                                  service_health.service_name, priority_map)
            
        except Exception as e:
            logger.error(f"Failed to send ntfy service notification: {e}")
            
    def _send_system_notification(self, services: Dict[str, ServiceHealth]):
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
            
            # Add service health timeline for all services
            message += f"\n\n📊 Service Health Timeline:"
            for service_name, svc in sorted(services.items()):
                # Get recent health states from checks (last 5-10 checks)
                health_states = []
                if svc.checks:
                    # Sort checks by timestamp to ensure chronological order
                    sorted_checks = sorted(svc.checks, key=lambda x: x.timestamp)
                    
                    # Extract unique health states in chronological order
                    for check in sorted_checks[-10:]:  # Last 10 checks
                        if check.status not in health_states or check.status != health_states[-1]:
                            health_states.append(check.status)
                
                # If no checks or only one state, show current status
                if not health_states:
                    health_states = [svc.overall_status]
                elif len(health_states) == 1 and health_states[0] != svc.overall_status:
                    health_states.append(svc.overall_status)
                
                # Format the timeline with arrows showing progression
                timeline_str = " → ".join([state.value.upper() for state in health_states])
                
                # Add service line with current status emoji
                current_emoji = status_emoji.get(svc.overall_status, "❓")
                message += f"\n{current_emoji} {service_name}: [{timeline_str}]"
            
            # Send system notification with appropriate tags
            priority_map = {
                HealthStatus.HEALTHY: "low",
                HealthStatus.WARNING: "default",
                HealthStatus.ERROR: "high", 
                HealthStatus.FATAL: "urgent",
                HealthStatus.UNKNOWN: "low"
            }
            
            self._send_ntfy_message(title, message, overall_status, "system", priority_map)
            
        except Exception as e:
            logger.error(f"Failed to send system ntfy notification: {e}")
    
    def _send_ntfy_message(self, title: str, message: str, status: HealthStatus, 
                          source: str, priority_map: dict):
        """Common method to send ntfy messages."""
        # Create comprehensive tags for filtering
        tags = [
            "experimance",
            "health",
            status.value,
            source,
            f"priority-{priority_map[status]}"
        ]
        
        # Add severity tags for easy filtering
        if status in [HealthStatus.ERROR, HealthStatus.FATAL]:
            tags.append("critical")
        elif status == HealthStatus.WARNING:
            tags.append("warning")
        elif status == HealthStatus.HEALTHY:
            tags.append("info")
        
        # Add source-specific tags
        if source == "system":
            tags.append("system")
        else:
            tags.append("service")
        
        # Prepare headers for ntfy.sh API
        headers = {
            "Title": title,
            "Priority": priority_map[status],
            "Tags": ",".join(tags)
        }
        
        if self.dryrun:
            # Format detailed dry-run log with headers info
            lines = [
                "=" * 60,
                f"🔧 [DRY RUN] NTFY NOTIFICATION ({source.upper()})",
                "=" * 60,
                f"🔗 URL: {self.url}",
                f"📧 Title: {title}",
                f"🔸 Priority: {priority_map[status]}",
                f"🏷️  Tags: {', '.join(tags)}",
                "",
                "📝 Message:",
            ]
            lines.extend(f"   {line}" for line in message.split('\n'))
            lines.append("=" * 60)
            logger.info("\n".join(lines))
        else:
            # Send using ntfy.sh API format with headers and message as data
            response = requests.post(self.url, data=message, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info(f"✅ Sent ntfy notification for {source} ({status})")


class LogHandler(NotificationHandler):
    """Log notifications to file."""
    
    def __init__(self, log_file: Optional[str] = None, enable_buffering: bool = False, buffer_time: float = 10.0):
        super().__init__("log", enable_buffering, buffer_time)
        self.log_file = log_file
        
    def _send_service_notification(self, service_health: ServiceHealth):
        """Log service health notification."""
        message = f"[{datetime.now().isoformat()}] Service {service_health.service_name} health: {service_health.overall_status.value}"
        
        if service_health.checks:
            latest_check = service_health.checks[-1]
            message += f" - Latest: {latest_check.name} ({latest_check.status.value})"
            if latest_check.message:
                message += f": {latest_check.message}"
        
        self._write_log_message(message)
        
    def _send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Log system health notification."""
        message = f"[{datetime.now().isoformat()}] System health update: {len(services)} services"
        
        status_counts = {}
        for svc in services.values():
            status_counts[svc.overall_status] = status_counts.get(svc.overall_status, 0) + 1
            
        for status, count in status_counts.items():
            message += f" | {status.value}: {count}"
        
        # Add service timelines for detailed logging
        message += "\nService Health Timelines:"
        for service_name, svc in sorted(services.items()):
            # Get recent health states from checks
            health_states = []
            if svc.checks:
                # Sort checks by timestamp to ensure chronological order
                sorted_checks = sorted(svc.checks, key=lambda x: x.timestamp)
                
                for check in sorted_checks[-10:]:  # Last 10 checks
                    if check.status not in health_states or check.status != health_states[-1]:
                        health_states.append(check.status)
            
            if not health_states:
                health_states = [svc.overall_status]
            elif len(health_states) == 1 and health_states[0] != svc.overall_status:
                health_states.append(svc.overall_status)
            
            timeline_str = " → ".join([state.value.upper() for state in health_states])
            message += f"\n  {service_name}: [{timeline_str}]"
        
        self._write_log_message(message)
    
    def _write_log_message(self, message: str):
        """Write message to log file and logger."""
        if self.log_file:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                logger.error(f"Failed to write health log: {e}")
        
        logger.info(message)


class WebhookHandler(NotificationHandler):
    """Send notifications via webhook."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None, 
                 dryrun: bool = False, enable_buffering: bool = False, buffer_time: float = 10.0):
        super().__init__("webhook", enable_buffering, buffer_time)
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.dryrun = dryrun
        
    def _send_service_notification(self, service_health: ServiceHealth):
        """Send webhook notification for service health change."""
        try:
            payload = {
                "type": "service_health",
                "timestamp": datetime.now().isoformat(),
                "service": service_health.to_dict()
            }
            
            self._send_webhook_payload(payload, "SERVICE")
            logger.info(f"Sent webhook notification for {service_health.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook service notification: {e}")
            
    def _send_system_notification(self, services: Dict[str, ServiceHealth]):
        """Send webhook notification for system-wide health changes."""
        try:
            payload = {
                "type": "system_health",
                "timestamp": datetime.now().isoformat(),
                "services": {name: svc.to_dict() for name, svc in services.items()}
            }
            
            self._send_webhook_payload(payload, "SYSTEM")
            logger.info("Sent system webhook notification")
            
        except Exception as e:
            logger.error(f"Failed to send system webhook notification: {e}")
    
    def _send_webhook_payload(self, payload: dict, notification_type: str):
        """Send webhook payload with dry-run support."""
        if not self.dryrun:
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
        else:
            # Consolidated dry-run log for webhook notification
            lines: list[str] = [
                "=" * 60,
                f"🌐 [DRY RUN] WEBHOOK NOTIFICATION ({notification_type})",
                "=" * 60,
                f"🔗 URL: {self.webhook_url}",
                f"📧 Type: {payload['type']}",
                f"⏰ Timestamp: {payload['timestamp']}",
            ]
            if notification_type == "SERVICE":
                service = payload['service']
                lines.extend([
                    f"🔸 Service: {service['service_name']}",
                    f"📊 Status: {service['overall_status']}",
                ])
                if service.get('uptime'):
                    uptime_hours = service['uptime'] / 3600
                    lines.append(f"⏱️  Uptime: {uptime_hours:.1f}h")
            else:
                services = payload['services']
                # summary counts
                status_counts: dict[str, int] = {}
                for svc in services.values():
                    status = svc['overall_status']
                    status_counts[status] = status_counts.get(status, 0) + 1
                lines.append(f"🔸 Services: {len(services)} total")
                for status, count in status_counts.items():
                    lines.append(f"   📊 {status.title()}: {count}")
            lines.append("=" * 60)
            logger.info("\n".join(lines))

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
        ntfy_handler = NtfyHandler(ntfy_topic, ntfy_server, dryrun=dry_run, 
                                  enable_buffering=enable_buffering, buffer_time=buffer_time)
        handlers.append(ntfy_handler)
    
    # Log handler (never buffered - we want immediate logging)
    log_file = os.environ.get("HEALTH_LOG_FILE")
    log_handler = LogHandler(log_file, enable_buffering=False)
    handlers.append(log_handler)
    
    # Webhook handler
    webhook_url = os.environ.get("HEALTH_WEBHOOK_URL")
    if webhook_url:
        webhook_headers = {}
        if os.environ.get("HEALTH_WEBHOOK_AUTH"):
            webhook_headers["Authorization"] = os.environ.get("HEALTH_WEBHOOK_AUTH")
        webhook_handler = WebhookHandler(webhook_url, webhook_headers, dryrun=dry_run,
                                       enable_buffering=enable_buffering, buffer_time=buffer_time)
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
    
    # Log handler (never buffered - we want immediate logging)
    log_handler = LogHandler(log_file, enable_buffering=False)
    handlers.append(log_handler)
    
    # ntfy handler
    if ntfy_topic:
        ntfy_handler = NtfyHandler(ntfy_topic, ntfy_server, dryrun=dry_run,
                                  enable_buffering=enable_buffering, buffer_time=buffer_time)
        handlers.append(ntfy_handler)
    
    # Webhook handler
    if webhook_url:
        webhook_handler = WebhookHandler(webhook_url, webhook_headers or {}, dryrun=dry_run,
                                       enable_buffering=enable_buffering, buffer_time=buffer_time)
        handlers.append(webhook_handler)
    
    logger.info(f"Initialized {len(handlers)} notification handlers")
    
    if enable_buffering:
        buffered_count = sum(1 for h in handlers if h.enable_buffering)
        logger.info(f"Buffering enabled for {buffered_count}/{len(handlers)} handlers with {buffer_time}s buffer time")
    
    if dry_run:
        logger.info("Notifications in dry-run mode - only logging enabled")
    
    return handlers
