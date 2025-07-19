#!/usr/bin/env python3
"""
Experimance Health Check Script
Monitors all services and sends alerts if issues are detected.
Uses the unified health system for consistent status reporting.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from experimance_common.config import resolve_path, get_project_services
from experimance_common.constants import PROJECT_SPECIFIC_DIR
from experimance_common.logger import setup_logging
from experimance_common.health import (
    HealthStatus, HealthReporter, SystemHealthMonitor, 
    create_health_reporter, system_monitor
)
from experimance_common.notifications import create_notification_handlers

# Configure logging
logger = setup_logging(__name__)

# Load environment variables from project .env file
try:
    from dotenv import load_dotenv
    env_file = resolve_path(".env", hint="project")
    load_dotenv(env_file)
except ImportError:
    # dotenv not available, that's okay
    pass

import aiohttp
import requests
import zmq
import zmq.asyncio

# Configuration
CONFIG = {
    "project": os.environ.get("PROJECT_ENV", "experimance"),
    "services": get_project_services(os.environ.get("PROJECT_ENV", "experimance")),
    "zmq_endpoints": {
        "events": "tcp://localhost:5555",
        "depth": "tcp://localhost:5556"
    },
    "thresholds": {
        "memory_percent": 90,
        "cpu_percent": 90,
        "disk_percent": 95,
        "restart_count": 3
    },
    "alerting": {
        "email_enabled": os.environ.get("ALERT_EMAIL_ENABLED", "false").lower() == "true",
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "email_from": os.environ.get("ALERT_EMAIL"),
        "email_to": os.environ.get("ALERT_EMAIL_TO", "").split(","),
        "email_password": os.environ.get("ALERT_EMAIL_PASSWORD"),
        "ntfy_enabled": True,
        "ntfy_topic": os.environ.get("NTFY_TOPIC", "experimance-alerts"),
        "ntfy_server": os.environ.get("NTFY_SERVER", "https://ntfy.sh"),
        "ntfy_priority": os.environ.get("NTFY_PRIORITY", "high"),
        "cooldown_minutes": 10  # Don't spam alerts
    }
}

# Setup logging using common utilities
logger = setup_logging(__name__, log_filename="healthcheck.log")

class HealthChecker:
    """Unified health checker using the new health system."""
    
    def __init__(self):
        self.last_alerts = {}
        self.zmq_context = zmq.asyncio.Context()
        
        # Create health reporter for the health check system itself
        self.health_reporter = create_health_reporter("healthcheck")
        
        # Setup notification handlers
        self.notification_handlers = create_notification_handlers()
        
        # Register handlers with the system monitor
        for handler in self.notification_handlers:
            system_monitor.add_notification_handler(self._handle_system_notification)
        
        logger.info(f"HealthChecker initialized with {len(self.notification_handlers)} notification handlers")
    
    def _handle_system_notification(self, services: Dict[str, Any]):
        """Handle system-wide health notifications."""
        # Send notifications via all handlers
        for handler in self.notification_handlers:
            try:
                handler.send_system_notification(services)
            except Exception as e:
                logger.error(f"Failed to send notification via {handler.name}: {e}")
    
    async def check_systemd_services(self) -> Dict[str, HealthStatus]:
        """Check systemd services and return health status."""
        service_health = {}
        
        for service in CONFIG["services"]:
            service_name = f"{service}@{CONFIG['project']}"
            
            try:
                # Check if service is active
                result = subprocess.run(
                    ["systemctl", "is-active", service_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    service_health[service] = HealthStatus.HEALTHY
                    logger.debug(f"Service {service_name} is active")
                else:
                    service_health[service] = HealthStatus.FATAL
                    logger.warning(f"Service {service_name} is not active: {result.stdout.strip()}")
                
                # Record health check for this service
                self.health_reporter.record_health_check(
                    f"systemd_{service}",
                    service_health[service],
                    f"Service {service_name} status: {result.stdout.strip()}"
                )
                
            except Exception as e:
                logger.error(f"Error checking service {service_name}: {e}")
                service_health[service] = HealthStatus.UNKNOWN
                self.health_reporter.record_health_check(
                    f"systemd_{service}",
                    HealthStatus.ERROR,
                    f"Failed to check service {service_name}: {e}"
                )
        
        return service_health
    
    async def check_zmq_endpoints(self) -> Dict[str, HealthStatus]:
        """Check ZMQ endpoints and return health status."""
        endpoint_health = {}
        
        for name, endpoint in CONFIG["zmq_endpoints"].items():
            try:
                # Create a subscriber socket with timeout
                socket = self.zmq_context.socket(zmq.SUB)
                socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
                socket.setsockopt(zmq.SUBSCRIBE, b"")
                socket.connect(endpoint)
                
                # Try to receive a message (non-blocking)
                try:
                    await socket.recv_multipart(zmq.NOBLOCK)
                    endpoint_health[name] = HealthStatus.HEALTHY
                    logger.debug(f"ZMQ endpoint {name} is responding")
                except zmq.Again:
                    # No message available, but connection is working
                    endpoint_health[name] = HealthStatus.HEALTHY
                    logger.debug(f"ZMQ endpoint {name} is connected (no messages)")
                except Exception as e:
                    logger.warning(f"ZMQ endpoint {name} ({endpoint}) error: {e}")
                    endpoint_health[name] = HealthStatus.ERROR
                    
                socket.close()
                
                # Record health check
                self.health_reporter.record_health_check(
                    f"zmq_{name}",
                    endpoint_health[name],
                    f"ZMQ endpoint {name} status"
                )
                
            except Exception as e:
                logger.error(f"Error checking ZMQ endpoint {name}: {e}")
                endpoint_health[name] = HealthStatus.ERROR
                self.health_reporter.record_health_check(
                    f"zmq_{name}",
                    HealthStatus.ERROR,
                    f"Failed to check ZMQ endpoint {name}: {e}"
                )
                
        return endpoint_health
    
    async def check_system_resources(self) -> Dict[str, float]:
        """Check system resource usage."""
        resources = {}
        
        try:
            # Memory usage
            with open("/proc/meminfo", "r") as f:
                meminfo = f.read()
            
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal:' in line][0].split()[1])
            mem_available = int([line for line in meminfo.split('\n') if 'MemAvailable:' in line][0].split()[1])
            mem_used_percent = ((mem_total - mem_available) / mem_total) * 100
            resources["memory_percent"] = mem_used_percent
            
            # Disk usage
            result = subprocess.run(["df", "/"], capture_output=True, text=True)
            disk_line = result.stdout.split('\n')[1]
            disk_used_percent = float(disk_line.split()[4].rstrip('%'))
            resources["disk_percent"] = disk_used_percent
            
            # CPU usage (simplified)
            with open("/proc/loadavg", "r") as f:
                load_avg = float(f.read().split()[0])
            # Assume 4 cores, so 100% would be load of 4
            cpu_percent = (load_avg / 4) * 100
            resources["cpu_percent"] = min(cpu_percent, 100)
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            
        return resources
    
    async def check_process_health(self) -> Dict[str, Dict]:
        """Check health of individual processes."""
        process_health = {}
        
        for service in CONFIG["services"]:
            service_name = f"{service}@{CONFIG['project']}"
            try:
                # Get service status
                result = subprocess.run(
                    ["systemctl", "show", service_name, "--property=ActiveState,SubState,ExecMainPID,NRestarts"],
                    capture_output=True,
                    text=True
                )
                
                status_dict = {}
                for line in result.stdout.split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        status_dict[key] = value
                
                process_health[service] = {
                    "active_state": status_dict.get("ActiveState", "unknown"),
                    "sub_state": status_dict.get("SubState", "unknown"),
                    "pid": status_dict.get("ExecMainPID", "0"),
                    "restart_count": int(status_dict.get("NRestarts", "0"))
                }
                
            except Exception as e:
                logger.error(f"Error checking process health for {service}: {e}")
                process_health[service] = {"error": str(e)}
                
        return process_health
    
    async def send_alert(self, subject: str, message: str) -> bool:
        """Send an alert via ntfy and/or email."""
        # Check cooldown
        now = datetime.now()
        if subject in self.last_alerts:
            if now - self.last_alerts[subject] < timedelta(minutes=CONFIG["alerting"]["cooldown_minutes"]):
                return False
        
        alert_sent = False
        
        # Try ntfy first (it's more reliable)
        if CONFIG["alerting"]["ntfy_enabled"]:
            try:
                import aiohttp
                
                ntfy_url = f"{CONFIG['alerting']['ntfy_server']}/{CONFIG['alerting']['ntfy_topic']}"
                
                # Create detailed message for ntfy
                full_message = f"{subject}\n\n{message}\n\nTime: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                
                headers = {
                    "Title": f"Experimance Alert: {subject}",
                    "Priority": CONFIG["alerting"]["ntfy_priority"],
                    "Tags": "warning,computer,experimance"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(ntfy_url, data=full_message, headers=headers) as response:
                        if response.status == 200:
                            logger.info(f"ntfy alert sent: {subject}")
                            alert_sent = True
                        else:
                            logger.warning(f"ntfy alert failed: HTTP {response.status}")
                            
            except Exception as e:
                logger.error(f"Error sending ntfy alert: {e}")
        
        # Fallback to email if configured
        if CONFIG["alerting"]["email_enabled"] and not alert_sent:
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                msg = MIMEMultipart()
                msg['From'] = CONFIG["alerting"]["email_from"]
                msg['To'] = ", ".join(CONFIG["alerting"]["email_to"])
                msg['Subject'] = f"Experimance Alert: {subject}"
                
                msg.attach(MIMEText(message, 'plain'))
                
                server = smtplib.SMTP(CONFIG["alerting"]["smtp_server"], CONFIG["alerting"]["smtp_port"])
                server.starttls()
                server.login(CONFIG["alerting"]["email_from"], CONFIG["alerting"]["email_password"])
                
                server.send_message(msg)
                server.quit()
                
                logger.info(f"Email alert sent: {subject}")
                alert_sent = True
                
            except Exception as e:
                logger.error(f"Error sending email alert: {e}")
        
        if alert_sent:
            self.last_alerts[subject] = now
            return True
        else:
            logger.error(f"Failed to send alert: {subject}")
            return False
    
    async def run_health_check(self) -> Dict:
        """Run a complete health check using the unified health system."""
        logger.info("Starting unified health check...")
        
        try:
            # Run all checks
            service_status = await self.check_systemd_services()
            zmq_status = await self.check_zmq_endpoints()
            resources = await self.check_system_resources()
            process_health = await self.check_process_health()
            
            # Record system resource health checks
            for resource, value in resources.items():
                threshold = CONFIG["thresholds"].get(resource, 100)
                if value > threshold:
                    self.health_reporter.record_health_check(
                        f"system_{resource}",
                        HealthStatus.WARNING,
                        f"{resource}: {value}% (threshold: {threshold}%)"
                    )
                else:
                    self.health_reporter.record_health_check(
                        f"system_{resource}",
                        HealthStatus.HEALTHY,
                        f"{resource}: {value}%"
                    )
            
            # Record process health checks
            for process, details in process_health.items():
                if details.get("restart_count", 0) > CONFIG["thresholds"]["restart_count"]:
                    self.health_reporter.record_health_check(
                        f"process_{process}_restarts",
                        HealthStatus.WARNING,
                        f"Process {process} has {details['restart_count']} restarts"
                    )
                else:
                    self.health_reporter.record_health_check(
                        f"process_{process}",
                        HealthStatus.HEALTHY,
                        f"Process {process} is healthy"
                    )
            
            # Compile results (for backward compatibility)
            results = {
                "timestamp": datetime.now().isoformat(),
                "project": CONFIG["project"],
                "services": {k: v.value for k, v in service_status.items()},
                "zmq_endpoints": {k: v.value for k, v in zmq_status.items()},
                "resources": resources,
                "processes": process_health,
                "health_summary": self.health_reporter.get_health_summary(),
                "system_health": system_monitor.get_system_health(),
                "alerts": []  # Maintained for backward compatibility
            }
            
            # Legacy alert collection for backward compatibility
            failed_services = [svc for svc, status in service_status.items() if status != HealthStatus.HEALTHY]
            if failed_services:
                alert_msg = f"Services with issues: {', '.join(failed_services)}"
                results["alerts"].append(alert_msg)
            
            # Resource alerts
            for resource, value in resources.items():
                threshold = CONFIG["thresholds"].get(resource, 100)
                if value > threshold:
                    alert_msg = f"{resource} is at {value:.1f}% (threshold: {threshold}%)"
                    results["alerts"].append(alert_msg)
            
            # Process restart alerts
            for service, health in process_health.items():
                if health.get("restart_count", 0) > CONFIG["thresholds"]["restart_count"]:
                    alert_msg = f"Service {service} has restarted {health['restart_count']} times"
                    results["alerts"].append(alert_msg)
            
            # Record overall health check completion
            self.health_reporter.record_health_check(
                "health_check_complete",
                HealthStatus.HEALTHY,
                "Health check completed successfully"
            )
            
            logger.info(f"Unified health check complete. Found {len(results['alerts'])} legacy issues.")
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            self.health_reporter.record_error(e, is_fatal=False)
            results = {
                "timestamp": datetime.now().isoformat(),
                "project": CONFIG["project"],
                "error": str(e),
                "health_summary": self.health_reporter.get_health_summary(),
                "system_health": system_monitor.get_system_health(),
                "alerts": [f"Health check error: {e}"]
            }
        
        return results
        
        # Always save results to file for status tracking
        try:
            from experimance_common.logger import get_log_file_path
            log_file = get_log_file_path("healthcheck.log")
            results_file = Path(log_file).parent / "health_status.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save health status file: {e}")
        
        return results
    
    async def continuous_monitoring(self, interval: int = 300):
        """Run continuous monitoring."""
        logger.info(f"Starting continuous monitoring (interval: {interval}s)")
        
        while True:
            try:
                results = await self.run_health_check()
                
                # Save results to file in the same directory as logs
                from experimance_common.logger import get_log_file_path
                log_file = get_log_file_path("healthcheck.log")
                results_file = Path(log_file).parent / "health_status.json"
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

async def main():
    """Main function."""
    checker = HealthChecker()
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Continuous monitoring mode
        await checker.continuous_monitoring()
    else:
        # Single check mode
        results = await checker.run_health_check()
        print(json.dumps(results, indent=2))
        
        # Exit with error code if there are issues
        if results["alerts"]:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
