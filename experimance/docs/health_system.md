# Experimance Unified Health Monitoring System

## ✅ **Status: Complete and Production Ready**

The unified health monitoring system is fully implemented and integrated across all Experimance services, providing comprehensive health tracking, smart notifications, and system-wide monitoring.

## 🎯 **Overview**

The Experimance health system provides real-time monitoring of all services through a unified architecture that automatically tracks health status, manages notifications, and enables comprehensive system monitoring.

### **Key Features**
- **🔄 Automatic Health Tracking**: All services automatically report health status
- **📁 File-Based Communication**: Health data persists in JSON files for inter-service communication
- **🎯 Smart Notifications**: Configurable notifications with cooldown and priority management
- **📊 Comprehensive Monitoring**: System-wide health visibility via healthcheck script
- **🔧 Zero Configuration**: Works out-of-the-box with `BaseService`

## 🏗️ **System Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Experimance Health System                        │
├─────────────────────────────────────────────────────────────────────┤
│  HealthStatus (enum)                                                │
│  ├─ HEALTHY    - Service operating normally                         │
│  ├─ WARNING    - Issues but still functional                        │
│  ├─ ERROR      - Significant issues affecting functionality         │
│  ├─ FATAL      - Non-functional, requires intervention              │
│  └─ UNKNOWN    - Status cannot be determined                        │
├─────────────────────────────────────────────────────────────────────┤
│  HealthReporter (per-service)                                       │
│  ├─ Records health checks with rich metadata                        │
│  ├─ Writes health status to service-specific JSON files             │
│  ├─ Tracks uptime, errors, restarts automatically                   │
│  ├─ Automatic overall status calculation                            │
│  └─ Smart notification management with cooldown                     │
├─────────────────────────────────────────────────────────────────────┤
│  Health Files (Inter-Service Communication)                         │
│  ├─ Development: logs/health/{service_type}.json                    │
│  ├─ Production: /var/cache/experimance/health/{service_type}.json   │
│  ├─ Real-time health status persistence                             │
│  └─ Enables health service monitoring                               │
├─────────────────────────────────────────────────────────────────────┤
│  HealthCheck Script (System Monitoring)                             │
│  ├─ Adaptive deployment detection (systemd vs processes)            │
│  ├─ Multiple check types (services, ZMQ, resources, processes)      │
│  ├─ Unified health system integration                               │
│  └─ Structured JSON output for monitoring tools                     │
├─────────────────────────────────────────────────────────────────────┤
│  NotificationHandlers (Configurable)                               │
│  ├─ LogHandler - File and console logging                           │
│  ├─ NtfyHandler - Push notifications (configurable)                 │
│  ├─ WebhookHandler - HTTP endpoint notifications                    │
│  └─ Extensible for email, Slack, Discord, etc.                      │
└─────────────────────────────────────────────────────────────────────┘
```

### **Health Data Flow**
1. **Service Health Checks** → `HealthReporter` records status
2. **Health Files** → JSON files written atomically for persistence  
3. **Health Service** → Monitors all health files for system overview
4. **Notifications** → Smart alerting based on status changes and cooldown
5. **Monitoring** → HealthCheck script provides comprehensive system status

## 🔧 **Core Components**

### **1. Automatic Health Reporting (BaseService)**
Every service extending `BaseService` automatically gets health monitoring:

```python
from experimance_common.base_service import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", "worker")
        # HealthReporter automatically created and configured!
        
    async def work(self):
        try:
            # Do work
            self.health_reporter.record_health_check(
                "work_task", 
                HealthStatus.HEALTHY,
                "Task completed successfully"
            )
        except Exception as e:
            # Automatic error reporting and health status update
            self.health_reporter.record_error(e, is_fatal=False)
```

### **2. Health File Communication**
Health data persists in JSON files for inter-service communication:

```bash
# Development
logs/health/
├── core.json          # Core service health
├── audio.json         # Audio service health  
├── display.json       # Display service health
└── ...

# Production  
/var/cache/experimance/health/
├── core.json
├── audio.json
└── ...
```

### **3. System Health Monitoring**
The health check script provides comprehensive system monitoring:

```bash
# Single health check
uv run python infra/scripts/healthcheck.py

# Continuous monitoring
uv run python infra/scripts/healthcheck.py monitor
```

**Features:**
- **Adaptive Mode Detection**: Automatically detects systemd (production) vs process (development)
- **Multiple Check Types**: Service status, ZMQ endpoints, system resources, process health  
- **Unified Integration**: Uses the same health system as all services
- **Structured Output**: JSON format for integration with monitoring tools

## 🎨 **Key Features**

### **1. Zero Configuration Required**
- **Automatic Setup**: All `BaseService` instances get health monitoring automatically
- **Smart Defaults**: Sensible configuration for development and production environments
- **File-Based Communication**: No complex networking or databases required

### **2. Production Ready**
- **Robust Error Handling**: Health system self-monitors and recovers from errors
- **Performance Optimized**: Efficient health check aggregation and minimal overhead
- **Environment Aware**: Automatically adapts to development vs production deployment

### **3. Comprehensive Monitoring**
- **Service Health**: Individual service status with detailed health checks
- **System Resources**: Memory, CPU, disk usage monitoring
- **Network Connectivity**: ZMQ endpoint health verification
- **Process Management**: Service process monitoring (systemd/development)

### **4. Smart Notifications**
- **Configurable Handlers**: Log, push notifications, webhooks, email support
- **Notification Management**: Automatic rate limiting prevents alert fatigue
- **Rich Context**: Detailed error information and health metadata
- **Priority Mapping**: Health status automatically maps to notification priority

## 🚀 **Usage Examples**

### **Basic Service with Automatic Health Monitoring**
```python
from experimance_common.base_service import BaseService
import logging

logger = logging.getLogger(__name__)

class CameraService(BaseService):
    def __init__(self):
        super().__init__("camera-service", "display")
        # Health monitoring automatically configured!
        
    async def start(self):
        logger.info("Starting camera service")
        await super().start()
        
    async def capture_frame(self):
        try:
            frame = await self.camera.get_frame()
            # Health check automatically recorded on successful operation
            return frame
        except Exception as e:
            # Error automatically recorded in health system
            logger.error(f"Frame capture failed: {e}")
            raise
```

### **Custom Health Checks**
```python
class DataService(BaseService):
    async def process_data(self):
        # Record specific health checks
        self.health_reporter.record_health_check(
            "data_processing",
            HealthStatus.HEALTHY,
            "Processed 1000 records successfully",
            metadata={"records_processed": 1000, "processing_time": 2.5}
        )
        
    async def check_database_connection(self):
        try:
            await self.db.ping()
            self.health_reporter.record_health_check(
                "database_connectivity",
                HealthStatus.HEALTHY,
                "Database connection healthy"
            )
        except Exception as e:
            self.health_reporter.record_health_check(
                "database_connectivity", 
                HealthStatus.ERROR,
                f"Database connection failed: {e}"
            )
```

### **System Health Monitoring**
```bash
# Development monitoring
uv run python infra/scripts/healthcheck.py

# Example output shows:
# - Service status (systemd/process based)
# - ZMQ endpoint health
# - System resource usage
# - Individual service health from health files

# Continuous monitoring for production
uv run python infra/scripts/healthcheck.py monitor
```

### **Health File Integration**
```python
# Health service can monitor all services by reading health files
from pathlib import Path
import json

health_dir = Path("logs/health")
for health_file in health_dir.glob("*.json"):
    with open(health_file) as f:
        service_health = json.load(f)
        print(f"Service {service_health['service_name']}: {service_health['overall_status']}")
```

## 📊 **Health System Benefits**

### **1. Comprehensive Visibility**
- ✅ Real-time health status for all services
- ✅ Historical health data and trends
- ✅ Rich error context for debugging
- ✅ System-wide health overview

### **2. Operational Excellence**
- ✅ Automatic health monitoring with zero configuration
- ✅ Smart notification management prevents alert fatigue
- ✅ Self-healing capabilities with detailed error reporting
- ✅ Production-ready monitoring and alerting

### **3. Developer Experience**
- ✅ Works out-of-the-box with `BaseService`
- ✅ Automatic health file generation for inter-service communication
- ✅ Comprehensive health check script for system monitoring
- ✅ Rich health metadata for debugging and analysis

### **4. System Integration**
- ✅ File-based communication enables health service monitoring
- ✅ Adaptive deployment detection (development vs production)
- ✅ Integration with logging system for comprehensive monitoring
- ✅ Structured health data for monitoring tool integration

## 🎯 **Best Practices**

### **1. Service Development**
- **Use `BaseService`**: Automatic health monitoring and logging integration
- **Add Custom Checks**: Include service-specific health validations
- **Handle Errors Gracefully**: Let the health system track and report issues
- **Monitor Health Files**: Use health file data for service coordination

### **2. System Monitoring**
- **Regular Health Checks**: Use the health check script for system oversight
- **Monitor Health Files**: Track service health via the file-based system
- **Configure Notifications**: Set up appropriate alert channels for your environment
- **Review Health Trends**: Use health data for system optimization

### **3. Production Deployment**
- **Environment Detection**: System automatically adapts to production mode
- **Health File Locations**: Ensure proper permissions for health file directories
- **Notification Configuration**: Configure notification handlers for your alert needs
- **Log Monitoring**: Integrate health logs with your monitoring infrastructure

## 🚀 **Getting Started**

### **1. For New Services**
```python
# Just extend BaseService - health monitoring included!
from experimance_common.base_service import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", "worker")
        # Health system automatically configured
```

### **2. For System Monitoring**
```bash
# Check overall system health
uv run python infra/scripts/healthcheck.py

# Run continuous monitoring
uv run python infra/scripts/healthcheck.py monitor
```

### **3. For Health Integration**
```python
# Add custom health checks
self.health_reporter.record_health_check(
    "custom_check",
    HealthStatus.HEALTHY, 
    "All systems operational"
)
```

The unified health system is production-ready and provides comprehensive monitoring with minimal configuration required! 🎉
