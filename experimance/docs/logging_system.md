# Experimance Logging System

## ✅ **Status: Complete and Working**

The adaptive logging system is now fully implemented and integrated into all services.

## 🔧 **For Service Developers**

### **No Setup Required!**
All services that extend `BaseService` automatically get proper logging:

```python
import logging
from experimance_common.base_service import BaseService

logger = logging.getLogger(__name__)

class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", "worker")
        # Logging is automatically configured!
        
    async def start(self):
        logger.info("Service starting")  # Works immediately!
        await super().start()
```

### **Automatic Behavior**
- **Development**: Logs to `logs/base_service.log` (local directory) + console output
- **Production**: Logs to `/var/log/experimance/base_service.log` (system location) - **file only**
- **External libraries**: Automatically quieted (httpx, PIL, etc.)
- **Log rotation**: Handled automatically in production
- **No duplication**: Production avoids console output since systemd captures stdout/stderr

## 🏗️ **Architecture**

### **Key Components**
1. **`experimance_common.logger.setup_logging()`** - Main logging configuration
2. **`experimance_common.logger.get_log_file_path()`** - Environment-aware file paths
3. **`experimance_common.logger.configure_external_loggers()`** - External library management
4. **`BaseService`** - Automatically calls `setup_logging()` for all services

### **Environment Detection**
Production mode when:
- Running as root (`os.geteuid() == 0`)
- `EXPERIMANCE_ENV=production`
- `/etc/experimance` directory exists

## 📁 **File Locations**

### **Development**
```
experimance/
├── logs/
│   ├── base_service.log      # All BaseService subclasses
│   ├── healthcheck.log       # Health monitoring script
│   ├── health_status.json    # Health status snapshot
│   └── health/               # Individual service health files
│       ├── core.json         # Core service health
│       ├── audio.json        # Audio service health
│       └── ...               # Other service health files
```

### **Production**
```
/var/log/experimance/
├── base_service.log
├── healthcheck.log
├── health_status.json
└── ...

/var/cache/experimance/health/
├── core.json
├── audio.json
└── ...
```

## 🔍 **Usage Examples**

### **Basic Service Logging**
```python
# Just use the logger - BaseService handles setup
import logging
logger = logging.getLogger(__name__)

class MyService(BaseService):
    async def work(self):
        logger.info("Doing work")
        logger.warning("Something unusual happened")
        logger.error("An error occurred")
```

### **Custom Logging Setup** (Advanced)
```python
from experimance_common.logger import setup_logging

# Custom configuration
logger = setup_logging(
    name="my-custom-service",
    log_filename="custom.log",
    level=logging.DEBUG,
    include_console=False,
    external_level=logging.ERROR
)
```

### **Health Monitoring**
```bash
# Single health check (development or production)
uv run python infra/scripts/healthcheck.py

# Continuous monitoring mode
uv run python infra/scripts/healthcheck.py monitor

# View health logs
tail -f logs/healthcheck.log

# View current health status
cat logs/health/health_status.json
```

### **Health Service Integration**
All services using `BaseService` automatically integrate with the unified health system:

```python
from experimance_common.base_service import BaseService

class MyService(BaseService):
    def __init__(self):
        super().__init__("my-service", "worker")
        # Health reporting is automatically configured!
        
    async def work(self):
        # Health checks are recorded automatically
        # Custom health checks can be added
        self.health_reporter.record_health_check(
            "custom_check",
            HealthStatus.HEALTHY,
            "All systems operational"
        )
```

## 📊 **Monitoring Integration**

### **Unified Health System**
The logging system is fully integrated with the unified health monitoring:

- **Automatic Health Reporting**: All `BaseService` instances automatically create health files
- **Health Status Files**: Located in `logs/health/` (development) or `/var/cache/experimance/health/` (production)
- **Real-time Monitoring**: Health service continuously monitors all service health files
- **Smart Notifications**: Health system handles notifications with cooldown and flush capabilities

### **Health Check Script**
The healthcheck script provides comprehensive system monitoring:

- **Adaptive Mode Detection**: Automatically detects systemd (production) vs process (development) deployment
- **Multiple Check Types**: Service status, ZMQ endpoints, system resources, process health
- **Unified Health Integration**: Uses the same health system as services
- **JSON Output**: Structured health status for monitoring tools
- **Issue Detection**: Identifies and reports all health problems

### **Health Status**
- Health checks create `health_status.json` with system state
- Logs provide detailed operation history
- Notification system handles alerts with proper cooldown

### **Log Rotation (Production)**
- Automatic via system logrotate
- Configured in `/etc/logrotate.d/experimance`
- Keeps logs manageable

## 🎯 **Best Practices**

1. **Use `BaseService`** - Gets logging and health monitoring automatically
2. **Use standard logger** - `logging.getLogger(__name__)`
3. **Appropriate log levels** - INFO for normal operations, WARNING for issues, ERROR for problems
4. **Health integration** - Let `BaseService` handle health reporting automatically
5. **Custom health checks** - Add specific health checks for service-specific functionality
6. **Test both environments** - Development and production modes
7. **Monitor health files** - Use the health check script for system monitoring

## 🛠️ **Implementation Notes**

- **Unified health system** - All services use the same health reporting mechanism
- **Health file monitoring** - Services write health status to individual files for monitoring
- **Adaptive deployment detection** - Health check script automatically detects production vs development
- **No breaking changes** - Existing `logging.getLogger(__name__)` calls work unchanged
- **Backward compatible** - Old services continue to work
- **Production ready** - Tested in both development and production-like environments
- **Health service integration** - Health service can monitor all services via health files

The logging and health monitoring system is now complete and ready for deployment! 🚀
