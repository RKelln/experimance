# Experimance Infrastructure Summary

## What We've Built

A comprehensive infrastructure solution for remote monitoring and management of the Experimance installation, designed for minimal maintenance during month-long exhibitions.

## Key Features ✅

### 1. **Easy Service Management**
- **One-command deployment**: `sudo ./infra/scripts/deploy.sh experimance install`
- **Project switching**: Support for multiple installations (experimance, sohkepayin, etc.)
- **Service lifecycle**: Start, stop, restart, status checking all automated

### 2. **Auto-Recovery**
- **systemd integration**: Native Ubuntu service management with automatic restart
- **Failure detection**: Services automatically restart on crashes
- **Resource limits**: Prevent runaway processes from consuming all resources

### 3. **Remote Monitoring**
- **Push notifications**: Simple, reliable ntfy.sh notifications to your phone
- **Email alerts**: Configurable SMTP notifications as fallback
- **Web dashboard**: Mobile-friendly interface at `http://installation-ip:8080`
- **Health checks**: Automated monitoring every 5 minutes
- **SSH access**: Secure remote access for troubleshooting

### 4. **Safe Updates**
- **Rollback capability**: Automatic rollback on failed updates
- **Backup system**: Configuration and state backups before changes
- **Git integration**: Simple `git pull` workflow with safety checks

## Time Investment

- **Initial setup**: 2-3 hours
- **Monthly maintenance**: 1-2 hours
- **Emergency response**: 15-30 minutes

## Files Created

```
infra/
├── systemd/                    # Service definitions (updated for uv)
│   ├── experimance-core@.service
│   ├── experimance-display@.service  
│   ├── experimance-health@.service
│   ├── image-server@.service
│   └── experimance@.target
├── scripts/                    # Management automation
│   ├── deploy.sh              # Main deployment script (environment-aware)
│   ├── get_project_services.py # Dynamic service detection
│   └── (other utility scripts planned)
└── docs/                      # Documentation
    ├── deployment.md          # Complete deployment guide
    └── README.md              # This summary

scripts/                       # Development tools
└── dev                        # Development service runner

services/health/               # Dedicated health monitoring service
├── src/experimance_health/
│   ├── health_service.py      # Health monitoring implementation
│   └── config.py             # Health service configuration
├── config.toml               # Default health configuration
└── (project overrides in projects/*/health.toml)

libs/common/src/experimance_common/
├── health.py                 # Health system utilities
├── notifications.py          # Notification handlers (ntfy, email, etc.)
└── (other common utilities)
```

## Quick Start Commands

### Development (Individual Services)
```bash
# Run individual services for development/testing  
./scripts/dev core     # Start core service
./scripts/dev display  # Start display service
./scripts/dev health   # Start health monitoring
./scripts/dev          # Show available services

# Each automatically:
# - Sets EXPERIMANCE_ENV=development
# - Uses current user (no sudo needed)
# - Handles dependencies with uv
# - Uses local cache directories
```

### Production (Full Deployment)
```bash
# One-time setup
sudo useradd -m -s /bin/bash experimance
sudo ./infra/scripts/deploy.sh experimance install
sudo ./infra/scripts/deploy.sh experimance start

# Daily operations
sudo ./infra/scripts/deploy.sh experimance status
sudo ./infra/scripts/deploy.sh experimance restart

# View available services
./infra/scripts/deploy.sh experimance services
```

## Emergency Quick Reference

### Check Status
```bash
# Development
./scripts/dev          # Show available services
ps aux | grep experimance  # Check running dev services

# Production  
sudo ./infra/scripts/deploy.sh experimance status
sudo systemctl status "experimance-*@experimance"
```

### Restart Everything
```bash
# Development: Stop dev services (Ctrl+C) and restart them

# Production
sudo ./infra/scripts/deploy.sh experimance restart
```

### View Logs
```bash
# Development: Logs go to console where you started ./scripts/dev

# Production
sudo journalctl -u experimance-core@experimance -f
sudo journalctl -u experimance-health@experimance -f  # Health notifications
```

### Health Monitoring
```bash
# View health status files
ls -la /var/cache/experimance/health/     # Production
ls -la cache/health/                      # Development

# Check health service (production only)
sudo journalctl -u experimance-health@experimance -f
```

## Monitoring Options

1. **Push Notifications**: Built-in ntfy.sh support for instant phone alerts
2. **Email Alerts**: Traditional email notifications as alternative
3. **Health Service**: Dedicated monitoring service with intelligent filtering
4. **File-based Status**: JSON health status files for inter-service communication

### Quick Setup

```bash
# Configure health monitoring (automatic with deploy script)
# Edit projects/experimance/health.toml for notification settings

# Install ntfy app on your phone and subscribe to your topic  
# The health service automatically sends test notifications when starting

# View health status
ls -la /var/cache/experimance/health/     # Production  
ls -la cache/health/                      # Development
```

## Nice-to-Have Features

- [ ] **Backup to cloud**: Automatic configuration backup to Google Drive/Dropbox
- [ ] **Performance metrics**: CPU, memory, GPU usage trends
- [ ] **Remote terminal**: Web-based terminal access
- [ ] **Log search**: Web interface for searching logs
- [ ] **Metrics dashboard**: Grafana-style performance graphs

## Risk Mitigation

- **Hardware failure**: Keep backup hardware on-site
- **Network loss**: Mobile hotspot backup
- **Power outage**: UPS for critical components
- **Service crashes**: Automatic restart + alerting
- **Configuration errors**: Version control + rollback

## Cost Estimate

- **Infrastructure**: $0 (using existing hardware)
- **Monitoring services**: $0-30/month (email/SMS)
- **Time investment**: 2-3 hours initial, 1-2 hours monthly

## Success Criteria

✅ **Simple**: One command to start/stop/restart all services
✅ **Reliable**: Automatic recovery from common failures
✅ **Monitored**: Immediate notification of serious issues
✅ **Maintainable**: Remote SSH access for troubleshooting
✅ **Updatable**: Safe remote updates with rollback
✅ **Documented**: Clear procedures for common tasks

This infrastructure provides enterprise-grade reliability while maintaining the simplicity needed for art installations.
