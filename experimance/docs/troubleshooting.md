# Experimance Troubleshooting Guide

This guide covers common issues, their causes, and solutions for the Experimance system. Use this as your first resource when encountering problems.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Service Startup Problems](#service-startup-problems)
4. [Hardware Issues](#hardware-issues)
5. [Communication Problems](#communication-problems)
6. [Performance Issues](#performance-issues)
7. [Production Deployment Issues](#production-deployment-issues)
8. [Development Issues](#development-issues)
9. [Getting Help](#getting-help)

## Quick Diagnostics

### System Health Check

Run these commands to quickly assess system health:

```bash
# Check if all services are running
./infra/scripts/status.sh experimance

# Test basic functionality
uv run python utils/tests/simple_test.py

# Check environment setup
uv run python utils/tests/check_env.py

# Verify ZMQ communication
uv run python utils/tests/test_zmq_utils.py
```

### Log Analysis

Check logs for immediate issues:

```bash
# View recent logs for all services
tail -f logs/*.log

# Check specific service logs
tail -f logs/core.log
tail -f logs/display.log
tail -f logs/image_server.log

# Check system logs (production)
sudo journalctl -u experimance-core@experimance -n 50
```

### Common Error Patterns

Look for these patterns in logs:

- `ZMQError`: Communication issues between services
- `ImportError`: Missing dependencies or installation problems
- `PermissionError`: File/device access issues
- `ConnectionError`: Network or hardware connection problems
- `TimeoutError`: Service responsiveness issues

## Installation Issues

### Python Version Problems

**Problem**: Wrong Python version or pyenv issues
```
ERROR: Python 3.11 required, found 3.9
```

**Solution**:
```bash
# Install correct Python version
pyenv install 3.11.9
pyenv local 3.11.9

# Verify version
python --version

# Recreate virtual environment
rm -rf .venv
uv sync
```

### Package Installation Failures

**Problem**: Dependencies fail to install
```
ERROR: Failed building wheel for package-name
```

**Solutions**:

1. **Update system dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install build-essential libssl-dev libusb-1.0-0-dev

# macOS
brew install libusb
xcode-select --install
```

2. **Clear package cache**:
```bash
uv cache clean
rm -rf .venv
uv sync
```

3. **Install specific problematic packages**:
```bash
# Common problematic packages
uv add --no-deps opencv-python
uv add --no-deps pyrealsense2
```

### UV Package Manager Issues

**Problem**: UV not found or not working
```
command not found: uv
```

**Solution**:
```bash
# Reinstall uv
curl -sSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Or use pip as fallback
pip install uv
```

## Service Startup Problems

### Core Service Won't Start

**Problem**: Core service fails to start
```
ERROR: Failed to initialize depth camera
```

**Diagnosis**:
```bash
# Check camera connection
uv run python services/core/tests/test_camera.py --info

# Test with mock camera
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present
```

**Solutions**:

1. **Camera permission issues**:
```bash
# Add user to camera group
sudo usermod -a -G dialout $USER
sudo usermod -a -G video $USER

# Logout and login again
```

2. **Missing camera drivers**:
```bash
# Install RealSense drivers (Ubuntu)
sudo apt install librealsense2-*

# Verify camera detection
rs-enumerate-devices
```

3. **Use mock mode for testing**:
```bash
# Create mock depth images
mkdir -p media/images/mocks/depth

# Start with mock camera
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present
```

### Display Service Issues

**Problem**: Display service fails to start or shows black screen

**Diagnosis**:
```bash
# Test display capabilities
uv run python services/display/tests/test_display.py

# Check OpenGL support
glxinfo | grep "OpenGL version"
```

**Solutions**:

1. **Graphics driver issues**:
```bash
# Update graphics drivers
sudo apt update && sudo apt upgrade

# Install OpenGL libraries
sudo apt install mesa-utils libgl1-mesa-glx
```

2. **Display configuration**:
```bash
# Check display settings in config
nano projects/experimance/display.toml

# Test with windowed mode
uv run -m experimance_display --windowed
```

### Image Server Problems

**Problem**: Image generation fails or is very slow

**Diagnosis**:
```bash
# Test image generation
uv run python services/image_server/tests/test_generators.py

# Check GPU availability
nvidia-smi  # For NVIDIA GPUs
```

**Solutions**:

1. **GPU not available**:
```bash
# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
uv run -m image_server
```

2. **Out of memory errors**:
```bash
# Reduce batch size in config
nano projects/experimance/image_server.toml
# Set smaller batch_size and lower resolution
```

3. **Model download issues**:
```bash
# Clear model cache
rm -rf ~/.cache/huggingface
rm -rf models/

# Manually download models
uv run python scripts/download_models.py
```

### Audio Service Issues

**Problem**: SuperCollider fails to start or no audio output

**Diagnosis**:
```bash
# Test audio system
uv run python services/audio/tests/test_audio.py

# Check audio devices
aplay -l  # Linux
system_profiler SPAudioDataType  # macOS
```

**Solutions**:

1. **SuperCollider not installed**:
```bash
# Install SuperCollider
sudo apt install supercollider  # Ubuntu
brew install supercollider      # macOS
```

2. **Audio device issues**:
```bash
# Configure audio device in config
nano projects/experimance/audio.toml

# Test with default audio device
uv run -m experimance_audio --audio-device-default
```

3. **Permission issues**:
```bash
# Add user to audio group
sudo usermod -a -G audio $USER
```

## Hardware Issues

### Depth Camera Problems

**Problem**: RealSense camera not detected or producing errors

**Diagnosis**:
```bash
# Check camera connection
rs-enumerate-devices

# Test camera functionality
rs-capture

# Check USB connection
lsusb | grep Intel
```

**Solutions**:

1. **Driver issues**:
```bash
# Reinstall RealSense drivers
sudo apt remove librealsense2-*
sudo apt install librealsense2-*

# Update firmware
rs-fw-update
```

2. **USB power issues**:
- Use USB 3.0 port with adequate power
- Try different USB cable
- Use powered USB hub if necessary

3. **Permission issues**:
```bash
# Set up udev rules
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Webcam Issues (Agent Service)

**Problem**: Webcam not accessible for agent vision

**Diagnosis**:
```bash
# Test webcam access
uv run python services/agent/tests/test_vision_imports.py

# List available cameras
uv run python scripts/list_cameras.py
```

**Solutions**:

1. **Camera in use by another application**:
```bash
# Find processes using camera
sudo lsof /dev/video0

# Kill conflicting processes
sudo pkill -f camera-app-name
```

2. **Permission issues**:
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Set camera permissions
sudo chmod 666 /dev/video*
```

### Audio Hardware Issues

**Problem**: Multi-channel audio not working

**Diagnosis**:
```bash
# Check audio interfaces
cat /proc/asound/cards

# Test multi-channel output
speaker-test -c 8 -t wav
```

**Solutions**:

1. **Configure ALSA/JACK**:
```bash
# Edit ALSA configuration
nano ~/.asoundrc

# Start JACK with correct settings
jackd -d alsa -r 48000 -p 512
```

2. **Check audio interface drivers**:
```bash
# Install audio interface drivers
sudo apt install linux-modules-extra-$(uname -r)
```

## Communication Problems

### ZMQ Port Conflicts

**Problem**: Services can't bind to ZMQ ports
```
ZMQError: Address already in use
```

**Diagnosis**:
```bash
# Check port usage
netstat -ln | grep 5555
ss -tulpn | grep 5555

# Find processes using ports
sudo lsof -i :5555
```

**Solutions**:

1. **Kill conflicting processes**:
```bash
# Find and kill processes using ZMQ ports
sudo pkill -f experimance
sudo pkill -f image_server

# Wait a moment for cleanup
sleep 2
```

2. **Change port configuration**:
```bash
# Edit port settings
nano projects/experimance/config.toml

# Use different port range
[zmq]
events_port = 6555
```

### Service Communication Timeouts

**Problem**: Services can't communicate with each other
```
TimeoutError: No response from service
```

**Diagnosis**:
```bash
# Test ZMQ communication
uv run python utils/tests/test_zmq_utils.py

# Check service health
./infra/scripts/status.sh experimance
```

**Solutions**:

1. **Increase timeout values**:
```bash
# Edit timeout settings
nano projects/experimance/config.toml

[zmq]
timeout_ms = 10000  # Increase from default
```

2. **Check network connectivity**:
```bash
# Test local network
ping localhost
telnet localhost 5555
```

3. **Restart services in order**:
```bash
# Stop all services
./infra/scripts/deploy.sh experimance stop

# Start core service first
uv run -m experimance_core &

# Wait, then start other services
sleep 5
uv run -m experimance_display &
uv run -m image_server &
```

## Performance Issues

### High CPU Usage

**Problem**: Services consuming too much CPU

**Diagnosis**:
```bash
# Monitor CPU usage
top -p $(pgrep -f experimance)
htop

# Profile specific service
uv run python -m cProfile -o profile.stats -m experimance_core
```

**Solutions**:

1. **Reduce processing frequency**:
```bash
# Edit processing rates
nano projects/experimance/core.toml

[depth_processing]
fps = 15  # Reduce from 30

[camera]
output_resolution = [320, 240]  # Reduce resolution
```

2. **Optimize image generation**:
```bash
# Use lower quality settings
nano projects/experimance/image_server.toml

[generation]
steps = 20  # Reduce from 50
guidance_scale = 5.0  # Reduce from 7.5
```

### Memory Issues

**Problem**: Services using too much memory or running out of memory

**Diagnosis**:
```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Check for memory leaks
valgrind --tool=memcheck python -m experimance_core
```

**Solutions**:

1. **Reduce memory usage**:
```bash
# Clear caches periodically
echo 3 | sudo tee /proc/sys/vm/drop_caches

# Reduce image cache size
nano projects/experimance/image_server.toml

[cache]
max_size_gb = 2  # Reduce cache size
```

2. **Add swap space**:
```bash
# Create swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow Image Generation

**Problem**: Image generation takes too long

**Diagnosis**:
```bash
# Test generation speed
time uv run python services/image_server/tests/test_speed.py

# Check GPU utilization
nvidia-smi -l 1
```

**Solutions**:

1. **Optimize generation settings**:
```bash
# Use faster models
nano projects/experimance/image_server.toml

[models]
primary = "flux-schnell"  # Faster model
steps = 4  # Fewer steps
```

2. **Use remote generation**:
```bash
# Configure cloud generation
nano projects/experimance/.env

VASTAI_API_KEY=your_key
FAL_API_KEY=your_key
```

## Production Deployment Issues

### Systemd Service Problems

**Problem**: Services won't start with systemd
```
systemctl status experimance-core@experimance
● experimance-core@experimance.service - failed
```

**Diagnosis**:
```bash
# Check service status
sudo systemctl status experimance-core@experimance

# View service logs
sudo journalctl -u experimance-core@experimance -n 50

# Check service file
cat /etc/systemd/system/experimance-core@.service
```

**Solutions**:

1. **Fix service file permissions**:
```bash
# Correct service file ownership
sudo chown root:root /etc/systemd/system/experimance-*.service
sudo chmod 644 /etc/systemd/system/experimance-*.service

# Reload systemd
sudo systemctl daemon-reload
```

2. **Fix user and path issues**:
```bash
# Ensure experimance user exists
sudo useradd -m experimance

# Fix file permissions
sudo chown -R experimance:experimance /home/experimance/experimance
```

3. **Environment issues**:
```bash
# Check environment in service
sudo systemctl edit experimance-core@experimance

# Add environment variables
[Service]
Environment="PATH=/home/experimance/.local/bin:/usr/bin:/bin"
Environment="PROJECT_ENV=experimance"
```

### Email Alerts Not Working

**Problem**: Health monitoring emails not being sent

**Diagnosis**:
```bash
# Test email configuration
./infra/scripts/healthcheck.py --test-email

# Check SMTP settings
python3 -c "import smtplib; print('SMTP available')"
```

**Solutions**:

1. **Configure email settings**:
```bash
# Set environment variables
export ALERT_EMAIL="your-email@gmail.com"
export ALERT_EMAIL_TO="your-email@gmail.com"
export ALERT_EMAIL_PASSWORD="your-app-password"

# Test email sending
./infra/scripts/healthcheck.py --test-email
```

2. **Gmail app password setup**:
- Enable 2-factor authentication on Gmail
- Generate app-specific password
- Use app password instead of regular password

### Web Dashboard Issues

**Problem**: Web dashboard not accessible

**Diagnosis**:
```bash
# Check if dashboard is running
ps aux | grep dashboard

# Test local access
curl http://localhost:8080

# Check firewall
sudo ufw status
```

**Solutions**:

1. **Start dashboard service**:
```bash
# Start dashboard manually
./infra/monitoring/dashboard.py &

# Or add to systemd
sudo systemctl enable experimance-dashboard
sudo systemctl start experimance-dashboard
```

2. **Fix firewall issues**:
```bash
# Allow dashboard port
sudo ufw allow 8080

# Check iptables
sudo iptables -L
```

## Development Issues

### Import Errors

**Problem**: Python modules not found
```
ImportError: No module named 'experimance_common'
```

**Solutions**:

1. **Reinstall in development mode**:
```bash
# From project root
uv sync --dev

# Install common library in editable mode
cd libs/common
uv pip install -e .
```

2. **Check Python path**:
```bash
# Verify Python path
python -c "import sys; print(sys.path)"

# Add to PYTHONPATH if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)/libs/common/src"
```

### Testing Issues

**Problem**: Tests fail or can't run

**Diagnosis**:
```bash
# Run simple test
uv run python utils/tests/simple_test.py

# Check test environment
uv run python utils/tests/check_env.py
```

**Solutions**:

1. **Install test dependencies**:
```bash
uv sync --dev
uv add --dev pytest pytest-asyncio pytest-mock
```

2. **Fix test configuration**:
```bash
# Check pytest configuration
cat pytest.ini

# Run tests with verbose output
uv run -m pytest -v
```

### Project Switching Issues

**Problem**: Wrong project active or project switching not working

**Diagnosis**:
```bash
# Check current project
uv run set-project

# Check project file
cat projects/.project
```

**Solutions**:

1. **Fix project file**:
```bash
# Set project manually
echo "experimance" > projects/.project

# Or use script
uv run set-project experimance
```

2. **Environment override**:
```bash
# Override with environment variable
export PROJECT_ENV=experimance
uv run -m experimance_core
```

## Getting Help

### Collecting Debug Information

When asking for help, collect this information:

```bash
# System information
uname -a
python --version
uv --version

# Service status
./infra/scripts/status.sh experimance

# Recent logs
tail -n 100 logs/*.log > debug_logs.txt

# Configuration
cat projects/experimance/config.toml > debug_config.txt

# Environment
env | grep -E "(PROJECT|EXPERIMANCE|CUDA)" > debug_env.txt
```

### Log Analysis

Enable verbose logging for debugging:

```bash
# Enable debug logging
uv run -m experimance_core --verbose

# Enable performance monitoring
uv run -m experimance_core --performance

# Enable visualization for camera debugging
uv run -m experimance_core --visualize
```

### Community Resources

- **Documentation**: Check service-specific READMEs in `services/*/README.md`
- **Examples**: Look at `utils/examples/` for usage examples
- **Tests**: Review `utils/tests/` for testing utilities
- **Issues**: Create detailed issue reports with logs and system info

### Emergency Recovery

If the system is completely broken:

```bash
# Nuclear option: complete reinstall
rm -rf .venv
rm -rf logs/*
./infra/scripts/deploy.sh install experimance dev

# Reset to known good state
git stash
git checkout main
git pull origin main
```

Remember: Most issues are configuration or environment related. Start with the quick diagnostics and work through the relevant sections systematically.