# Frequently Asked Questions (FAQ)

This document answers common questions about Experimance, from basic usage to advanced configuration and troubleshooting.

## Table of Contents

1. [General Questions](#general-questions)
2. [Installation and Setup](#installation-and-setup)
3. [Hardware Requirements](#hardware-requirements)
4. [Configuration](#configuration)
5. [Development](#development)
6. [Performance](#performance)
7. [Troubleshooting](#troubleshooting)
8. [Production Deployment](#production-deployment)

## General Questions

### What is Experimance?

Experimance is a framework for creating interactive art installations that respond to audience presence and interaction. It combines real-time interaction detection, AI-generated visuals, spatial audio, and conversational AI to create immersive experiences.

### What kind of installations can I create with Experimance?

Experimance is designed for interactive art installations such as:
- **Interactive sand tables** with projected visuals
- **Responsive wall projections** that react to movement
- **Conversational art pieces** with AI agents
- **Environmental installations** with spatial audio
- **Multi-modal experiences** combining vision, audio, and interaction

### Do I need programming experience to use Experimance?

Basic programming knowledge is helpful but not required for simple installations. The system provides:
- **Configuration-based setup** for most customization
- **Pre-built projects** you can adapt
- **Comprehensive documentation** with examples
- **Interactive setup scripts** for new projects

For advanced customization, Python knowledge is beneficial.

### Is Experimance open source?

Yes, Experimance is open source. You can modify, extend, and distribute it according to the license terms.

## Installation and Setup

### What operating systems are supported?

- **Ubuntu 24.04+** (recommended for production)
- **macOS 10.15+** (good for development)
- **Other Linux distributions** (may require additional setup)
- **Windows** (not officially supported, but may work with WSL)

### How long does installation take?

- **Quick install**: 15-30 minutes with the automated script
- **Manual install**: 1-2 hours depending on your system
- **Full development setup**: 2-3 hours including all tools and dependencies

### Can I run Experimance without special hardware?

Yes! Experimance includes mock modes for development:

```bash
# Run without depth camera
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present

# Run without webcam
uv run -m experimance_agent \
  --no-vision-webcam_enabled \
  --no-vision-audience_detection_enabled
```

### Why does installation fail with "Python 3.11 required"?

Experimance requires Python 3.11 specifically due to Intel RealSense camera compatibility. Install the correct version:

```bash
# Using pyenv (recommended)
pyenv install 3.11.9
pyenv local 3.11.9

# Verify version
python --version  # Should show 3.11.x
```

### What if I don't have a depth camera?

You can use mock depth data for development and testing:

1. **Create mock depth images**:
```bash
mkdir -p media/images/mocks/depth
# Add grayscale PNG files representing depth data
```

2. **Use mock mode**:
```bash
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth
```

## Hardware Requirements

### What are the minimum system requirements?

**Minimum**:
- CPU: Quad-core 2.0GHz
- RAM: 8GB
- Storage: 10GB free space
- GPU: Integrated graphics (for basic functionality)

**Recommended**:
- CPU: 8-core 3.0GHz+
- RAM: 16GB+
- Storage: 50GB+ SSD
- GPU: Dedicated GPU with 4GB+ VRAM

### Which depth cameras are supported?

**Officially supported**:
- Intel RealSense D415
- Intel RealSense D435
- Intel RealSense D455

**May work with modifications**:
- Other Intel RealSense models
- Azure Kinect (requires custom integration)
- Orbbec cameras (requires custom integration)

### Do I need a powerful GPU for image generation?

It depends on your needs:

**Local generation**:
- **NVIDIA GPU with 8GB+ VRAM** for best performance
- **Apple Silicon Macs** work well with MPS backend
- **CPU-only** is possible but very slow

**Cloud generation**:
- Use services like Vast.ai, Fal.ai, or Runware (Vast.ai is already set up)
- Minimal local GPU requirements
- Higher latency but more powerful models

### What audio hardware do I need?

**Basic setup**:
- Built-in audio output works for testing
- USB audio interface for better quality

**Multi-channel spatial audio**:
- 8+ channel audio interface
- Multiple speakers positioned around the space
- Low-latency audio interface (< 10ms)

## Configuration

### How do I switch between projects?

```bash
# List available projects
ls projects/

# Switch to a project
uv run set-project fire

# Check current project
uv run set-project
```

### How do I customize the experience behavior?

Edit the project configuration file:

```bash
# Edit main project config
nano projects/experimance/config.toml

# Key settings to adjust:
[experience]
era_progression_speed = 1.0      # How fast eras change
interaction_sensitivity = 0.7    # How sensitive to interaction
idle_timeout_minutes = 5         # When to start idle behavior
```

### How do I add my own AI models?

1. **For image generation**, edit `image_server.toml`:
```toml
[models]
flux_model = "your-custom-model"
sd_model = "your-stable-diffusion-model"
lora_models = ["your-lora-1", "your-lora-2"]
```

2. **For conversation**, edit `agent.toml`:
```toml
[conversation]
llm_provider = "openai"  # or "anthropic", "local"
model = "gpt-4"
system_prompt = "Your custom system prompt..."
```

### How do I configure API keys?

Add them to your project's `.env` file:

```bash
# Edit environment file
nano projects/experimance/.env

# Add your keys
OPENAI_API_KEY=your_key_here
ELEVENLABS_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here
```

### Can I use different AI providers?

Yes! Experimance supports multiple providers:

**LLM providers**: OpenAI, Anthropic, local models
**TTS providers**: ElevenLabs, OpenAI, Azure, local TTS
**STT providers**: Deepgram, OpenAI Whisper, Azure
**Image providers**: Local models, Vast.ai, Fal.ai, Runware

## Development

### How do I create a new project?

Use the interactive script:

```bash
uv run python scripts/create_new_project.py
```

Or manually:

```bash
# Create project directory
mkdir projects/my_project

# Copy template files
cp -r projects/experimance/* projects/my_project/

# Edit configuration
nano projects/my_project/config.toml
```

### How do I add a new service?

1. **Create service structure**:
```bash
mkdir -p services/my_service/src/my_service
mkdir -p services/my_service/tests
```

2. **Follow the service template** in `libs/common/README_SERVICE.md`

3. **Add to systemd** (for production):
```bash
cp infra/systemd/experimance-core@.service \
   infra/systemd/experimance-my-service@.service
```

### How do I test my changes?

```bash
# Run unit tests
uv run -m pytest

# Test specific service
uv run -m pytest services/core/tests/

# Test with mock hardware
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present
```

### How do I debug service communication?

```bash
# Test ZMQ communication
uv run python utils/tests/test_zmq_utils.py

# Monitor ZMQ messages
uv run python utils/examples/zmq_monitor.py

# Check port usage
netstat -ln | grep 555
```

## Performance

### Why is image generation slow?

**Common causes**:
- Using CPU instead of GPU
- Large image sizes
- Too many generation steps
- Insufficient VRAM

**Solutions**:
```bash
# Check GPU usage
nvidia-smi

# Reduce generation settings
nano projects/experimance/image_server.toml
# Set smaller resolution, fewer steps

# Use cloud generation
# Configure Vast.ai or Fal.ai in .env
```

### Why is the system using too much CPU?

**Common causes**:
- High depth camera FPS
- Large depth camera resolution
- Multiple services on one machine

**Solutions**:
```bash
# Reduce depth processing load
nano projects/experimance/core.toml

[depth_processing]
fps = 15  # Reduce from 30
output_resolution = [320, 240]  # Reduce resolution
```

### How do I optimize for my hardware?

**For powerful hardware**:
```toml
[generation]
batch_size = 4
steps = 50

[depth_processing]
fps = 30
output_resolution = [640, 480]
```

**For limited hardware**:
```toml
[generation]
batch_size = 1
steps = 10

[depth_processing]
fps = 10
output_resolution = [320, 240]
```

### Can I run services on different machines?

Yes! Configure ZMQ to use network addresses:

```toml
# On machine running core service
[zmq]
bind_address = "0.0.0.0"  # Listen on all interfaces

# On other machines
[zmq]
core_address = "192.168.1.100"  # IP of core machine
```

## Troubleshooting

### Services won't start - what should I check?

1. **Check Python version**: `python --version` (should be 3.11.x)
2. **Check virtual environment**: `which python` (should be in .venv)
3. **Check dependencies**: `uv run python utils/tests/simple_test.py`
4. **Check ports**: `netstat -ln | grep 555`
5. **Check logs**: `tail -f logs/*.log`

### The display shows a black screen

**Common causes**:
- Graphics driver issues
- Wrong monitor configuration
- OpenGL compatibility problems

**Solutions**:
```bash
# Test display
uv run python services/display/tests/test_display.py

# Check OpenGL
glxinfo | grep "OpenGL version"

# Try windowed mode
uv run -m experimance_display --windowed
```

### No audio output

**Common causes**:
- SuperCollider not installed
- Wrong audio device
- Audio permissions

**Solutions**:
```bash
# Install SuperCollider
sudo apt install supercollider  # Ubuntu
brew install supercollider      # macOS

# Check audio devices
aplay -l  # Linux
system_profiler SPAudioDataType  # macOS

# Test audio
uv run python services/audio/tests/test_audio.py
```

### Camera not detected

**Common causes**:
- Missing drivers
- USB power issues
- Permission problems

**Solutions**:
```bash
# Install RealSense drivers
sudo apt install librealsense2-*

# Check camera detection
rs-enumerate-devices

# Check USB connection
lsusb | grep Intel

# Fix permissions
sudo usermod -a -G dialout $USER
```

### Services can't communicate

**Common causes**:
- Port conflicts
- Firewall blocking ports
- Services starting in wrong order

**Solutions**:
```bash
# Check port usage
sudo lsof -i :5555

# Kill conflicting processes
sudo pkill -f experimance

# Start services in order
uv run -m experimance_core &
sleep 5
uv run -m experimance_display &
```

## Production Deployment

### How do I deploy to a production system?

Use the automated deployment script:

```bash
# Install and start all services
sudo ./infra/scripts/deploy.sh experimance install
sudo ./infra/scripts/deploy.sh experimance start

# Check status
sudo ./infra/scripts/deploy.sh experimance status
```

### How do I monitor the system in production?

Experimance includes comprehensive monitoring:

1. **Web dashboard**: `http://installation-ip:8080`
2. **Email alerts**: Configure in `.env`
3. **Log monitoring**: Automatic log rotation and aggregation
4. **Health checks**: Automated service health monitoring

### How do I update a production system?

```bash
# Update to latest version
sudo ./infra/scripts/deploy.sh experimance update

# Rollback if needed
sudo ./infra/scripts/deploy.sh experimance rollback
```

### How do I backup the system?

```bash
# Backup configuration and media
tar -czf experimance-backup-$(date +%Y%m%d).tar.gz \
  projects/ \
  media/images/generated/ \
  logs/

# Backup to cloud storage
rsync -av experimance-backup-*.tar.gz user@backup-server:/backups/
```

### What about security?

**Basic security measures**:
- Use strong passwords for dashboard access
- Keep API keys secure in `.env` files
- Bind ZMQ to localhost only unless needed
- Regular system updates
- Monitor logs for suspicious activity

**Advanced security**:
- Use SSL/TLS for web dashboard
- Encrypt ZMQ communication for remote connections
- Set up firewall rules
- Use VPN for remote access

### How do I scale for larger installations?

**Horizontal scaling**:
- Run image generation on separate GPU machines
- Use multiple display nodes for large projections
- Distribute audio processing across multiple machines

**Performance optimization**:
- Use SSD storage for faster I/O
- Increase system RAM for larger caches
- Use dedicated network for service communication
- Optimize configuration for your hardware

### What if something goes wrong during an exhibition?

**Emergency procedures**:

1. **Check web dashboard**: `http://installation-ip:8080`
2. **Restart services**: `sudo systemctl restart experimance-*`
3. **Check logs**: `sudo journalctl -u experimance-core@experimance -n 50`
4. **Fallback mode**: Use mock hardware if sensors fail
5. **Contact support**: Have logs and system info ready

**Prevention**:
- Test thoroughly before exhibitions
- Have backup hardware ready
- Monitor system health continuously
- Train staff on basic troubleshooting

---

## Still Need Help?

If your question isn't answered here:

1. **Check the documentation**: Service-specific READMEs have detailed information
2. **Search the logs**: Most issues leave traces in the log files
3. **Try the troubleshooting guide**: `docs/troubleshooting.md` has detailed solutions
4. **Create an issue**: Include logs, system info, and steps to reproduce

Remember: Most issues are configuration-related and can be solved by carefully checking the configuration files and logs.