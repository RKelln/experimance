# Getting Started with Experimance

Welcome to Experimance! This guide will walk you through setting up and running your first interactive art installation, from initial setup to creating your own project.

## Table of Contents

1. [What is Experimance?](#what-is-experimance)
2. [Prerequisites](#prerequisites)
3. [Quick Installation](#quick-installation)
4. [Your First Installation](#your-first-installation)
5. [Understanding the System](#understanding-the-system)
6. [Development Setup](#development-setup)
7. [Creating Your Own Project](#creating-your-own-project)
8. [Next Steps](#next-steps)

## What is Experimance?

Experimance is a framework for creating interactive art installations that respond to audience presence and interaction. The system combines:

- **Real-time interaction detection** using depth cameras
- **AI-generated visuals** that respond to audience behavior
- **Spatial audio** that creates immersive soundscapes
- **Conversational AI agents** that can interact with visitors
- **Modular architecture** that allows customization for different projects

### Example Projects

- **Experimance #5**: An interactive sand table where audience interactions drive AI-generated satellite imagery through different eras of human development
- **Feed the Fires**: A community-focused installation where participants share stories with a "fire spirit" that responds with visuals and audio

## Prerequisites

Before you begin, ensure you have:

### System Requirements
- **Operating System**: Ubuntu 20.04+ or macOS 10.15+
- **Python**: 3.11 (required for Intel RealSense compatibility)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for dependencies and media
- **Network**: Internet connection for AI services and package installation

### Hardware (Optional for Development)
- **Depth Camera**: Intel RealSense D415, D435, or D455 (for interaction detection)
- **Webcam**: Any USB webcam (for agent vision)
- **Audio Interface**: Multi-channel audio interface (for spatial audio)

### Software Dependencies
- **Git**: For cloning the repository
- **uv**: Python package manager (will be installed automatically)

## Quick Installation

The fastest way to get started is using our automated installation script:

```bash
# Clone the repository
git clone <repository-url>
cd experimance

# Run the automated installer
./infra/scripts/deploy.sh install experimance dev
```

This script will:
1. Install pyenv and Python 3.11
2. Install uv package manager
3. Create a virtual environment
4. Install all dependencies
5. Set up the development environment
6. Download required model files

### Manual Installation

If you prefer to install manually or the automated script doesn't work for your system:

```bash
# Install uv package manager
curl -sSf https://astral.sh/uv/install.sh | sh

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libssl-dev libusb-1.0-0-dev libsdl2-dev ffmpeg libasound2-dev portaudio19-dev

# Install system dependencies (macOS)
brew install libusb sdl2 ffmpeg portaudio

# Create and activate virtual environment
uv sync

# Verify installation
uv run python -c "import experimance_common; print('Installation successful!')"
```

## Your First Installation

Let's run a basic installation to see the system in action:

### Step 1: Set the Active Project

```bash
# Set the project to 'experimance' (the default example)
uv run set-project experimance

# Verify the project is set
uv run set-project
# Should output: experimance
```

### Step 2: Start the Core Services

Open multiple terminal windows/tabs and run each service:

**Terminal 1 - Core Service (the brain):**
```bash
# Start with mock depth camera for testing without hardware
uv run -m experimance_core \
  --depth-processing-mock-depth-images-path media/images/mocks/depth \
  --presence-always-present
```

**Terminal 2 - Display Service (visuals):**
```bash
uv run -m experimance_display
```

**Terminal 3 - Image Server (AI generation):**
```bash
uv run -m image_server
```

**Terminal 4 - Audio Service (sound):**
```bash
uv run -m experimance_audio
```

### Step 3: Test the System

With all services running, you should see:

1. **Core Service**: Logs showing era progression and interaction detection
2. **Display Service**: A window showing generated images and transitions
3. **Image Server**: Logs of image generation requests and completions
4. **Audio Service**: SuperCollider starting and audio feedback

The system will automatically progress through different "eras" and generate corresponding imagery.

### Step 4: Interact with the System

Since we're using mock presence, the system will behave as if someone is always present. You can:

- Watch the era progression in the core service logs
- See new images being generated and displayed
- Hear audio feedback and environmental sounds

## Understanding the System

### Service Architecture

Experimance uses a microservices architecture with ZeroMQ for communication:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Core     │◄──►│   Display   │    │    Audio    │
│  Service    │    │   Service   │    │   Service   │
└─────────────┘    └─────────────┘    └─────────────┘
       │                                      │
       ▼                                      ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Image     │    │    Agent    │    │   Health    │
│   Server    │    │   Service   │    │   Monitor   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Key Concepts

- **Era System**: The experience progresses through different time periods based on interaction
- **Biomes**: Different environmental themes (forest, desert, urban, etc.)
- **ZMQ Messages**: Services communicate using structured messages
- **Project System**: Multiple installations can share the same codebase with different configurations

### Configuration Files

Each project has its own configuration in `projects/{project_name}/`:

- `.env`: Environment variables (API keys, secrets)
- `config.toml`: Main project configuration
- `{service}.toml`: Service-specific settings
- `constants.py`: Project-specific constants
- `schemas.py`: Custom message schemas

## Development Setup

For development work, you'll want additional tools:

### Install Development Tools

```bash
# Install development dependencies
uv sync --dev

# Install useful tools
uv tool install ruff      # Code formatting and linting
uv tool install pytest    # Testing framework
```

### Running Tests

```bash
# Run all tests
uv run -m pytest

# Run specific service tests
uv run -m pytest services/core/tests/

# Run with coverage
uv run -m pytest --cov=experimance_common

# Test specific functionality
uv run python services/core/tests/test_camera.py --mock
```

### Development Workflow

1. **Make Changes**: Edit code in your preferred editor
2. **Test Changes**: Run relevant tests
3. **Format Code**: `uv run ruff format .`
4. **Check Linting**: `uv run ruff check .`
5. **Test Integration**: Run services together to test interactions

### Debugging Tips

```bash
# Enable verbose logging
uv run -m experimance_core --verbose

# Enable visualization for depth camera
uv run -m experimance_core --visualize

# Test individual components
uv run python services/core/tests/test_camera.py --info
```

## Creating Your Own Project

To create a new installation project:

### Option 1: Interactive Script

```bash
uv run python scripts/create_new_project.py
```

This will guide you through:
- Choosing a project name
- Selecting which services to include
- Copying configuration templates
- Setting up project-specific schemas

### Option 2: Manual Creation

```bash
# Create project directory
mkdir projects/my_project

# Copy template files
cp projects/experimance/.env projects/my_project/
cp projects/experimance/config.toml projects/my_project/
cp projects/experimance/*.toml projects/my_project/

# Edit configuration for your project
nano projects/my_project/config.toml
```

### Customize Your Project

1. **Edit Configuration**: Modify `config.toml` for your installation's needs
2. **Add Custom Schemas**: Define project-specific message types in `schemas.py`
3. **Set Environment Variables**: Configure API keys and secrets in `.env`
4. **Test Your Project**: `uv run set-project my_project && uv run -m experimance_core`

## Next Steps

Now that you have Experimance running, explore these areas:

### Learn More About the System

- **[Architecture Documentation](architecture.md)**: Deep dive into system design
- **[Service Documentation](../services/)**: Detailed guides for each service
- **[Configuration Guide](configuration.md)**: Understanding all configuration options

### Customize Your Installation

- **[Multi-Channel Audio](multi_channel_audio.md)**: Set up spatial audio
- **[Vision System](../services/agent/README_VISION.md)**: Configure cameras and detection
- **[Image Generation](../services/image_server/README.md)**: Customize AI image generation

### Production Deployment

- **[Infrastructure Guide](../infra/README.md)**: Deploy to production systems
- **[Health Monitoring](health_system.md)**: Set up monitoring and alerts
- **[Troubleshooting](troubleshooting.md)**: Common issues and solutions

### Contributing

- **[Development Workflow](development_workflow.md)**: Best practices for contributing
- **[Testing Guide](../utils/tests/README.md)**: Writing and running tests
- **[Code Standards](../libs/common/README_SERVICE.md)**: Coding conventions

## Getting Help

If you run into issues:

1. **Check the logs**: Each service logs to both console and files in `logs/`
2. **Review documentation**: Service-specific READMEs have detailed troubleshooting
3. **Test components**: Use the testing utilities in `utils/tests/`
4. **Ask for help**: Create an issue in the repository with logs and system info

Welcome to the Experimance community! We're excited to see what you create.