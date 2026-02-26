# Experimance Documentation Index

Welcome to the Experimance documentation! This index will help you find the information you need, whether you're just getting started or diving deep into advanced topics.

## Quick Start

New to Experimance? Start here:

- **[Getting Started Guide](getting_started.md)** - Complete walkthrough from installation to your first running installation
- **[FAQ](faq.md)** - Answers to common questions and quick solutions
- **[Main README](../README.md)** - Project overview and basic setup instructions

## Core Documentation

### System Understanding
- **[Architecture Overview](architecture.md)** - System design, service interactions, and communication patterns
- **[Configuration Guide](configuration.md)** - Complete guide to all configuration files and options
- **[Technical Design](technical_design.md)** - Detailed technical specifications and implementation details

### Development and Maintenance
- **[Development Workflow](development_workflow.md)** - Best practices for contributing, testing, and debugging
- **[Troubleshooting Guide](troubleshooting.md)** - Solutions for common issues and problems
- **[FAQ](faq.md)** - Frequently asked questions and quick answers

## Service Documentation

Each service has its own detailed documentation:

### Core Services
- **[Core Service](../services/core/README.md)** - Central orchestration and interaction detection
  - [Depth Camera Setup](../services/core/README_DEPTH.md) - Camera configuration and troubleshooting
- **[Display Service](../services/display/README.md)** - Visual output and rendering
- **[Image Server](../services/image_server/README.md)** - AI image generation
  - [Generator System Guide](../services/image_server/docs/generators.md) - Complete guide to image and audio generation
- **[Agent Service](../services/agent/README.md)** - Conversational AI and vision
  - [Vision System](../services/agent/README_VISION.md) - Camera and detection configuration
- **[Audio Service](../services/audio/README.md)** - Spatial audio and sound design
  - [Surround Sound Setup](../services/audio/docs/surround_sound.md) - Multi-channel audio configuration
  - [Audio Docs Index](../services/audio/docs/index.md) - Index of audio service documentation

### Support Services
- **[Health Monitor](../services/health/README.md)** - System monitoring and alerts
- **[Transition Service](../services/transition/README.md)** - Video transitions between images

## Specialized Topics

### Audio Systems
- **[Multi-Channel Audio](multi_channel_audio.md)** - Complete guide to spatial audio setup
- **[Multi-Channel Quick Reference](multi_channel_quick_reference.md)** - Quick setup guide
- **[Audio Cache Management](../scripts/README_AUDIO_CACHE.md)** - Cache management tools

### Infrastructure and Deployment
- **[Infrastructure Guide](../infra/README.md)** - Production deployment and management
- **[Health System](health_system.md)** - Service monitoring and health management
- **[Logging System](logging_system.md)** - Centralized logging and debugging

### Advanced Features
- **[Image ZMQ Utilities](image_zmq_utilities.md)** - Image processing and ZMQ integration
- **[Shader Effects System](shader_effects_system.md)** - Visual effects and shaders
- **[Smart Plug Matter Control](smart_plug_matter_control.md)** - IoT device integration
- **[Mock Detector Testing](mock_detector_testing.md)** - Testing without hardware

## Development Resources

### Common Libraries
- **[Common Library](../libs/common/README.md)** - Shared utilities and base classes
  - [Service Development](../libs/common/README_SERVICE.md) - Creating new services
  - [Service Testing](../libs/common/README_SERVICE_TESTING.md) - Testing strategies
  - [ZMQ Communication](../libs/common/README_ZMQ.md) - Inter-service communication

### Testing and Utilities
- **[Testing Guide](../utils/tests/README.md)** - Testing utilities and strategies
- **[ZMQ Testing](../utils/tests/README_ZMQ_TESTS.md)** - ZMQ communication testing
- **[Scripts Documentation](../scripts/README.md)** - Utility scripts and tools

### Project Management
- **[Project Management](project_management.md)** - Project organization and workflows

## Installation-Specific Documentation

### Example Installations
- **[FMC 2025-09](installations/FMC_2025-09.md)** - Specific installation documentation

### Hardware Setup
- **[TouchDesigner Integration](../infra/scripts/README_TOUCHDESIGNER.md)** - TouchDesigner workflow
- **[LaunchD Scheduler](../infra/scripts/README_LAUNCHD_SCHEDULER.md)** - macOS service management

## Quick Reference

### Common Commands

```bash
# Project management
uv run set-project experimance          # Switch projects
uv run set-project                      # Check current project

# Service management
uv run -m experimance_core              # Start core service
uv run -m experimance_display           # Start display service
uv run -m image_server                  # Start image server

# Testing and debugging
uv run python utils/tests/simple_test.py    # Basic system test
uv run -m pytest                            # Run all tests
./infra/scripts/status.sh experimance       # Check service status

# Development
uv run ruff format .                    # Format code
uv run ruff check .                     # Check linting
uv sync --dev                          # Install dev dependencies
```

### Configuration Files

```
projects/experimance/
├── .env                    # Environment variables and API keys
├── config.toml            # Main project configuration
├── core.toml              # Core service settings
├── display.toml           # Display service settings
├── image_server.toml      # Image generation settings
├── audio.toml             # Audio service settings
├── agent.toml             # Agent service settings
└── health.toml            # Health monitoring settings
```

### Log Files

```
logs/
├── core.log               # Core service logs
├── display.log            # Display service logs
├── image_server.log       # Image generation logs
├── agent.log              # Agent service logs
├── audio.log              # Audio service logs
├── health.log             # Health monitoring logs
└── healthcheck.log        # Health check results
```

## Getting Help

### Troubleshooting Steps

1. **Check the FAQ** - [FAQ](faq.md) covers most common issues
2. **Review logs** - Check `logs/*.log` for error messages
3. **Run diagnostics** - Use `uv run python utils/tests/simple_test.py`
4. **Check configuration** - Verify settings in `projects/*/config.toml`
5. **Consult troubleshooting guide** - [Troubleshooting Guide](troubleshooting.md)

### Documentation by User Type

#### **New Users**
Start with: [Getting Started](getting_started.md) → [FAQ](faq.md) → [Configuration Guide](configuration.md)

#### **Developers**
Focus on: [Architecture](architecture.md) → [Development Workflow](development_workflow.md) → [Service Documentation](../libs/common/README_SERVICE.md)

#### **System Administrators**
Review: [Infrastructure Guide](../infra/README.md) → [Health System](health_system.md) → [Troubleshooting](troubleshooting.md)

#### **Artists/Creators**
Explore: [Getting Started](getting_started.md) → [Configuration Guide](configuration.md) → [Multi-Channel Audio](multi_channel_audio.md)

### Support Resources

- **Documentation Issues**: If you find errors or gaps in documentation, please create an issue
- **Technical Support**: Include logs, system info, and steps to reproduce when asking for help
- **Feature Requests**: Describe your use case and how the feature would help your installation

## Contributing to Documentation

We welcome documentation improvements! When contributing:

1. **Follow the existing style** - Use clear headings, code examples, and practical guidance
2. **Include examples** - Show both configuration and command examples
3. **Test your instructions** - Verify that your documentation works on a fresh system
4. **Update this index** - Add new documents to the appropriate sections

### Documentation Standards

- Use Markdown format
- Include table of contents for long documents
- Provide code examples with syntax highlighting
- Link to related documentation
- Keep language clear and accessible
- Include troubleshooting sections where appropriate

---

**Need something specific?** Use your browser's search function (Ctrl/Cmd+F) to find keywords across this index, or check the [FAQ](faq.md) for quick answers to common questions.
