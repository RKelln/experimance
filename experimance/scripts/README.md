# Scripts Directory

This directory contains utility scripts for managing the Experimance project.

## Available Scripts

### `tune_detector.py`

Interactive tool for tuning audience detection parameters with live webcam feedback.

**Usage:**
```bash
# List available detector profiles
uv run python scripts/tune_detector.py --list-profiles

# Start tuning with default profile (indoor_office)
uv run python scripts/tune_detector.py

# Start with a specific profile
uv run python scripts/tune_detector.py --profile gallery_dim

# Use a different camera (if multiple cameras available)
uv run python scripts/tune_detector.py --camera 1

# Enable verbose logging
uv run python scripts/tune_detector.py --verbose
```

**Features:**
- Real-time visualization of HOG person detection (green boxes)
- Motion detection overlay (red contours)
- Interactive parameter adjustment via trackbars
- Live confidence and presence detection feedback
- Save tuned parameters to new detector profiles

**Controls:**
- **q/ESC**: Quit the application
- **s**: Save current settings to a new profile (`{profile_name}_tuned.toml`)
- **r**: Reset to original profile values
- **h**: Toggle HOG detection visualization
- **m**: Toggle motion detection visualization  
- **i**: Toggle parameter info display

**Parameters you can tune:**
- **Detection Scale Factor**: How much to downscale frames (smaller = faster, less accurate)
- **Min Person Height**: Minimum height in pixels for person detection
- **Motion Threshold**: Minimum contour area for motion detection
- **Motion Intensity**: Sensitivity threshold for motion detection
- **Stability**: Majority vote threshold for temporal smoothing
- **HOG Threshold**: Person detection confidence threshold
- **HOG Scale**: Pyramid scale factor for multi-scale detection
- **Win Stride**: HOG window stride (larger = faster, less accurate)
- **MOG2 Var Threshold**: Background subtraction variance threshold
- **MOG2 History**: Number of frames for background model
- **Person Base Confidence**: Base confidence score for person detection
- **Motion Weight**: Weight of motion confidence in final score

**Tips:**
- Start with a default profile that matches your environment
- Adjust parameters gradually and observe the effect on detection
- Test with different lighting conditions and audience scenarios
- Save multiple profiles for different environments
- Use the verbose flag to see detailed detection information

**Requirements:**
- Webcam connected to the system
- OpenCV installed (included in agent dependencies)
- No pipecat dependencies needed (loads only core detection modules)

### `create_new_project.py`
Interactive script to create a new project configuration.

**Usage:**
```bash
uv run python scripts/create_new_project.py
```

**What it does:**
- Prompts for a new project name
- Lets you select which services to include
- Copies service configuration files from existing projects or defaults
- Creates project-specific `constants.py`, `schemas.py`, and `.env` files
- Creates type stubs for the new project
- Updates the global type stubs to include the new project

**Features:**
- Interactive service selection with descriptions
- Choice to copy configs from existing projects or use service defaults
- Creates minimal template files when no source is available
- Automatically updates type stubs for proper IDE support
- Validates project names and prevents conflicts

**Example workflow:**
1. Run the script: `uv run python scripts/create_new_project.py`
2. Enter project name: `my_art_project`
3. Select services: `1,2,5` (core, display, image_server)
4. Choose config source: existing project or defaults
5. Confirm and let it create all the files
6. Customize the generated configs for your project

### `update_pyi_stubs.py`
Updates type stub files for dynamic module loading. Run this after adding schemas or constants to the shared or per project code.

**Usage:**
```bash
uv run python scripts/update_pyi_stubs.py          # Update files
uv run python scripts/update_pyi_stubs.py --diff   # Show diffs first
uv run python scripts/update_pyi_stubs.py --dry-run # Preview only
```

**What it does:**
- Regenerates `libs/common/src/experimance_common/schemas.pyi`
- Regenerates `libs/common/src/experimance_common/constants.pyi`
- Ensures proper type checking for dynamically loaded project-specific modules

## Adding New Scripts

When adding new utility scripts:

1. **Make them executable:** `chmod +x scripts/your_script.py`
2. **Add a shebang:** `#!/usr/bin/env python3`
3. **Add to this README:** Document what it does and how to use it
4. **Use `uv run`:** Scripts should be runnable with `uv run python scripts/your_script.py`
5. **Include error handling:** Use try/catch and provide helpful error messages

## Project Structure Integration

These scripts work with the multi-project architecture:

- **Project configs** go in `projects/{project_name}/{service}.toml`
- **Project schemas** extend base schemas in `projects/{project_name}/schemas.py`
- **Project constants** override base constants in `projects/{project_name}/constants.py`
- **Environment files** set PROJECT_ENV in `projects/{project_name}/.env`

The scripts help maintain this structure automatically.
