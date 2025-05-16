# Experimance Installation Testing Utilities

This directory contains utility scripts for testing and troubleshooting the Experimance package installation.

## Available Test Scripts

### 1. `simple_test.py`
A minimal script that checks basic imports without requiring any extras.

```bash
# Run from the experimance root directory
uv run python utils/tests/simple_test.py
```

This test:
- Verifies Python version and environment
- Checks if experimance and experimance_common can be imported
- Lists installed experimance packages


### 2. `check_env.py`
Checks the Python environment and system configurations.

```bash
# Run from the experimance root directory
uv run python utils/tests/check_env.py
```

This test:
- Verifies Python version compatibility
- Checks for required system libraries
- Displays environment variables and paths
