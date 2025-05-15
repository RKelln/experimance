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

### 2. `test_imports.py`
A more comprehensive test that checks imports of various modules and subpackages.

```bash
# Run from the experimance root directory
uv run python utils/tests/test_imports.py
```

This test:
- Displays Python path and environment details
- Attempts to import experimance and its submodules
- Verifies ZeroMQ communication utilities
- Reports success or failure for each import

### 3. `check_env.py`
Checks the Python environment and system configurations.

```bash
# Run from the experimance root directory
uv run python utils/tests/check_env.py
```

This test:
- Verifies Python version compatibility
- Checks for required system libraries
- Displays environment variables and paths

### 4. `check_install.py`
Verifies that all required dependencies are correctly installed.

```bash
# Run from the experimance root directory
uv run python utils/tests/check_install.py
```

This test:
- Checks that all required packages are installed
- Verifies version compatibility
- Tests imports of critical dependencies

## Troubleshooting Checklist

If you encounter installation issues, try these steps in order:

1. Run `simple_test.py` to check basic imports
2. Run `check_env.py` to verify your environment
3. Run `check_install.py` to verify dependencies
4. Run `test_imports.py` for a full import test

## Common Issues and Solutions

### Import Errors

If you see errors like `ModuleNotFoundError: No module named 'experimance'`:

1. Ensure you're in the experimance virtual environment:
   ```bash
   source .venv/bin/activate
   ```

2. Check that the package is installed in development mode:
   ```bash
   uv pip list | grep experimance
   ```

3. Try reinstalling the package:
   ```bash
   uv pip install -e .
   ```

### Missing ZMQ or SDL2

If you encounter errors related to ZeroMQ or SDL2:

1. Ensure system dependencies are installed:
   ```bash
   sudo apt-get install libzmq3-dev libsdl2-dev
   ```

2. Reinstall the Python bindings:
   ```bash
   uv pip install --force-reinstall pyzmq pysdl2 pysdl2-dll
   ```

### Path Issues

If Python can't find packages even though they're installed:

1. Try using the bootstrap module:
   ```python
   import bootstrap
   import experimance
   ```

2. Create or update the .pth file:
   ```bash
   uv run python utils/tests/fix_imports.py
   ```
