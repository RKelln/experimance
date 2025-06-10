# Display Service Tests

This directory contains tests for the Experimance Display Service.

## Types of Tests

1. **Headless Tests**: These tests mock the window and rendering systems to run without creating actual windows. 
   They are designed to test service logic and are suitable for CI/CD environments.

2. **Window Tests**: These tests create real Pyglet windows to test visual rendering. 
   They are useful for visual inspection but should be run manually.

## Running Tests

### Headless Tests

Run all tests that don't require a display:

```bash
# From the display service directory
pytest tests/

# From the project root
pytest services/display/tests/
```

### Window Tests

Run tests that create real windows (requires a display):

```bash
# Run a specific window test
pytest tests/test_display.py::test_display_service --display

# Run all window tests
pytest tests/ --display
```

### Specific Test Files

- `test_display_headless.py`: Comprehensive headless tests
- `test_display_direct.py`: Headless tests for the direct interface
- `test_config.py`: Configuration loading tests
- `test_display.py`: Simple interactive window test (requires display)
- `test_display_service.py`: More comprehensive window test (requires display)
- `test_integration.py`: Full integration tests with real rendering (requires display)

## Creating New Tests

When creating new tests that involve window creation or rendering:

1. Mark the test function with `@pytest.mark.skip(reason="...")` 
2. Import the test file only when explicitly running it to avoid display dependency
3. Consider providing both headless (mocked) and window (real) versions

For example:

```python
# For window tests
@pytest.mark.skip(reason="Test requires a display")
def test_visual_feature():
    # Real window test here...
```
