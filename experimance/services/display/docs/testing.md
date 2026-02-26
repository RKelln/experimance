# Testing the Display Service

**Files touched:**
- `tests/` — test suite
- `tests/conftest.py` — pytest configuration and shared fixtures
- `tests/mocks.py` — mock objects for window and OpenGL

## Test Types

### Headless tests
Mock the window and rendering systems so no display is required. Suitable for CI/CD.

Relevant files:
- `tests/test_display_headless.py` — comprehensive headless tests
- `tests/test_display_headless_mode.py` — headless mode flag tests
- `tests/test_display_direct.py` — tests via the direct (non-ZMQ) interface
- `tests/test_config.py` — configuration loading and validation
- `tests/test_display_comprehensive.py` — broad coverage tests
- `tests/test_image_renderer_handle_image_ready.py` — image renderer unit tests
- `tests/test_panorama_display.py` — panorama renderer tests
- `tests/test_pyglet_utils.py` — utility function tests

### Window tests
Create real pyglet windows to verify visual rendering. Run manually with a live display.

Relevant files:
- `tests/test_display.py` — basic window creation
- `tests/test_display_service.py` — service window tests
- `tests/test_integration.py` — full message-to-render integration

## Running Tests

### Headless (CI / default)

```bash
# From the display service directory
cd services/display
uv run pytest tests/

# From the project root
uv run pytest services/display/tests/
```

### Window tests (requires a display)

```bash
# Run a specific window test
pytest tests/test_display.py::test_display_service --display

# Run all window tests
pytest tests/ --display
```

### Specific test file

```bash
uv run pytest services/display/tests/test_config.py -v
uv run pytest services/display/tests/test_display_headless.py -v
```

## Writing New Tests

### Headless tests (preferred for CI)

Use the `headless` display config flag and mock out window operations:

```python
import pytest
from experimance_display.config import create_test_display_config

@pytest.fixture
def test_config():
    return create_test_display_config()   # headless=True, debug_overlay=False

def test_my_feature(test_config):
    # Test logic here — no window is created
    ...
```

### Window tests

Mark tests explicitly so they are skipped in headless environments:

```python
@pytest.mark.skip(reason="Requires a live display")
def test_visual_feature():
    # Real window test
    ...
```

## Direct Interface (no ZMQ)

You can exercise the service without any ZMQ infrastructure using `trigger_display_update`:

```python
from experimance_display import DisplayService
from experimance_display.config import create_test_display_config

config = create_test_display_config()
service = DisplayService(config)
await service.start()

service.trigger_display_update("text_overlay", {
    "text_id": "test-1",
    "content": "Hello",
    "speaker": "agent",
    "duration": 5.0
})

await service.stop()
```
