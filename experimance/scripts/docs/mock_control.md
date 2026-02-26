# mock_control.py

Controls the mock audience detector during development and testing. Lets you manually trigger presence/absence signals, set person counts, and run automated test scenarios — without needing a real camera.

See `scripts/mock_control.py`.  
Related: [`tune_detector.md`](tune_detector.md)

## Quick Start

```bash
# Signal that someone is present
uv run python scripts/mock_control.py present

# Signal that nobody is present
uv run python scripts/mock_control.py absent

# Set a specific person count
uv run python scripts/mock_control.py count 3

# Show current control directory state
uv run python scripts/mock_control.py status

# Interactive keyboard mode
uv run python scripts/mock_control.py --interactive
uv run python scripts/mock_control.py -i
```

## Automated Test Scenarios

```bash
# Run all automated scenarios
uv run python scripts/mock_control.py --test-scenarios
```

### Available Scenarios

| Scenario | What it tests |
|---|---|
| **Idle Timeout** | Person present → greeting → silence → re-engagement message → person leaves → no more re-engagements |
| **Re-engagement Only** | Focused test: presence → greeting → wait ~30s → verify re-engagement fires |
| **Rapid Changes** | Quick presence on/off cycles to test debouncing |
| **Multiple People** | Count increases 1→3→5→2→1→0 to test person-count handling |
| **Conversation Cycle** | Person arrives → conversation → leaves → cooldown → new person arrives |

Automated scenarios include timestamped step logs. Match the timestamps against agent service logs to verify expected behavior.

## How It Works

The mock controller writes control files to a directory watched by `MockDetector` in `services/agent/`. The agent picks up these files and injects them as if they came from a real camera.

Uses `FileControlHelper` from `services/agent/src/agent/vision/mock_detector.py`.

## Interactive Mode Controls

In `--interactive` mode, use keyboard shortcuts to trigger state changes in real time:
- Press `p` → set present
- Press `a` → set absent
- Press `1`–`9` → set person count
- Press `q` → quit

## Tips

- Run the agent service first, then control it with `mock_control.py` from a second terminal
- Use `--test-scenarios` with `tail -f logs/dev/agent.log` open in another window
- The `status` command shows which control files are currently active
- After any test, call `absent` to ensure a clean state
