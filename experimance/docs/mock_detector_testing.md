# Mock Audience Detector for Testing

This mock audience detector allows you to test the fire_agent service presence detection without needing a camera. It's particularly useful for:

- Development and testing
- Debugging presence detection logic
- Testing conversation flows
- Integration testing with other services

## Setup

### 1. Configure the Mock Detector

Use the provided test configuration:

```bash
# Copy the mock detector config
cp projects/fire/config_mock_detector.toml projects/fire/config.toml
```

Or manually enable it in your existing config by adding:

```toml
[mock_detector]
enabled = true
control_method = "file"          # "file" or "keyboard"
control_dir = "/tmp/mock_detector"
initial_state = false            # Start with no audience
initial_count = 0

[reolink]
enabled = false                  # Disable real camera
```

### 2. Start the Fire Agent

```bash
# Set the project to fire
scripts/project fire

# Start the agent with mock detector
uv run -m fire_agent
```

The agent will start and show logs indicating the mock detector is active:

```
INFO: Using mock audience detector for testing
INFO: File-based control enabled in: /tmp/mock_detector
INFO: Mock detector initialized successfully
```

## Control Methods

### Interactive Mode (Recommended)

Use the interactive control script for real-time testing:

```bash
# Start interactive mode
uv run python scripts/mock_control.py -i

# Then use single-letter commands:
mock> p        # Signal presence (1 person)
mock> c3       # Set count to 3 people  
mock> a        # Signal absence
mock> s        # Show status
mock> h        # Show help
mock> q        # Quit
```

This is perfect for testing while the agent is running, giving you immediate feedback and easy control.

### Individual Commands

For scripting or one-off commands:

```bash
# Signal presence
python scripts/mock_control.py present

# Signal absence  
python scripts/mock_control.py absent

# Set specific person count
python scripts/mock_control.py count 3

# Check status
python scripts/mock_control.py status
```

### File-Based Control (Advanced)

Or manually create control files:

```bash
# Signal presence
touch /tmp/mock_detector/present

# Signal absence
touch /tmp/mock_detector/absent

# Set person count to 2
touch /tmp/mock_detector/count_2
```

### Testing Scenarios

### Basic Presence Detection
```bash
# Start interactive mode
uv run python scripts/mock_control.py -i

# Test sequence:
mock> a        # Start with no audience
mock> p        # Someone arrives
mock> a        # They leave
```

### Multiple People
```bash
# In interactive mode:
mock> p        # One person arrives  
mock> c3       # More people join (3 total)
mock> c1       # Some leave (1 remaining)
mock> a        # Everyone leaves
```

### Rapid Changes
```bash
# Test quick presence changes in interactive mode:
mock> p        # Present
mock> a        # Absent  
mock> p        # Present again
mock> c5       # 5 people
mock> a        # All leave
```

## What Happens

When you signal presence:
1. The mock detector updates its internal state
2. The fire_agent detects the change in its polling loop
3. The agent starts conversation/greeting flows
4. OSC messages are sent (if enabled)
5. ZMQ messages are published to other services

When you signal absence:
1. The mock detector updates to absent state
2. The agent handles conversation cleanup
3. OSC and ZMQ absence messages are sent

## Debugging

Monitor the agent logs to see detection events:

```bash
# Watch logs in real-time
tail -f logs/fire_agent.log

# Or check specific logs
grep "Mock detection" logs/fire_agent.log
grep "Audience" logs/fire_agent.log
```

## Integration with Other Services

The mock detector publishes the same ZMQ messages as the real detector:

- `AudiencePresent` messages to fire_core
- OSC messages to audio/visual systems
- Transcript messages for conversation flow

This means you can test the entire system integration using the mock detector.

## Switching Back to Real Camera

To switch back to real camera detection:

1. Set `mock_detector.enabled = false` in config
2. Set `reolink.enabled = true` and configure camera settings
3. Restart the agent

## Troubleshooting

### Mock detector not responding
- Check that `/tmp/mock_detector` directory exists and is writable
- Verify the control files are being created (use `mock_control.py status`)
- Check agent logs for any errors

### Agent not starting conversation
- Verify `vision.audience_detection_enabled = true` 
- Check that proactive greeting is enabled
- Monitor logs for detection events

### OSC not working
- Verify `osc.enabled = true` and correct host/port
- Check firewall settings
- Use OSC debugging tools to monitor messages
