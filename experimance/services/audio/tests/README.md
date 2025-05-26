# Experimance Audio Service Testing

This directory contains test scripts for the Experimance Audio Service.

## OSC Bridge Tests

The `test_osc_bridge.py` script tests the OSC communication from the `OscBridge` class to SuperCollider or any other OSC receiver.

### Requirements

- `liblo-tools` - Provides `oscdump` utility for listening to OSC messages
  - Install: `sudo apt install liblo-tools`
- `python-osc` - Python library for sending OSC messages (should already be installed as a dependency)

### Using the Test Script

The easiest way to run the tests is using the provided test script:

```bash
# From the audio service directory
cd scripts
./test_osc.sh help
```

The script supports three modes:

1. **Manual Mode** - Send individual OSC messages and verify reception:
   ```bash
   ./test_osc.sh manual --message /listening --args true
   ./test_osc.sh manual --message /spacetime --args forest ancient
   ```

2. **Integrated Mode** - Start SuperCollider and run a full test sequence:
   ```bash
   ./test_osc.sh integrated
   ```

3. **Unit Test Mode** - Run automated unit tests:
   ```bash
   ./test_osc.sh unittest
   ```

### Running Tests Directly

You can also run the tests directly using Python:

#### Automated Tests

Run the automated tests with:

```bash
# From the audio service directory
python -m tests.test_osc_bridge
```

This will run all test cases and report results.

#### Manual Testing

For manual testing with specific OSC messages:

```bash
# From the audio service directory
python -m tests.test_osc_bridge --manual --message /spacetime --args forest ancient
python -m tests.test_osc_bridge --manual --message /include --args birds
python -m tests.test_osc_bridge --manual --message /exclude --args water
python -m tests.test_osc_bridge --manual --message /listening --args true
python -m tests.test_osc_bridge --manual --message /speaking --args false
python -m tests.test_osc_bridge --manual --message /transition --args start
python -m tests.test_osc_bridge --manual --message /reload
```

#### Integrated Testing

For integrated testing with SuperCollider:

```bash
# From the audio service directory
python -m tests.test_osc_bridge --integrated
```

### Testing with SuperCollider Manually

To test with SuperCollider manually:

1. Start the test OSC script:
   ```bash
   cd scripts
   ./run_sc.sh -s ../sc_scripts/test_osc.scd
   ```

2. In another terminal, run any of the manual test commands above to send OSC messages.

3. Verify that the SuperCollider script responds to the messages as expected.

### How the Test Works

The test script:

1. Starts an `oscdump` process to listen for OSC messages on port 5568 (default)
2. Creates an instance of the `OscBridge` class 
3. Sends various OSC messages using the `OscBridge` methods
4. Captures the output from `oscdump` to verify messages are being received
5. Validates that the messages contain the expected content

This ensures end-to-end verification that our OSC bridge is correctly sending messages that can be received by OSC-compatible applications.
