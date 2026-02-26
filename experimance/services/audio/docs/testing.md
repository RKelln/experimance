# Audio Service Testing

## Overview

Testing focuses on OSC message flow between Python and SuperCollider and on
SuperCollider startup/logging.

## Environment assumptions

- `liblo-tools` installed for `oscdump`
- SuperCollider installed for integrated tests
- Run from repo root or `services/audio` directory

## Quick start

```bash
./services/audio/scripts/test_osc.sh help
```

## OSC bridge tests

Manual mode:

```bash
./services/audio/scripts/test_osc.sh manual --message /listening --args true
./services/audio/scripts/test_osc.sh manual --message /spacetime --args forest ancient
```

Integrated mode (starts SuperCollider and sends a sequence of messages):

```bash
./services/audio/scripts/test_osc.sh integrated
```

Unit tests:

```bash
./services/audio/scripts/test_osc.sh unittest
```

## Running tests directly

```bash
cd services/audio
python -m tests.test_osc_bridge
```

## Troubleshooting

- If `oscdump` is missing: `sudo apt install liblo-tools`
- If SuperCollider does not start, verify `sclang` is in PATH.

## Files touched

- `services/audio/scripts/test_osc.sh`
- `services/audio/tests/test_osc_bridge.py`
- `services/audio/tests/test_sc_output.py`
