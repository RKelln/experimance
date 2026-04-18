# Audio Service Testing

## Overview

Testing focuses on OSC message flow between Python and SuperCollider and on
SuperCollider startup/logging.

For day-to-day troubleshooting, prefer testing against the running audio service
instead of starting standalone SuperCollider in the test harness.

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

Manual mode now defaults to send-only (no `oscdump`) so it is safe against a
live audio service without local port-binding conflicts. Use `--oscdump` only
when you explicitly want local reception verification.

```bash
./services/audio/scripts/test_osc.sh manual --oscdump --message /spacetime --args forest ancient
```

You can also force send-only explicitly with `--send-only`.

By default, the manual and integrated OSC tools read the active audio config and
use its configured OSC send port. You only need `--port` when intentionally
testing a non-default target, and `--config` when you want a config other than
the active project audio config.

Integrated mode (starts SuperCollider and sends a sequence of messages):

```bash
./services/audio/scripts/test_osc.sh integrated
```

Unit tests:

```bash
./services/audio/scripts/test_osc.sh unittest
```

## Interactive CLI testing

You can also send OSC directly with the audio CLI:

```bash
uv run -m experimance_audio.cli
```

The CLI also reads the active audio config by default, including OSC host/port,
config directory, SuperCollider script path, and `sclang` path.

Example commands:

```text
spacetime tropical_island pre_industrial
include river
volume music 0
volume sfx 0
volume environment 0.9
status
```

## Running tests directly

```bash
cd services/audio
uv run tests/test_osc_bridge.py
```

From repo root:

```bash
uv run services/audio/tests/test_osc_bridge.py --integrated --script services/audio/sc_scripts/experimance_audio.scd
```

## Troubleshooting

- If `oscdump` is missing: `sudo apt install liblo-tools`
- If SuperCollider does not start, verify `sclang` is in PATH.
- If integrated mode shows JACK/ALSA failures on `hw:0`, run the audio service
	with a config that sets `supercollider.device` explicitly (for example
	`hw:1,0`) and test with manual mode or CLI.
- `WebEngineContext` headless warnings can appear even when startup succeeds.
	Prioritize debugging ALSA/JACK errors such as `Cannot open PCM device`.
- If the audio service reports a JACK mismatch or repeatedly restarts JACK,
	make sure you are on a version that includes the fixed JACK config parser in
	`audio_service.py`.
- Reliable startup sequence on this machine:
	1. `jack_control start` if JACK is not already up.
	2. `uv run -m experimance_audio`
	3. `uv run -m experimance_audio.cli`

## Recommended test sequence for demos

1. Start audio service with your target config.
2. Send `spacetime` via manual script or CLI.
3. Verify environmental layers are audible.
4. Confirm `music_volume = 0` (and optionally `sfx_volume = 0`) for
	 environment-only runs.

## Files touched

- `services/audio/scripts/test_osc.sh`
- `services/audio/tests/test_osc_bridge.py`
- `services/audio/tests/test_sc_output.py`
