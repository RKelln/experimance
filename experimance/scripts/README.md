# Scripts

Utility and development scripts for the Experimance project. These are not packaged as console commands — see [Promoted CLI Tools](#promoted-cli-tools) for tools that have graduated to the main package.

## Development

| Script | Description |
|---|---|
| [`dev`](dev) | Start one or more services for local development — [docs](docs/dev.md) |
| [`project`](project) | Show or switch the active project (`./scripts/project fire`) |
| [`set_project.py`](set_project.py) | Underlying project switcher (called by `project`) |
| [`create_new_project.py`](create_new_project.py) | Interactive wizard to scaffold a new project — [docs](docs/create_new_project.md) |
| [`update_pyi_stubs.py`](update_pyi_stubs.py) | Regenerate `.pyi` type stubs after schema/constant changes — [docs](docs/update_pyi_stubs.md) |
| [`validate_schemas.py`](validate_schemas.py) | Validate Python schemas against JSON config files |
| [`check_docs_links.py`](check_docs_links.py) | Check repo-local markdown links (`uv run python scripts/check_docs_links.py`) |
| [`git_identity_remote`](git_identity_remote) | Set git author identity for the current shell (`source scripts/git_identity_remote`) |
| [`git_daily_card.py`](git_daily_card.py) | Generate visual PNG summary cards from daily git commits |
| [`repomix_profiles.sh`](repomix_profiles.sh) | Build repomix outputs for different service scopes |

## Audio

| Script | Description |
|---|---|
| [`audio_cache_manager.py`](audio_cache_manager.py) | Inspect, clean, and manage the audio generation cache — [docs](docs/audio_cache_manager.md) |
| [`audio_recovery.py`](audio_recovery.py) | Diagnose and recover USB audio devices (Yealink, ICUSBAUDIO7D, ReSpeaker) — [docs](docs/audio_recovery.md) |
| [`list_audio_devices.py`](list_audio_devices.py) | List all PyAudio devices with index, channels, and sample rate |
| [`normalize_audio.sh`](normalize_audio.sh) | Normalize audio files from `layers.json` to EBU R128 (−23 LUFS) |
| [`pipewire_multi_sink.py`](pipewire_multi_sink.py) | Create a PipeWire virtual multi-channel sink for multi-speaker setups — [docs](docs/pipewire_multi_sink.md) |
| [`test_multi_channel_audio.py`](test_multi_channel_audio.py) | Calibrate per-channel delays for multi-speaker echo cancellation — [docs](docs/test_multi_channel_audio.md) |
| [`test_osc.py`](test_osc.py) | Test OSC communication with SuperCollider |

## Vision / Detection

| Script | Description |
|---|---|
| [`list_cameras.py`](list_cameras.py) | Discover Reolink cameras on the local network — [docs](docs/list_cameras.md) |
| [`list_webcams.py`](list_webcams.py) | List and test available webcam devices |
| [`test_reolink_camera.py`](test_reolink_camera.py) | Interactive Reolink HTTP API client for presence detection and camera control — [docs](docs/test_reolink_camera.md) |
| [`mock_control.py`](mock_control.py) | Manually control the mock audience detector during testing — [docs](docs/mock_control.md) |
| [`tune_detector.py`](tune_detector.py) | Live HOG/MOG2 detector parameter tuning with webcam feedback — [docs](docs/tune_detector.md) |

## Media / Images

| Script | Description |
|---|---|
| [`images_to_video.py`](images_to_video.py) | Convert timestamped images to a video with crossfade — [docs](docs/images_to_video.md) |
| [`combine_triptychs.py`](combine_triptychs.py) | Stitch triptych tile images into seamless panoramas — [docs](docs/combine_triptychs.md) |
| [`view_images.py`](view_images.py) | Browse generated images with filtering, autoplay, and keyboard nav |
| [`image_watch.sh`](image_watch.sh) | Monitor a remote gallery host for new images and display them locally |
| [`generate_environmental_sound_json.py`](generate_environmental_sound_json.py) | Generate environmental sound prompt JSON from Anthropocene data |
| [`list_anthropocene_elements.py`](list_anthropocene_elements.py) | Extract unique elements from `anthropocene.json` |
| [`zip_media.sh`](zip_media.sh) | Package the `media/` directory into a zip for deployment |

## Promoted CLI Tools

Scripts promoted to the main package — use `uv run` instead:

| Old script | Current command |
|---|---|
| [`vastai_cli.py`](vastai_cli.py) _(deprecated wrapper)_ | `uv run vastai <command>` |
| — | `uv run transcripts list` |

To promote a script:
1. Move logic to `src/experimance/<tool>_cli.py`
2. Add an entry to `[project.scripts]` in `pyproject.toml`
3. Remove or keep the old script as a thin deprecated wrapper
4. Update references here

## Adding New Scripts

1. `chmod +x scripts/your_script.py`
2. Add a `#!/usr/bin/env python3` shebang
3. Add a short docstring with one-line description and usage
4. Add an entry to this README
5. If the script is complex, create `scripts/docs/your_script.md`

Scripts should be runnable with `uv run python scripts/your_script.py`.
