# Roadmap and Known Gaps — experimance-common

This document tracks near-term work items and known technical debt. Items are loosely prioritized.

## Near-term goals

### API consistency (high priority)

- **Mock handler naming**: `MockPubSubService.set_message_handler` should be renamed to `set_default_handler` (or aliased) to match `PubSubService.set_default_handler`. Currently the mock and the real service use different method names for the same concept, causing confusion when writing tests. See `CODE_REVIEW.md` item 1 and 4.

- **Handler signature documentation**: Clarify in [zmq.md](zmq.md) and [testing.md](testing.md) that:
  - Per-topic handlers (`add_message_handler`): `async def fn(message)`
  - Default/catch-all handlers (`set_default_handler`): `async def fn(topic, message)`

### Test coverage gaps (medium priority)

- **image_utils.py**: Add tests for:
  - `convert_images_to_mp4` — currently uses `os.system` (see code quality item below); needs subprocess-based implementation and tests for missing ffmpeg
  - `save_ndarray_as_tempfile` — no unit tests
  - `cleanup_old_temp_files` — no tests for timing edge cases
  - `choose_image_transport_mode` — add tests for zero-size files, missing files, remote vs. local address detection
- **logger.py**: Test `get_log_directory()` path selection under mocked environment variables (`EXPERIMANCE_ENV=production`, `EUID=0`, `/etc/experimance` present/absent)
- **base_service.py**: Test signal handling (`SIGINT`, `SIGTERM`) in CI/mock contexts; ensure signal setup doesn't interfere with test harness

### Code quality (medium priority)

- **`image_utils.convert_images_to_mp4`**: Replace `os.system("ffmpeg ...")` with `subprocess.run([...], check=True, capture_output=True)` for proper error detection and reporting when `ffmpeg` is missing or fails
- **FIXMEs in `image_utils.py`**: Address format detection edge case in `png_to_base64url`
- **FIXME in `health.py`**: Current design supports only one service per service type; needs design consideration if multiple services of the same type need to run concurrently

### Documentation (low priority)

- Add `docs/image_utils.md` covering image transport modes, base64 encoding, temp-file lifecycle, and `ffmpeg` dependency
- Add `docs/health.md` covering `HealthStatus`, `HealthCheck`, `ServiceHealth`, file-based IPC protocol, and the one-service-per-type limitation
- Add `docs/osc.md` covering the OSC bridge to SuperCollider (ports, client API, config)

## Known gaps

| Area | Gap | Impact |
|------|-----|--------|
| Mock API | `MockPubSubService.set_message_handler` vs. `set_default_handler` naming | Medium — confusing when porting tests |
| `image_utils` | `os.system` call; no error propagation if ffmpeg missing | Medium — silent failure in production |
| `health.py` | One service per type limit | Low — not currently an issue, but noted |
| `check_docs_links.py` | Only scans `services/`; misses `libs/` docs | Low — manual review needed for lib docs |
| Test coverage | Image utils, signal handling, logger path | Medium |

## Completed items

- Migrated documentation from root-level `README_*.md` files into `docs/` (2026-02)
- Composition-based ZMQ architecture replacing inheritance-based approach
- `BaseServiceConfig.from_overrides()` standardized config loading pattern
- `active_service()` context manager for reliable test lifecycle management
