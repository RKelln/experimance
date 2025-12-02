# Code Review: services/core

Generated from the `repomix-service-core.xml` summary of the `services/core` module (review date: 2025-12-02).

Purpose: Confirm documentation (READMEs and guides) reflect the code, identify mismatches, and provide a prioritized, actionable list of code quality issues, missing tests, and recommended fixes.

Reviewed artifacts: source under `src/experimance_core` and `fire_core`, `tests/` in `services/core`, and service-level docs (`README.md`, `README_DEPTH.md`, `DESIGN.md`, `NEW_CORE_GUIDE.md`).

---

## Executive summary

- The `core` service is well-structured and covers its responsibilities: camera acquisition, depth processing, presence/state management, ZMQ-based event publication, and worker coordination. The shared ZMQ components in `libs/common` are used consistently.
- The READMEs and depth docs are comprehensive and helpful. The documentation can be improved by highlighting mock/CI modes, detailing the camera error/recovery configuration, and listing the most-used CLI flags for quick discovery.
- The code is mature but shows several high-level reliability/maintainability opportunities and test gaps (summarized below). Addressing these will reduce hard-to-debug runtime issues and improve testability and onboarding.

---

## Documentation mismatches and suggested README updates (High priority)

1) Clarify mock-dev startup and Full Mock Mode
   - Finding: `README.md` references `--depth-processing-mock-depth-images-path` and `--presence-always-present`, but it lacks a concise guide showing how to provide mock images and how to run the core service with mocked ZMQ for local-only or CI runs.
   - Recommendation: Add a `Full Mock Mode` section showing how to start the core with mock depth images, always-present mode, and how to use `libs/common` helper functions to create a mock ZMQ service in tests (e.g., `create_mock_zmq_service()`).

2) Document camera error/recovery behavior explicitly
   - Finding: `realsense_camera.py` implements retries, reset, and fallback config loading behavior; this is useful but not well documented in the main README.
   - Recommendation: Add a short camera recovery section describing `camera.max_retries`, `camera.retry_delay`, `camera.max_retry_delay`, `camera.aggressive_reset`, and `camera.skip_advanced_config`, and link to `scripts/debug_camera.py` as a helpful troubleshooting script.

3) CLI flags index
   - Finding: CLI args are auto-generated from Pydantic config models; developers can discover flags with `--help`, but a short index of commonly-used flags would speed onboarding.
   - Recommendation: Add a mini-index showing `--visualize`, `--presence-always-present`, `--depth-processing-mock-depth-images-path`, `--camera-debug-depth`, and `--verbose`.

---

## Code quality findings (priority order)

The following findings are prioritized by the risk they pose to reliability and the effort required to remediate.

### High

1) Blocking calls in async contexts
   - Symptom: Some runtime code paths use `subprocess.run()` and other blocking calls in modules that are imported or accessible by the service at runtime (e.g., in `audio_utils.py`, `camera_utils.py`), which can block the event loop.
   - Fix: Run blocking operations in threads (`asyncio.to_thread`) or switch to `asyncio.create_subprocess_exec`/async APIs. Add a test to assert no call blocks the event loop for more than a configured time threshold.

2) Broad exception handling that hides failures
   - Symptom: Use of `except Exception: pass` and overly broad `except Exception:` handlers occurs across multiple modules.
   - Fix: Narrow exception handlers to the expected exception types, ensure errors are logged with context, and re-raise or surface errors where necessary. Add tests asserting that critical failures are logged and/or propagate as designed.

3) External resources and graceful fallback
   - Symptom: Camera/hardware dependencies are used by default in runtime paths — while the system has reset and retry behaviors, we should make mock mode the simpler path for CI and dev.
   - Fix: Introduce an explicit `mode` flag (`device | mock` or `auto` with `fail_to_mock`) and make the fallback behavior deterministic. Document this in README and add tests for both modes.

### Medium

4) Event-loop and timestamp APIs
   - Symptom: Code uses `asyncio.get_event_loop().time()` and `asyncio.get_event_loop()` within coroutines; `time.monotonic()` or `asyncio.get_running_loop()` is preferred.
   - Fix: Replace `get_event_loop()` time usage with `time.monotonic()` for elapsed/interval calculations and use `get_running_loop()` only when necessary within coroutines. Add a CI linter/grep check to detect deprecated patterns.

5) Sleep loops and cooperative cancellation
   - Symptom: Background loops use `await asyncio.sleep()` without structured cancellation checks, leading to slow shutdowns.
   - Fix: Introduce `asyncio.Event()` flags or proper task cancellation and use structured task tracking on startup/stop. Add tests for prompt task cancellation (fast shutdown tests).

### Low

6) Global state and singletons affecting test isolation
   - Symptom: Singleton-like objects (e.g., `mock_message_bus`) are used in mocks and tests which can leak state between tests.
   - Fix: Introduce factory functions and pass dependencies into constructors (dependency injection) for better isolation.

7) Duplicate logic in prototype vs production modules
   - Symptom: Prototype files (`depth_finder_prototype.py`) and production files (`depth_processor.py`) contain similar logic that increases maintenance burden.
   - Fix: Extract common code to a shared library API or base class and maintain distinct test coverage for prototype-specific behavior.

8) Logging and observability
   - Symptom: Coverage of health checks is present, but the README and `BaseService.health_check()` should clearly show which checks are required for production monitoring.
   - Fix: Ensure `health_check()` consistently reports component health (camera state, ZMQ connectivity, worker statuses) and add a README section describing how health data is reported and monitored.

---

## Tests and coverage: missing or weak areas

Covered areas observed:
- Good unit test coverage for presence management, smoothing algorithms, and some worker interactions.

Gaps & recommended tests (actionable):

1) Depth processor edge cases (`test_depth_processor_edge_conditions.py`) — High
   - Focus: No frames, extremely noisy frames, abrupt changes in frame sizes, and queue-saturation behavior (verify stable operations under abnormal inputs).

2) Presence manager concurrency (`test_presence_concurrency.py`) — Medium
   - Focus: Concurrent frame injection, verify correct hysteresis and stable state transitions without races.

3) Mocked pipeline smoke test (`test_pipeline_mocked.py`) — High
   - Focus: Use `MockDepthProcessor` + `MockPubSubService` to validate: camera (mock) -> processing -> publish events for presence & CHANGEMAP messages. Ensure rates and payload shapes are in spec. Use `mock_environment` and `wait_for_messages` helpers.

4) Prompt generator deterministic outputs (`test_prompt_generator_outputs.py`) — Medium
   - Focus: Verify seeded prompt generation produces deterministic templates and negative prompts.

5) Error path tests (High)
   - Focus: Validate `CameraState.ERROR` after repeated failures, `realsense_camera._reset_camera()` returns `False` on failure paths, and that handlers log and propagate errors.

6) CI smoke tests and mocks
   - Focus: Add a `ci/smoke-core` test job that runs the core service with `MockDepthProcessor` and `MockPubSubService` in headless mode to assert essential pipeline behavior on PRs.

---

## Suggested README updates (conservative edits)

1) Add “Full Mock Mode” and CI-friendly run examples to `README.md` and `README_DEPTH.md`.
2) Add camera retry and reset behavior summary to core README and depth README.
3) Add a short CLI flags index for quick dev access to commonly used startup options.
4) Add a `DEVELOPMENT.md` snippet showing how to run with mocks and how to run the smoke tests locally.

---

## Implementation Plan (Prioritized)

1) Small docs updates (Low risk):
   - Add `Full Mock Mode` and camera retry sections.
   - Add a `DEVELOPMENT.md` snippet.

2) High priority fixes (Blocking issues):
   - Replace blocking `subprocess.run` in runtime modules with async-safe `asyncio.to_thread` or use async subprocess APIs.
   - Replace `asyncio.get_event_loop().time()` with `time.monotonic()` and `get_running_loop()` usage where appropriate.

3) Tests: Add critical unit/edge tests:
   - `test_realsense_camera_mocked.py`, `test_depth_processor_edge_conditions.py`, `test_pipeline_mocked.py`.

4) CI & static checks:
   - Add grep/lint rules to detect `subprocess.run` usage in runtime modules and `asyncio.get_event_loop` usages.
   - Add CI smoke test job that runs core with `MockDepthProcessor` and mocked ZMQ.

5) Refactor & future work:
   - Extract shared depth processing helpers for prototype vs production modules.
   - Introduce `mode` (mock | device) for deterministic CI behavior.

---

## Observations and follow-ups

- The code and docs are generally in good shape with several high-value quick wins that would improve reliability and developer on-boarding.
- After addressing the top-priority fixes, consider adding a small `ci/smoke-core` job that runs the core service in mock mode to catch integration issues early.

---
