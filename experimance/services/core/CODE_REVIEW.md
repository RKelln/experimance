# Code Review: services/core

Generated from the `repomix-service-core.xml` summary of the `services/core` module (review date: 2025-12-02).

Purpose: verify documentation (READMEs and guides) is consistent with the code, identify mismatches, and provide a prioritized, actionable list of code quality issues, missing tests, and recommended fixes.

Reviewed artifacts: source under `src/experimance_core` and `fire_core`, `tests/` in `services/core`, and service-level docs (`README.md`, `README_DEPTH.md`, `DESIGN.md`, `NEW_CORE_GUIDE.md`).

---

**Executive summary**

- The `core` service is well-structured into camera acquisition, depth processing, visualization, presence detection, and prompt generation subsystems. Many focused unit tests exist; integration points are partially mocked for CI.
- Documentation exists but some doc pages assume the developer has hardware configured; adding short guidance for running in a fully mocked/dev mode will improve on-boarding.
- The code contains several maintainability and reliability opportunities (detailed below). Implementing the suggestions will improve robustness, testability, and developer onboarding.

---

**Documentation: issues & suggested edits**

Note: the repository root `README.md` already contains detailed Quick Install, Manual Installation, and `Running Services` sections (including `uv` commands and environment-variable examples). To avoid duplication, service-level docs should prefer short per-service pointers to the root README and only include small, service-specific snippets (env vars, hardware caveats, mocked-run examples).

1) Per-service Quickstart (medium priority)
   - Problem: While the root `README.md` covers global installation and run patterns, `services/core/README.md` lacks a concise per-service Quickstart that points to the root instructions and lists the small commands developers need for working on the core service specifically (set `PROJECT_ENV`, run the service, run service tests, run in mock mode).
   - Suggested per-service snippet to add near the top of `services/core/README.md`:

```
See the repository Quick Install and Run instructions in the project `README.md` for full environment setup.

Quickstart (service-local)

export PROJECT_ENV=experimance
uv run -m experimance_core
pytest services/core/tests -q

# To run without hardware, use the mock depth processor
export USE_MOCK_DEPTH=1
uv run -m experimance_core
```

2) Environment variables and example `.env` (high)
   - Problem: code references `PROJECT_ENV` and other environment settings; there is useful guidance in the root README, but `services/core` should include a minimal `.env.example` or a pointer to `projects/{project}/.env` for quick copying.
   - Suggested `.env.example` content (add at `services/core/.env.example` or top-level):

```
# PROJECT_ENV selects which project config is used (maps to projects/<name>/)
PROJECT_ENV=experimance
# DEBUG or log level
LOG_LEVEL=INFO
```

3) Hardware / system prerequisites (medium)
   - Problem: depth and camera modules require system-level dependencies (librealsense, udev rules). `README_DEPTH.md` contains details but should begin with a short bullet list of system packages and links. Also add a clear note in `services/core/README.md` about using `mock_depth_processor.py` for development without hardware.

4) Module map (low)
   - Add a short table mapping major modules to responsibilities (e.g., `realsense_camera.py` — Realsense device wrapper; `depth_processor.py` — filtering/smoothing/queueing for depth frames; `presence.py` — presence detection heuristics; `prompter.py`/`prompt_generator.py` — LLM prompt assembly).

   5) Outdated module references (high)
      - Problem: `README_DEPTH.md` and some scripts still refer to `robust_camera.py` and a `dataclass CameraConfig` which no longer exist. The current implementation uses `realsense_camera.py` as a hardware wrapper, a `DepthProcessor` in `depth_processor.py`, and `CameraConfig` is a Pydantic `BaseModel` defined in `config.py`.
      - Recommendation: Update documentation to reference current module names and types.

---

**Concrete README snippets to paste**

- Quickstart snippet: see the Quickstart block above.
- `.env.example` snippet: see the snippet above.
- Run tests snippet: add under a "Development" heading the `pytest` command and note hardware vs mock test markers (for example, instruct developers how to run only unit tests or integration tests).

---

**Code quality findings (code smells & improvement opportunities)**

I inspected module names and tests via the repomix summary and list below issues in priority order.

1) Async vs blocking code (High)
   - Symptom: modules that manage I/O (camera access, file writes) may call blocking system APIs or use blocking subprocess calls from within code that otherwise uses asyncio. This can starve loops if run in an async context.
   - Recommendation: ensure all long-running or blocking operations are run in threads (`asyncio.to_thread(...)`) or use non-blocking equivalents. Add tests and lints to detect blocking usage in async functions.

2) Event loop usage (Medium)
   - Symptom: code uses `asyncio.get_event_loop().time()` or `asyncio.get_event_loop()` directly. In modern asyncio, prefer `asyncio.get_running_loop()` when inside a coroutine, or `time.monotonic()` for timestamps.
   - Recommendation: replace `get_event_loop()` uses inside coroutines with `get_running_loop()` and prefer `time.monotonic()` for elapsed time.

3) Sleep loops and cancellation (Medium)
   - Symptom: background loops that call `await asyncio.sleep()` appear in base services; check that shutdown cancels them promptly and that loops use an `asyncio.Event()`/boolean flag for cooperative shutdown.
   - Recommendation: ensure loops check a cancellation token and register tasks with `add_done_callback`/structured task management in service base class.

4) Broad exception handlers (High)
   - Symptom: some modules swallow exceptions or use `except Exception: pass` in places that could hide bugs (in worker loops, handlers). This is a reliability risk.
   - Recommendation: narrow exception catches, log at appropriate levels with context, and consider re-raising critical exceptions. Add unit tests to validate error paths.

5) Global state and singletons (Low→Medium)
   - Symptom: global module-level variables or implicit singletons make unit-testing harder.
   - Recommendation: pass dependencies through constructors where possible (dependency injection) and add factory functions that return test-friendly mocks.

6) Duplication of logic between prototypes and production modules (Low)
   - Symptom: `depth_finder_prototype.py` alongside `depth_processor.py` may have overlapping code.
   - Recommendation: extract shared logic into clearly named helpers and make prototype code explicitly separated and marked experimental.

7) Logging and observability (Medium)
   - Symptom: code references logging but improve observability by adding structured health-check endpoints and ensuring health reporting is wired to the base service.
   - Recommendation: add health/metrics hooks (if not already), ensure `health_check()` covers all critical components.

8) Use of external resources (High)
   - Symptom: camera and realsense modules interact with hardware. Add graceful fallback paths when hardware isn't present and document how to run with `mock_depth_processor.py`.

   9) Doc / logger naming drift (Low→Medium)
      - Symptom: Several scripts, README snippets, and logger names still refer to the previous `robust_camera` naming (e.g., `robust_camera.py`, `logging.getLogger('robust_camera')`, and `scripts/test_robust_camera_integration.py`). This can confuse developers when tracing logs or opening code references.
      - Recommendation: Standardize on new module names (`depth_processor`, `realsense_camera`, `mock_depth_processor`) across code comments, script names, loggers, and docs. For logger naming, using module paths (e.g., `experimance_core.depth_processor`) is preferred.

---

**Tests and coverage: missing or weak areas**

Covered areas observed:
- There are tests for core image publishing, presence management, queue smoothing, and some integration tests. Good coverage exists for many logic paths.

Missing/weak areas (recommended tests):
1) Camera wrapper unit tests
   - Test: `test_realsense_camera_mocked.py` — validate behavior when camera disconnects intermittently, and verify that the camera wrapper raises/recovers or returns expected sentinel values.

2) Depth processor edge cases
   - Test: `test_depth_processor_edge_conditions.py` — zero frames, extremely noisy frames, abrupt shape changes, and queue-saturation behavior.

3) Presence manager concurrency
   - Test: `test_presence_concurrency.py` — concurrently feed frames and verify outputs and no race conditions.

4) Integration smoke test (mocked pipeline)
   - Test: `test_pipeline_mocked.py` — run camera (mock) -> depth -> presence -> publish with `MockPubSubService` and assert messages published with expected shape and rate. This can run quickly in CI and exercise more code than isolated unit tests.

5) Prompt generation deterministic behaviour
   - Test: `test_prompt_generator_outputs.py` — ensure that `prompt_generator` and `prompter` produce expected tokens/strings given fixed deterministic inputs.

6) Error path tests
   - Cover exception handling and ensure handlers log and propagate appropriately.

---

**Prioritized roadmap (recommended next steps)**

Urgent (1–2 days):
- Add Quickstart and `.env.example` to `services/core/README.md` (see snippets above).
- Replace broad `except:` blocks where they can hide failures; add logging and re-raise where appropriate.
- Add a small integration smoke test that runs the mocked pipeline (camera mock + `MockPubSubService`) in CI.

Short-term (1–2 weeks):
- Audit and convert blocking calls in async flow to `asyncio.to_thread` or other non-blocking patterns.
- Add unit tests for camera wrapper, depth processor, and presence concurrency.
- Add a `docs/` index with Module Map and a developer onboarding checklist.

Medium-term (1–2 months):
- Add health/metrics endpoints and integrate them into CI health checks.
- Refactor duplicated prototype logic into shared helpers and mark prototypes explicitly experimental.

---

**Suggested small follow-ups (pull-request friendly)**

- PR #1: Add `services/core/.env.example` and Quickstart section in `services/core/README.md`.
- PR #2: Add `tests/test_pipeline_mocked.py` implementing a quick end-to-end run using existing `mock_depth_processor.py` and `MockPubSubService` from `libs/common` mocks.
- PR #3: Replace `asyncio.get_event_loop()` with `get_running_loop()` inside coroutines and add `time.monotonic()` for timing.
- PR #4: Update documentation and script names that reference `robust_camera` to the current modules (`depth_processor`, `realsense_camera`) and standardize logger names to `experimance_core.depth_processor` or appropriate module path.

---

**Commands & checks (to run locally)**

```bash
# create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# run core (dev)
export PROJECT_ENV=experimance
uv run -m experimance_core

# run tests just for core
pytest services/core/tests -q

# run the new mocked pipeline test once added
pytest services/core/tests/test_pipeline_mocked.py -q
```



---
**Actioned changes (local)**

- Added `services/core/.env.example` with minimal variables (`PROJECT_ENV`, `EXPERIMANCE_CORE_LOG_LEVEL`, `USE_MOCK_DEPTH`, `DEPTH_MOCK_PATH`).
- Added a per-service Quickstart snippet to `services/core/README.md` and a `Module Map` table to aid developer onboarding.
- Updated `README_DEPTH.md` to replace references to `robust_camera` with `realsense_camera.py` and `depth_processor.py` and updated import examples to use `CameraConfig` from `config.py`.
- Noted where broad `except:` handlers and `get_event_loop()` usages exist, recommending specific exception handling and the use of `get_running_loop()` or `time.monotonic()`.

