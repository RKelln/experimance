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

