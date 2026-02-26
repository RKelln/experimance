# Core Service Roadmap

This file tracks near-term goals and known gaps for `services/core` (both `experimance_core` and `fire_core`).

---

## Near-term goals

### Testing

- **Smoke test in CI** – Add a headless pytest job that runs the core with `MockDepthProcessor` and a mocked ZMQ service, asserting that presence and `CHANGE_MAP` events are published at the expected rate and with the correct payload shape.
- **Depth processor edge cases** – No-frame, noisy-frame, and abrupt resolution-change scenarios (`test_depth_processor_edge_conditions.py`).
- **Presence manager concurrency** – Concurrent frame injection to verify hysteresis remains correct without races (`test_presence_concurrency.py`).
- **Prompt generator determinism** – Seeded runs should produce identical templates and negative prompts.
- **Fire core unit tests** – State machine, tiler, and LLM integration unit tests are still largely missing.

### Code quality

- **Async-safe blocking calls** – Replace `subprocess.run()` in runtime paths (`camera_utils.py` etc.) with `asyncio.to_thread` or `asyncio.create_subprocess_exec`. Prevents event-loop stalls.
- **Event-loop time API** – Replace `asyncio.get_event_loop().time()` with `time.monotonic()` throughout. Use `asyncio.get_running_loop()` only inside coroutines when truly needed.
- **Narrow exception handlers** – Replace `except Exception: pass` with specific exception types, logging, and deliberate error propagation.
- **Structured task cancellation** – Background loops should honour cancellation with `asyncio.Event` flags or structured `asyncio.TaskGroup`, not just `await asyncio.sleep()`.

### Documentation

- **Health monitoring guide** – Document what `BaseService.health_check()` reports and how to monitor it in production (log location, fields, format).

---

## Known gaps

| Gap | Impact | Notes |
|-----|--------|-------|
| No CI smoke test for core | Medium | Mock run validates most of the pipeline without hardware |
| `depth_finder.py` prototype still in tree | Low | Duplicate logic vs `depth_processor.py`; candidate for removal |
| `sohkepayin_core` partially implemented | Low | Tests exist but service is not yet complete |
| `TODO.md` phases are stale | Low | Many phases marked incomplete are actually done; TODO.md removed in favour of this file |
| DESIGN.md event schemas are outdated | — | Consolidated and corrected in `docs/architecture.md` |
| Redis-backed distributed state | Low | Planned future work; relevant only for multi-machine deployments |

---

## Future work (not scheduled)

- Multiple camera support (additional RealSense for wider coverage).
- Prometheus metrics endpoint on `BaseService`.
- Hot-swap camera reconnect without service restart.
- Plugin architecture for adding new era types without code changes.
- A/B testing framework for experience variations.
- Container deployment (`Dockerfile` + `docker-compose`).
