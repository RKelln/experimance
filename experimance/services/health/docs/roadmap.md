# Health Service Roadmap

## Near-Term Goals

### Better Development Experience
- [ ] Provide a `scripts/fake_service.py` helper that writes a synthetic health file on a timer,
  making it easy to test the health service in isolation without running the full system.
- [ ] Add a `--dry-run` flag that prints which notifications would be sent without delivering them.

### Observability
- [ ] Expose a simple HTTP `/status` endpoint (read-only) that returns the current in-memory health
  snapshot as JSON, so external tools can poll without reading the file system directly.
- [ ] Emit structured log lines on every check cycle (not just on change) at `DEBUG` level to
  simplify debugging stale-detection issues.

### Notifications
- [ ] Add a configurable `notify_on_recovery` option (separate from `notify_on_healthy`) so
  operators can receive a single "service recovered" alert without enabling full healthy-status
  chatter.
- [ ] Support multiple ntfy topics (e.g. separate critical vs informational topics).

### Reliability
- [ ] Add integration tests that write health files with controlled timestamps and assert that the
  correct `HealthStatus` is detected (staleness, missing file, parse error).
- [ ] Validate that `expected_services` matches `experimance_common.SERVICE_TYPES` at startup and
  log a warning for any discrepancy.

## Known Gaps

| Gap | Notes |
|---|---|
| No HTTP endpoint | In-memory state is only visible via logs or by reading health files directly |
| WARNING never notifies at default level | Intentional design choice but can surprise operators; document more prominently |
| Grace period default mismatch | `config.py` default is 60 s but `config.toml` sets 10 s; document and align |
| No unit tests for notification filtering | The cooldown and level-filter logic in `_should_notify` is not covered by automated tests |
| `transition` service not in `expected_services` | The transition service is still WIP and intentionally excluded from monitoring |
