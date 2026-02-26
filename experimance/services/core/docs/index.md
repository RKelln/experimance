# docs/

Documentation for `services/core` – the central coordinator for the Experimance and Feed the Fires interactive art installations.

## Index

| File | Description |
|------|-------------|
| [architecture.md](architecture.md) | Service composition pattern, ZMQ port map, published/subscribed message schemas, state machine design, internal module map, and async task structure for both projects. |
| [depth-camera.md](depth-camera.md) | Intel RealSense camera setup, `CameraConfig` reference, automatic error recovery settings, mock processor usage, test scripts, and troubleshooting for common camera errors. |
| [fire-core.md](fire-core.md) | Feed the Fires core service: story/transcript pipeline, request state machine, smart interruption policy, tiling strategy, ZMQ channels, and development guide. |
| [new-service-guide.md](new-service-guide.md) | Step-by-step guide for creating a new project-specific core service following the established `BaseService` + `ControllerService` pattern. |
| [roadmap.md](roadmap.md) | Near-term goals, known gaps, and future work for `services/core`. |
