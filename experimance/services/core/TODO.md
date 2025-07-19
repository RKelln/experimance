# Experimance Core Service - TODO

## Phase 1: Basic Service Infrastructure ✅
- [x] Create base service class `ExperimanceCoreService` extending `ZMQPublisherSubscriberService`
- [x] Implement configuration loading from `config.toml`
- [x] Set up ZMQ publisher socket on port 5555 (`events` channel)
- [x] Set up ZMQ subscriber socket on port 5555 for coordination messages
- [x] Register message handlers for `ImageReady`, `AgentControl`, `AudioStatus`
- [x] Create basic async service lifecycle (start/stop/run)
- [x] Add logging configuration and structured logging
- [x] Implement signal handlers for graceful shutdown

## Phase 2: State Management
- [x] Define state data structures (Era, Biome, UserInteraction, etc.)
- [x] Implement era state machine with transition logic
- [x] Create state persistence (load/save from JSON)
- [x] Add idle timer and wilderness drift functionality
- [x] Implement user interaction score calculation
- [x] Create state validation and error recovery

## Phase 3: Depth Processing Integration ✅
- [x] Integrate existing `depth_finder.py` module as-is
- [x] Create async task for continuous depth monitoring
- [x] Implement depth change detection and scoring
- [x] Add hand detection to prevent false triggers
- [x] Create depth map preprocessing for image generation
- [x] Add configurable depth processing parameters

## Phase 4: Prompt Generation Integration  
- [ ] Integrate existing `prompter.py` module as-is
- [ ] Load location and development data from JSON files
- [ ] Implement era/biome to prompt mapping
- [ ] Create prompt caching for performance
- [ ] Add fallback prompts for error cases
- [ ] Test prompt generation across all era/biome combinations

## Phase 5: Event Publishing
- [ ] Implement `EraChanged` event publishing
- [ ] Implement `RenderRequest` event publishing  
- [ ] Implement `IdleStateChanged` event publishing
- [ ] Add event throttling to prevent spam
- [ ] Create event logging for debugging

## Phase 6: Agent Integration
- [ ] Subscribe to `agent_ctrl` channel (tcp://*:5559)
- [ ] Implement `AudiencePresent` event handler
- [ ] Implement `SuggestBiome` event handler
- [ ] Add agent influence on state machine
- [ ] Create agent response validation
- [ ] Add timeout handling for agent communication

## Phase 7: Audio Integration & Tag Extraction
- [ ] Implement audio tag extraction from prompts (keyword matching)
- [ ] Create audio tag database/configuration loading
- [ ] Implement tag comparison logic (include/exclude lists)
- [ ] Publish `AudioCommand` messages for era/biome changes
- [ ] Publish `AudioCommand` messages for interaction sounds
- [ ] Handle audio status responses and coordination
- [ ] Add audio tag state persistence
- [ ] Test audio tag extraction across all prompt types

## Phase 8: Service Coordination & Response Handling
- [ ] Implement `ImageReady` message handler
- [ ] Coordinate audio transitions with image completion
- [ ] Implement `AgentControl` message handler
- [ ] Implement `AudioStatus` message handler
- [ ] Add service response timeout handling
- [ ] Create coordination state tracking
- [ ] Add service health monitoring

## Phase 9: Depth Difference Visualization
- [ ] Implement depth difference detection and image generation
- [ ] Publish `VideoMask` messages to display service
- [ ] Add interaction visualization configuration
- [ ] Test depth difference rendering pipeline
- [ ] Add configurable visualization parameters
- [ ] Integrate with user interaction scoring

## Phase 10: Interaction Sound Management
- [ ] Implement hand detection sound triggers (continuous while detected)
- [ ] Add sand sensor integration for touch detection (future)
- [ ] Implement interaction sound triggering logic
- [ ] Create sound cue database and configuration
- [ ] Implement sound playback control (start/stop/pause)
- [ ] Add 3D spatialization for interaction sounds
- [ ] Test interaction sound management with user actions
- [ ] Optimize sound management for performance

## Phase 11: Configuration & Persistence
- [ ] Create comprehensive configuration schema
- [ ] Implement configuration validation
- [ ] Add runtime configuration reloading
- [ ] Create state checkpointing every 30 seconds
- [ ] Implement crash recovery from saved state
- [ ] Add configuration migration for version updates

## Phase 12: Error Handling & Resilience
- [ ] Implement depth camera reconnection logic
- [ ] Add graceful degradation when services unavailable
- [ ] Create error metrics and reporting
- [ ] Add circuit breaker pattern for external dependencies
- [ ] Implement exponential backoff for retries
- [ ] Add health check endpoint

## Phase 13: Performance Optimization
- [ ] Profile depth processing performance
- [ ] Optimize event publishing frequency
- [ ] Add configurable processing resolution
- [ ] Implement memory management for long-running operation
- [ ] Add performance metrics collection
- [ ] Optimize prompt generation caching

## Phase 14: Testing & Quality
- [ ] Unit tests for state machine logic
- [ ] Unit tests for configuration loading
- [ ] Integration tests with mock ZMQ services
- [ ] Mock depth camera for testing
- [ ] Performance tests for sustained operation
- [ ] Memory leak detection tests
- [ ] End-to-end experience flow tests

## Phase 15: Monitoring & Observability
- [ ] Add Prometheus metrics exports
- [ ] Implement structured logging with correlation IDs
- [ ] Create health check endpoints
- [ ] Add performance dashboards
- [ ] Implement alerting for critical failures
- [ ] Add debug mode for detailed tracing

## Phase 16: Documentation & Deployment
- [ ] Create service README with setup instructions
- [ ] Document configuration options
- [ ] Create troubleshooting guide
- [ ] Add deployment scripts
- [ ] Create service monitoring documentation
- [ ] Add performance tuning guide

## Technical Debt & Future Work

### Code Organization
- [ ] Consider splitting `depth_finder` into separate service
- [ ] Evaluate `prompter` enhancement with LLM integration
- [ ] Create plugin architecture for era extensions
- [ ] Add A/B testing framework for experience variations

### Infrastructure
- [ ] Add Redis support for distributed state (FUTURE)
- [ ] Implement service mesh integration (FUTURE)
- [ ] Add container deployment support (MAYBE)
- [ ] Create auto-scaling capabilities (FUTURE)

### Dependencies
- [x] Update technical design document with latest architecture
- [x] Evaluate replacing multiprocessing with pure async
- [ ] Consider migration to faster depth processing (Rust/C++)
- [ ] Evaluate real-time performance requirements

## Configuration Files to Create
- [ ] `config.toml` - Main service configuration
- [ ] `depth_proc.toml` - Depth processing parameters  
- [ ] `saved_data/default_state.json` - Initial state
- [ ] `config/audio_tags.json` - Known audio tags for extraction
- [ ] `prompts/` directory structure with Jinja2 templates
- [ ] `data/locations.json` - Geographic location data
- [ ] `data/anthropocene.json` - Era development data

## Dependencies to Add
- [ ] Review and update `pyproject.toml` dependencies
- [ ] Add development dependencies for testing
- [ ] Consider optional dependencies for different modes
- [ ] Add type checking with mypy
- [ ] Add code formatting with ruff

## Integration Points to Test
- [ ] ZMQ message exchange with display service
- [ ] ZMQ message exchange with audio service
- [ ] ZMQ message exchange with agent service
- [ ] ZMQ message exchange with image_server service
- [ ] OSC sensor data reception
- [ ] Depth camera hardware interface

## Performance Targets
- [ ] Depth processing: 15-30 Hz minimum
- [ ] State updates: 1-5 Hz for smooth transitions
- [ ] Event publishing latency: < 100ms
- [ ] Agent response time: < 200ms  
- [ ] Memory usage: < 500MB sustained
- [ ] Installation runtime: 4+ weeks without restart

## Known Issues to Address
- [ ] Technical design document outdated (display service changes)
- [ ] Multiprocessing approach needs async conversion
- [ ] Missing OSC sensor integration
- [ ] No current health monitoring
- [ ] Limited error recovery mechanisms
- [ ] Configuration management needs improvement
