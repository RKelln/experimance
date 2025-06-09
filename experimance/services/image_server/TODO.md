# Image Server Service TODO

## Architecture & Design
- [x] Analyze current prototype implementation
- [x] Define service architecture following Experimance patterns
- [x] Implement main `ImageServerService` class inheriting from `ZmqPublisherSubscriberService`
- [x] Create strategy pattern for image generation backends
- [x] Design message handling for `RenderRequest` and `ImageReady`

## Core Service Implementation
- [x] Create `ImageServerService` class with proper ZMQ communication
- [ ] Implement `RenderRequest` message handler
- [ ] Implement `ImageReady` message publishing
- [x] Add configuration management with TOML support
- [x] Implement proper async/await patterns
- [x] Add graceful shutdown and cleanup

## Image Generation Strategies
- [x] Create abstract `ImageGenerator` base class
- [x] Implement `MockImageGenerator` for testing
- [ ] Implement `LocalSDXLGenerator` for local generation
- [x] Implement `FalAIGenerator` for remote FAL.AI service
- [ ] Implement `OpenAIGenerator` for DALL-E integration
- [x] Add strategy selection and configuration

## Message Schemas & Communication
- [ ] Define `RenderRequest` message schema validation
- [ ] Define `ImageReady` message schema
- [ ] Implement proper error handling for malformed messages
- [ ] Add request ID tracking for correlation
- [ ] Implement timeout handling for long-running generations

## Testing Strategy (TDD)
- [ ] Create unit tests for `ImageServerService` lifecycle
- [ ] Create unit tests for message handling
- [ ] Create unit tests for each generation strategy
- [ ] Create integration tests for ZMQ communication
- [ ] Create mock tests for external API dependencies
- [ ] Add performance tests for image generation throughput

## Configuration & Deployment
- [ ] Update `config.toml` with proper service configuration
- [ ] Add environment variable support for API keys
- [ ] Implement cache management for generated images
- [ ] Add logging configuration
- [ ] Create service entry point script

## Error Handling & Monitoring
- [ ] Implement comprehensive error handling
- [ ] Add retry logic for failed generations
- [ ] Implement health check endpoints
- [ ] Add metrics collection for monitoring
- [ ] Implement proper logging with structured output

## Documentation
- [ ] Create service documentation
- [ ] Document message schemas
- [ ] Add API documentation for each generator
- [ ] Create deployment guide
- [ ] Add troubleshooting guide

## Integration
- [ ] Test integration with experimance core service
- [ ] Test integration with display service
- [ ] Validate against technical design requirements
- [ ] Performance testing under load

## Future Enhancements
- [ ] Add support for additional online services (runware, etc.)
- [ ] Implement image caching and deduplication
- [ ] Add support for batch processing
- [ ] Implement priority queuing for requests
- [ ] Add support for different output formats
- [ ] Implement rate limiting for external APIs