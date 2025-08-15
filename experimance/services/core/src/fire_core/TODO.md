# Fire Core Service - TODO

## Architecture & Setup
- [x] Create `pyproject.toml` for fire_core package
- [x] Define package structure and dependencies
- [x] Create main service entry point (`__main__.py`)
- [x] Set up logging configuration specific to fire
- [x] Create service configuration schema and default config file

## Core State Machine (`fire_core.py`)
- [x] Implement state machine: Idle → Listening → BaseImage → Tiles
- [x] Define state transitions and triggers
- [x] Implement ZMQ message handlers:
  - [x] `StoryHeard` message handler
  - [x] `UpdateLocation` message handler  
  - [x] `ImageReady` message handler
- [x] Add error handling and recovery logic
- [x] Implement graceful shutdown and cleanup

## LLM Integration (`llm.py`)
- [x] Create LLM client wrapper (OpenAI/Anthropic/local)
- [x] Implement `infer_location()` function
  - [x] Parse story content and extract location/setting details
  - [x] Map to fire biomes and emotions
  - [x] Generate initial prompt for image generation
- [x] Implement `update_location()` function
  - [x] Modify existing prompts based on location updates
  - [x] Preserve narrative context while adjusting setting
- [x] Add prompt templates and examples
- [x] Implement retry logic and error handling
- [x] Add configuration for different LLM providers

## Prompt Builder (`prompt_builder.py`)
- [x] Create prompt templates for fire aesthetic
- [x] Implement story-to-prompt conversion logic
- [x] Add support for biome-specific prompt elements
- [x] Include emotion-based artistic direction
- [x] Support for different prompt styles (realistic, dreamy, abstract)
- [x] Implement prompt validation and safety checks

## Tiling System (`tiler.py`)
- [x] Design tiling strategy for panorama images
- [x] Implement base image analysis and tile calculation
- [x] Create tile overlap and masking logic
- [x] Generate properly positioned tile requests
- [x] Implement edge blending and transparency masks
- [x] Support configurable tile sizes and overlap amounts
- [x] Add validation for tile positioning and coverage

## Message Handling & Communication
- [x] Implement `RenderRequest` message creation and dispatch
- [x] Handle `ImageReady` responses from image_server
- [x] Create `DisplayMedia` messages with proper positioning
- [x] Implement `ContentType.CLEAR` message for display clearing
- [x] Add message queuing and priority handling
- [x] Implement timeout handling for image generation

## Configuration & Environment
- [x] Create fire-specific configuration file
- [x] Define image dimensions and aspect ratios
- [x] Configure tiling parameters (count, overlap, etc.)
- [x] Set LLM provider and API configuration
- [x] Define timeout and retry policies
- [x] Add environment variable support

## Error Handling & Resilience
- [x] Implement comprehensive error handling
- [x] Add timeout mechanisms for all async operations
- [x] Create fallback strategies for failed image generation
- [x] Implement service health monitoring
- [x] Add graceful degradation for partial failures
- [x] Log errors and performance metrics

## Testing
- [ ] Create unit tests for each module
- [ ] Implement integration tests with mock services
- [ ] Add tests for state machine transitions
- [ ] Test tiling logic with various image sizes
- [ ] Create tests for LLM integration
- [ ] Add performance and load testing

## Documentation
- [ ] Document service API and message protocols
- [ ] Create configuration guide
- [ ] Document tiling algorithm and parameters
- [ ] Add troubleshooting guide
- [ ] Document LLM prompt engineering guidelines

## Integration & Deployment
- [ ] Test integration with existing image_server
- [ ] Verify compatibility with fire_display
- [ ] Test with agent service message flow
- [ ] Create deployment scripts and configuration
- [ ] Add monitoring and logging hooks

## Performance Optimization
- [ ] Optimize image processing and tiling performance
- [ ] Implement efficient message handling
- [ ] Add caching for repeated operations
- [ ] Profile and optimize critical paths
- [ ] Implement resource management and cleanup

## Questions for Clarification
1. **Base Image Dimensions**: What should be the target resolution for the base panorama image?
2. **Tile Count**: How many tiles should be generated? Is this configurable?
3. **LLM Provider**: Which LLM service should be used by default?
4. **Timeout Values**: What are appropriate timeout values for image generation?
5. **Error Recovery**: How should the service recover from failed image generation?
6. **Configuration**: Should configuration be in TOML files or environment variables?
