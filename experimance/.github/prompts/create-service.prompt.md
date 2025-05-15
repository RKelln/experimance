# Create New Experimance Service

Generate a new service for the Experimance system with the following structure:

1. Create a proper package structure with __init__.py files
2. Implement ZMQ communication using experimance_common utilities
3. Set up proper logging
4. Implement async event handling
5. Add command-line argument parsing
6. Include a main() function using asyncio.run()

The service should follow the standard pattern:
- Initialize ZMQ sockets using ports from DEFAULT_PORTS
- Implement a message_loop() for subscribing
- Implement a publish_loop() if needed
- Handle graceful shutdown with KeyboardInterrupt

Example services are in the services/ directory.
