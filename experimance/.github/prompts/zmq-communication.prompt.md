# Implement ZMQ Communication Layer

Create a ZeroMQ communication layer for an Experimance service following these requirements:

1. Use the `experimance_common.zmq_utils` module for ZMQ sockets
2. Configure the following connections:
   - Publisher for service-specific events
   - Subscriber to the coordinator service
   - Optional PUSH/PULL sockets for work distribution
3. Handle the common message types: Heartbeat and EraChanged
4. Include proper error handling and socket cleanup
5. Follow the async patterns used elsewhere in the project

Include proper documentation with examples of how to send and receive messages.

## Message Structure

Messages should follow the standard structure:
```python
{
    "type": MessageType.EXAMPLE,
    "timestamp": time.time(),
    "data": {
        # Message-specific fields
    }
}
```
