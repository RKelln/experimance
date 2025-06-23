# TODO

## Services

### Core

- [x] convert prototype to production code
  - [x] integrate ZMQ
- [x] write tests
- [x] add depth cam
- [x] add prompt generator
  - [x] convert from prototype
- [] add sensors

### Sand Sensors
- [] add prototype code
- [] test sensors
- [] write tests
- [] integrate ZMQ

### Depth finder
- [x] convert prototype to production code
  - [x] integrate ZMQ
- [x] write tests

### Display
- [x] convert prototype to production code
  - [x] integrate ZMQ
- [] write tests
- [] all layers functional

### Audio
- [x] create supercollider prototype 
  - [] integrate ZMQ

### Image server
- [x] convert prototype to production code
  - [x] integrate ZMQ
- [ ] convert cli to new zmq
- [ ] write tests
- [ ] test runware and vast.ai and other cloud options

### Voice Agent

- [] convert prototype to production code
- [] research mem0 memory integration: https://docs.mem0.ai/integrations/livekit



## ZMQ
- [] services -> core comms
  - add message broker with XPUB and XSUB that sits between core and other services so that events go out from core and "updates" go back to core from the services
 - https://chatgpt.com/share/6855c763-9268-8005-88e6-8307808fc0f0
   - add core pull and others service push