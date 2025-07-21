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
- [x] convert renderrequest to file uri 
- [ ] more/better blur on depth map

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
- [x] all layers functional
- [x] convert cli to new ZMQ

### Audio
- [x] create supercollider prototype 
  - [x] integrate ZMQ
  - [x] add surround sound (agent, environment and music)
- [ ] fix scd script 
  - [ ] so it works without USB audio device 
  - [ ] works in scide (audio paths)

### Image server
- [x] convert prototype to production code
  - [x] integrate ZMQ
- [x] convert cli to new zmq
- [ ] write tests
- [ ] test runware and vast.ai and other cloud options
  - [x] vastai
    - [ ] fix provisioning or workaround
    - [ ] allow for multiple loras
    - [x] helper script
- [ ] diffusers speed-ups tests

### Voice Agent

- [x] convert prototype to production code
- [] research mem0 memory integration

### Infrastructure
- [ ] test production deploy

## ZMQ
- [] services -> core comms
  - add message broker with XPUB and XSUB that sits between core and other services so that events go out from core and "updates" go back to core from the services ? or
  - https://chatgpt.com/share/6855c763-9268-8005-88e6-8307808fc0f0
    - add core pull and others service push