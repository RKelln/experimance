# Surround Sound Setup and Implementation Guide

This document contains everything we discovered while setting up 5.1 (6-channel) surround sound with SuperCollider on Linux, and how to implement multi-channel audio output for the Experimance project.

## Hardware Setup

### USB Audio Device Configuration
- **Device**: 5.1 surround sound USB audio interface
- **Channels**: 6 channels total
- **Channel Mapping**:
  - 0: Left Front
  - 1: Right Front  
  - 2: Center
  - 3: LFE (Low Frequency Effects/Subwoofer)
  - 4: Left Surround (Rear Left)
  - 5: Right Surround (Rear Right)

### ALSA/JACK Configuration
- Use `qjackctl` to configure JACK for 6 output channels
- Set sample rate to 48000 Hz (standard for USB devices)
- Ensure USB device is selected as the audio interface
- JACK will override ALSA device settings when active

## SuperCollider Implementation

### Key Discovery: Simple Channel Routing
The working approach for multi-channel output in SuperCollider SynthDefs is:

```supercollider
SynthDef(\mysynth, {
    arg channel = 0, freq = 440, amp = 0.5;
    var sig = SinOsc.ar(freq) * amp;
    
    // Direct output to specific channel - THIS WORKS
    Out.ar(channel, sig);
}).add;
```

### What DOESN'T Work
These approaches caused runtime errors or compilation failures:

1. **Conditional multiplication with booleans**:
```supercollider
// ERROR: Can't multiply boolean by UGen
Out.ar(0, [
    (channel == 0) * sig,
    (channel == 1) * sig,
    // ...
]);
```

2. **Conditional multiplication with .asInteger**:
```supercollider
// Compiles but doesn't route properly
Out.ar(0, [
    (channel == 0).asInteger * sig,
    (channel == 1).asInteger * sig,
    // ...
]);
```

3. **Select.ar with nested arrays**:
```supercollider
// ERROR: Select input was not audio rate
Out.ar(0, Select.ar(channel, [
    [sig, 0, 0, 0, 0, 0],
    [0, sig, 0, 0, 0, 0],
    // ...
]));
```

4. **Select.kr for audio output**:
```supercollider
// ERROR: Out input is not audio rate
Out.ar(0, [
    Select.kr(channel, [sig, 0, 0, 0, 0, 0]),
    // ...
]);
```

5. **Array.collect with conditionals**:
```supercollider
// Doesn't work properly in SynthDef context
output = output.collect({ |val, i| 
    if(i == channel, sig, 0)
});
```

### The Working Solution
**Use `Out.ar(channel, sig)` directly** - this outputs the signal to the specified channel number only.

## Testing Framework

The `surround_sound.scd` script provides comprehensive testing functions:

### Basic Channel Tests
```supercollider
~testChannel.(4);        // Test rear left channel
~testChannel.(5);        // Test rear right channel
~testAllChannels.();     // Test all channels sequentially
```

### Stereo Pair Tests
```supercollider
~testStereo.(\frontlr);  // Test front left/right pair
~testStereo.(\rearlr);   // Test rear left/right pair
```

### Different Audio Types
```supercollider
~testSynthType.(\noise, 4);   // Pink noise on rear left
~testSynthType.(\saw, 5);     // Sawtooth on rear right
~testSynthType.(\perc, 2);    // Percussion on center
```

### Frequency Response
```supercollider
~testSweep.(4);          // Frequency sweep on rear left
~testBass.(3);           // Bass test on LFE channel
```

### Physical Verification
```supercollider
~testRearSpeakersOnly.(); // Specifically test rear speakers
```

## Implementation for Experimance Audio

### SynthDef Template
When adding multi-channel support to `experimance_audio.scd`, use this pattern:

```supercollider
SynthDef(\myExperimentSynth, {
    arg freq = 440, amp = 0.5, channel = 0, gate = 1;
    var sig, env;
    
    // Your audio generation code here
    env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
    sig = SinOsc.ar(freq) * amp * env;
    
    // Simple, reliable channel routing
    Out.ar(channel, sig);
}).add;
```

### Usage Examples
```supercollider
// Play on specific channels
Synth(\myExperimentSynth, [\channel, 0]);  // Front left
Synth(\myExperimentSynth, [\channel, 4]);  // Rear left
Synth(\myExperimentSynth, [\channel, 3]);  // LFE/Sub

// For stereo or multi-channel sounds, create multiple synths
~frontStereo = [
    Synth(\myExperimentSynth, [\channel, 0, \freq, 440]),  // Left
    Synth(\myExperimentSynth, [\channel, 1, \freq, 440])   // Right
];

~rearStereo = [
    Synth(\myExperimentSynth, [\channel, 4, \freq, 220]),  // Rear left
    Synth(\myExperimentSynth, [\channel, 5, \freq, 220])   // Rear right
];
```

## Troubleshooting

### JACK Connection Issues
If channels don't work, check JACK connections:
```supercollider
~checkJackConnections.();   // Check current connections
~connectAllChannels.();     // Manually connect all channels
```

### Device Detection
```supercollider
~listAudioDevices.();       // List available devices
~checkDeviceChannels.();    // Check channel configuration
```

### Channel Verification
```supercollider
~testRearSpeakersOnly.();   // Specifically test problematic channels
~testAllChannels.();        // Sequential test of all channels
```

## Common Pitfalls

1. **Don't try to create 6-channel arrays** - use direct channel routing instead
2. **Avoid conditional logic in SynthDefs** - SuperCollider handles this differently than expected
3. **Use control-rate channel parameters** - the channel number should be a control value, not audio-rate
4. **Check JACK connections** - SuperCollider may not auto-connect all channels
5. **Verify physical wiring** - ensure speakers are connected to correct outputs on USB device

## Server Configuration

### Essential Settings
```supercollider
// Configure server for 6 channels
s.options.numOutputBusChannels = 6;
s.options.device = "plughw:1,0";  // USB device
s.options.sampleRate = 48000;
s.boot;
```

### Verification
```supercollider
// Check server status
s.queryAllNodes;
s.options.numOutputBusChannels;  // Should return 6
s.sampleRate;                    // Should return 48000.0
```

## Integration Notes for Experimance

When integrating multi-channel support into the main Experimance audio system:

1. **Modify existing SynthDefs** to accept a `channel` parameter
2. **Use `Out.ar(channel, sig)`** for all audio output
3. **Update message handling** to include channel routing information
4. **Test thoroughly** with the provided testing functions
5. **Consider spatial audio algorithms** for automatic channel assignment based on virtual position

## Files
- `surround_sound.scd` - Complete testing and setup script
- `experimance_audio.scd` - Main audio system (to be updated with multi-channel support)

## Hardware Tested
- USB 5.1 surround sound interface on Linux
- 6-channel output verified with physical speakers
- JACK/ALSA configuration confirmed working

## Stereo Audio Routing

### Working Approaches for Stereo Pairs

**Method 1: Separate Out.ar calls (RECOMMENDED)**
```supercollider
SynthDef(\stereo_synth, {
    arg startChan = 0, freq = 440, amp = 0.5;
    var sigL, sigR;
    sigL = SinOsc.ar(freq) * amp;
    sigR = SinOsc.ar(freq * 1.01) * amp;  // Slightly detuned
    
    // Output to adjacent channels
    Out.ar(startChan, sigL);       // Left channel
    Out.ar(startChan + 1, sigR);   // Right channel
}).add;
```

**Method 2: Mono to Stereo Expansion**
```supercollider
SynthDef(\mono_to_stereo, {
    arg startChan = 0, freq = 440, amp = 0.5;
    var sig = SinOsc.ar(freq) * amp;
    
    // Duplicate mono signal to two channels
    Out.ar(startChan, sig);       // Left channel
    Out.ar(startChan + 1, sig);   // Right channel (same signal)
}).add;
```

**Method 3: Stereo to Mono Downmix**
```supercollider
SynthDef(\stereo_to_mono, {
    arg channel = 2, freq = 440, amp = 0.5;
    var sigL, sigR, mono;
    sigL = SinOsc.ar(freq) * amp;
    sigR = SinOsc.ar(freq * 1.01) * amp;
    
    // Proper stereo downmix (average L+R)
    mono = (sigL + sigR) * 0.5;
    Out.ar(channel, mono);
}).add;
```

### Audio Bus Routing

Audio buses provide flexible signal routing between synths:

```supercollider
// Source synth outputs to bus
SynthDef(\bus_source, {
    arg busnum = 10, freq = 440, amp = 0.5;
    var sig = SinOsc.ar(freq) * amp;
    Out.ar(busnum, sig);         // Left to bus
    Out.ar(busnum + 1, sig * 0.8); // Right to bus (different level)
}).add;

// Router synth reads from bus and outputs to channels
SynthDef(\bus_router, {
    arg inBus = 10, outChan = 0, gate = 1;
    var sigL, sigR, env;
    env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
    sigL = In.ar(inBus, 1);
    sigR = In.ar(inBus + 1, 1);
    Out.ar(outChan, sigL * env);
    Out.ar(outChan + 1, sigR * env);
}).add;

// Usage:
~source = Synth(\bus_source, [\busnum, 10]);
~router = Synth(\bus_router, [\inBus, 10, \outChan, 0], ~source, \addAfter);
```

### Channel Pair Recommendations

For 5.1 surround, use these channel pairs:
- **Front Stereo**: channels 0,1 (Left Front, Right Front)
- **Rear Stereo**: channels 4,5 (Left Rear, Right Rear)  
- **Center + LFE**: channels 2,3 (Center, LFE) - typically mono sources

## Integration Recommendations for Experimance Audio

### 1. SynthDef Modifications

**For existing mono SynthDefs**, add a `channel` parameter:
```supercollider
// BEFORE (stereo output):
SynthDef(\mysynth, {
    arg freq = 440, amp = 0.5;
    var sig = SinOsc.ar(freq) * amp;
    Out.ar(0, [sig, sig]);  // Fixed stereo output
}).add;

// AFTER (configurable channel output):
SynthDef(\mysynth, {
    arg freq = 440, amp = 0.5, channel = 0;
    var sig = SinOsc.ar(freq) * amp;
    Out.ar(channel, sig);  // Direct channel output
}).add;
```

**For stereo SynthDefs**, add `startChan` parameter:
```supercollider
SynthDef(\stereosynth, {
    arg freq = 440, amp = 0.5, startChan = 0;
    var sigL, sigR;
    sigL = SinOsc.ar(freq) * amp;
    sigR = SinOsc.ar(freq * 1.01) * amp;
    
    Out.ar(startChan, sigL);       // Left channel
    Out.ar(startChan + 1, sigR);   // Right channel
}).add;
```

### 2. Message Handling Extensions

Extend the audio service message handling to include channel routing:

```python
# In experimance_audio message handlers:
def handle_audio_message(self, message):
    if message.type == "spatial_audio":
        # Calculate channel based on spatial position
        channel = self.calculate_channel_from_position(
            message.data.get("x", 0), 
            message.data.get("y", 0)
        )
        self.send_to_supercollider({
            "synth": message.data.get("synth", "sine_test"),
            "freq": message.data.get("freq", 440),
            "amp": message.data.get("amp", 0.5),
            "channel": channel
        })
    
    elif message.type == "stereo_audio":
        # Route stereo content to specific channel pairs
        start_chan = self.get_stereo_pair_start(message.data.get("location", "front"))
        self.send_to_supercollider({
            "synth": message.data.get("synth", "stereosynth"),
            "startChan": start_chan,
            "freq": message.data.get("freq", 440),
            "amp": message.data.get("amp", 0.5)
        })

def calculate_channel_from_position(self, x, y):
    """Map 2D position to surround channel"""
    # Example mapping for sand table coordinates
    if x < -0.3:  # Left side
        return 4 if y < 0 else 0  # Rear left or front left
    elif x > 0.3:  # Right side  
        return 5 if y < 0 else 1  # Rear right or front right
    else:  # Center
        return 2 if y > 0 else 3  # Center or LFE

def get_stereo_pair_start(self, location):
    """Get starting channel for stereo pairs"""
    return {
        "front": 0,   # Channels 0,1
        "rear": 4,    # Channels 4,5
        "center": 2   # Channels 2,3 (center+LFE)
    }.get(location, 0)
```

### 3. Spatial Audio Algorithm

Implement automatic channel selection based on virtual positioning:

```supercollider
// In SuperCollider, create a spatial audio router
SynthDef(\spatialAudio, {
    arg freq = 440, amp = 0.5, x = 0, y = 0, spread = 0.5;
    var sig, channels;
    sig = SinOsc.ar(freq) * amp;
    
    // Calculate channel distribution based on position
    channels = [
        max(0, (1 - abs(x + 0.5)) * max(0, y + 0.5)),  // Front left
        max(0, (1 - abs(x - 0.5)) * max(0, y + 0.5)),  // Front right
        max(0, (1 - abs(x)) * max(0, y + 0.5)),        // Center
        max(0, (1 - abs(x)) * max(0, -y + 0.5)),       // LFE
        max(0, (1 - abs(x + 0.5)) * max(0, -y + 0.5)), // Rear left
        max(0, (1 - abs(x - 0.5)) * max(0, -y + 0.5))  // Rear right
    ].normalizeSum * spread;
    
    // Output to all channels with calculated weights
    Out.ar(0, sig * channels[0]);  // Front left
    Out.ar(1, sig * channels[1]);  // Front right
    Out.ar(2, sig * channels[2]);  // Center
    Out.ar(3, sig * channels[3]);  // LFE
    Out.ar(4, sig * channels[4]);  // Rear left
    Out.ar(5, sig * channels[5]);  // Rear right
}).add;
```

### 4. Configuration Options

Add surround sound configuration to the audio service:

```toml
# In audio service config.toml
[surround]
enabled = true
channels = 6
default_mode = "spatial"  # "spatial", "stereo", "mono"
channel_mapping = [
    "front_left",    # 0
    "front_right",   # 1  
    "center",        # 2
    "lfe",           # 3
    "rear_left",     # 4
    "rear_right"     # 5
]

[spatial_mapping]
# Map sand table coordinates to channels
table_width = 1.0
table_height = 1.0
front_threshold = 0.3
rear_threshold = -0.3
```

### 5. Implementation Priority

**Phase 1: Basic Multi-Channel Support**
1. Add `channel` parameter to all existing SynthDefs
2. Update message handlers to accept channel routing
3. Test with manual channel assignment

**Phase 2: Stereo Pair Support**  
1. Add stereo SynthDefs with `startChan` parameter
2. Implement stereo pair routing logic
3. Add configuration for stereo placement

**Phase 3: Spatial Audio**
1. Implement automatic channel calculation from coordinates
2. Add smooth transitions between channels
3. Test with real sand table positioning data

## External Audio Routing

### Challenge: Multiple Audio Sources
You want to route audio from different programs to specific surround channels:
- **Environmental audio** (stereo) → Front channels (0,1) 
- **Music** (stereo) → Rear channels (4,5)
- **AI agent voice** (mono) → Center channel (2)

### Solution Options

#### Option 1: Use System JACK (RECOMMENDED)
Instead of letting SuperCollider start its own JACK server, start JACK first and have SuperCollider connect to it:

**Setup:**
```bash
# Start JACK with your USB device first
qjackctl &  # Start JACK with GUI
# OR command line:
jackd -d alsa -d plughw:1,0 -r 48000 -p 512 -n 2 -P 6

# Then start SuperCollider and connect to existing JACK
# In SuperCollider:
Server.default.options.device = nil;  // Use system JACK
s.boot;
```

**Benefits:**
- All programs can connect to the same JACK server
- Easy routing with `qjackctl` patchbay or command line tools
- SuperCollider becomes just another JACK client

#### Option 2: SuperCollider as Audio Router
Use SuperCollider to receive external audio and route it:

```supercollider
// Create audio input buses for external programs
SynthDef(\external_router, {
    arg inBus = 8, outChan = 0, amp = 1.0;
    var sig = In.ar(inBus, 1);  // Read from input bus
    Out.ar(outChan, sig * amp);
}).add;

// Route different inputs to different outputs
~envAudio = Synth(\external_router, [\inBus, 8, \outChan, 0]);   // Env → Front L
~envAudio = Synth(\external_router, [\inBus, 9, \outChan, 1]);   // Env → Front R
~musicL = Synth(\external_router, [\inBus, 10, \outChan, 4]);    // Music → Rear L  
~musicR = Synth(\external_router, [\inBus, 11, \outChan, 5]);    // Music → Rear R
~voice = Synth(\external_router, [\inBus, 12, \outChan, 2]);     // Voice → Center
```

**External programs connect to SuperCollider's input buses**

#### Option 3: ALSA Loopback + Routing
Use ALSA loopback devices to create virtual audio interfaces:

```bash
# Load ALSA loopback module
sudo modprobe snd-aloop

# This creates virtual sound cards that programs can use
# Route them with ALSA or PulseAudio tools
```

#### Option 4: PipeWire (Modern Linux)
If using PipeWire instead of PulseAudio:

```bash
# PipeWire can route between different programs easily
pw-link program1:output_FL surround_device:input_0  # Env → Front L
pw-link program1:output_FR surround_device:input_1  # Env → Front R
pw-link program2:output_L surround_device:input_4   # Music → Rear L
pw-link program2:output_R surround_device:input_5   # Music → Rear R
pw-link ai_voice:output surround_device:input_2     # Voice → Center
```

### Recommended Implementation

**For Experimance, I recommend Option 1 (System JACK):**

1. **Start JACK first** with 6 outputs to your USB device
2. **Configure SuperCollider** to connect to existing JACK (not start its own)
3. **Route external programs** using JACK connections:

```bash
# Environmental audio program → Front channels
jack_connect environmental_audio:out_L system:playback_1
jack_connect environmental_audio:out_R system:playback_2

# Music program → Rear channels  
jack_connect music_player:out_L system:playback_5
jack_connect music_player:out_R system:playback_6

# AI voice → Center channel
jack_connect ai_voice:out system:playback_3

# SuperCollider → All channels (for generated audio)
jack_connect SuperCollider:out_1 system:playback_1
jack_connect SuperCollider:out_2 system:playback_2
# ... etc for all 6 channels
```

### Configuration Changes Needed

**In experimance_audio service:**
```python
# Before starting SuperCollider, ensure JACK is running
def setup_jack_server(self):
    # Check if JACK is running
    result = subprocess.run(['jack_lsp'], capture_output=True)
    if result.returncode != 0:
        # Start JACK if not running
        subprocess.Popen([
            'jackd', '-d', 'alsa', '-d', 'plughw:1,0', 
            '-r', '48000', '-p', '512', '-n', '2', '-P', '6'
        ])
        time.sleep(2)  # Wait for JACK to start

def start_supercollider(self):
    # Configure SC to use system JACK
    sc_startup = '''
    Server.default.options.device = nil;  // Use system JACK
    Server.default.options.numOutputBusChannels = 6;
    s.boot;
    '''
```

**Benefits of this approach:**
- ✅ All programs can output to specific surround channels
- ✅ SuperCollider generated audio works alongside external audio  
- ✅ Easy to manage with JACK patchbay tools
- ✅ Can save/load routing configurations
- ✅ Real-time connection changes without restarting

**Challenges:**
- ⚠️ Requires JACK knowledge for setup/troubleshooting
- ⚠️ More complex initial configuration
- ⚠️ Need to handle JACK startup/shutdown properly

This would give you full control over routing any audio source to any surround channel while keeping SuperCollider's generated audio integrated into the same surround system.

## Summary: Key Implementation Recommendations

Based on our testing, here are the essential findings for integrating surround sound into Experimance:

### ✅ What Works Reliably
1. **Direct channel routing**: `Out.ar(channel, sig)` - Simple and bulletproof
2. **Stereo pairs**: Separate `Out.ar()` calls for adjacent channels
3. **Audio buses**: Flexible for complex routing scenarios

### ❌ What to Avoid
1. Array outputs: `Out.ar(0, [sig1, sig2, sig3, sig4, sig5, sig6])` - Unreliable
2. Conditional multiplication in SynthDefs - Causes runtime errors
3. Select.ar with nested arrays - Compilation failures

### 🎯 Recommended Implementation Path

**Start Simple**: Add `channel` parameter to existing SynthDefs
```supercollider
// Convert this:
Out.ar(0, [sig, sig]);
// To this:  
Out.ar(channel, sig);
```

**Add Stereo Support**: Use adjacent channel pairs
```supercollider
Out.ar(startChan, sigL);     // Left
Out.ar(startChan + 1, sigR); // Right
```

**Enable Spatial Audio**: Map sand table coordinates to channels
- Front area (y > 0): channels 0,1 (front L/R)
- Rear area (y < 0): channels 4,5 (rear L/R)  
- Center: channel 2, Bass: channel 3

This approach provides immediate multi-channel capability while maintaining backward compatibility with the existing Experimance audio system.
