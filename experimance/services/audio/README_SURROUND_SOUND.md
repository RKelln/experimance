# Surround Sound Implementation Guide

This document contains the essential information for implementing 6-channel surround sound with SuperCollider and jackdbus in the Experimance project.

## Architecture Overview

The Experimance audio system uses a **pure jackdbus approach** with automatic configuration:

- **jackdbus**: Managed entirely via `jack_control` commands
- **6-Channel USB Device**: Automatically detected and configured  
- **SuperCollider**: Connects to existing JACK server with `device = nil`
- **AI Agent**: Uses separate USB speakerphone (not routed through JACK)

## Channel Layout

6-channel surround sound mapping:
- **Channel 0**: Left Front
- **Channel 1**: Right Front  
- **Channel 2**: Center
- **Channel 3**: LFE (Low Frequency Effects/Subwoofer)
- **Channel 4**: Left Surround (Rear Left)
- **Channel 5**: Right Surround (Rear Right)

## Audio Routing Strategy

- **Environmental Audio**: Front channels (0,1) for ambient sounds
- **Music**: Rear channels (4,5) for background music
- **Special Effects**: Center (2) and LFE (3) for emphasis
- **AI Agent Voice**: Separate USB speakerphone device

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

2. **Array outputs with conditionals** - unreliable and causes errors

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

// For stereo pairs, create multiple synths
~frontStereo = [
    Synth(\myExperimentSynth, [\channel, 0, \freq, 440]),  // Left
    Synth(\myExperimentSynth, [\channel, 1, \freq, 440])   // Right
];
```

## Stereo Audio Routing

### Working Approach for Stereo Pairs

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

### Channel Pair Recommendations

For 5.1 surround, use these channel pairs:
- **Front Stereo**: channels 0,1 (Left Front, Right Front)
- **Rear Stereo**: channels 4,5 (Left Rear, Right Rear)  
- **Center + LFE**: channels 2,3 (Center, LFE) - typically mono sources

## Implementation Priority

**Phase 1: Basic Multi-Channel Support** ✅ COMPLETE
1. Add `channel` parameter to all existing SynthDefs
2. Update message handlers to accept channel routing  
3. Test with manual channel assignment

**Phase 2: Stereo Pair Support** ✅ COMPLETE
1. Add stereo SynthDefs with `startChan` parameter
2. Implement stereo pair routing logic
3. Add configuration for stereo placement

**Phase 3: Spatial Audio** 🚧 IN PROGRESS
1. Implement automatic channel calculation from coordinates
2. Add smooth transitions between channels
3. Test with real sand table positioning data

## Current Implementation Status

The Experimance audio system now uses a **simplified architecture**:

✅ **jackdbus Integration**: Pure jackdbus approach with automatic configuration  
✅ **6-Channel Surround**: Environmental audio (front) and music (rear) routing  
✅ **USB Device Support**: Automatic detection and configuration  
✅ **SuperCollider Integration**: Connects to existing JACK server seamlessly  
✅ **Separate AI Audio**: AI agent uses dedicated USB speakerphone device  

**No complex external routing needed** - the system automatically handles all audio routing through jackdbus configuration.

## Summary: Key Implementation Recommendations

Based on our testing, here are the essential findings for integrating surround sound into Experimance:

### ✅ What Works Reliably
1. **Direct channel routing**: `Out.ar(channel, sig)` - Simple and bulletproof
2. **Stereo pairs**: Separate `Out.ar()` calls for adjacent channels
3. **Pure jackdbus architecture**: Automatic configuration and management

### ❌ What to Avoid
1. Array outputs: `Out.ar(0, [sig1, sig2, sig3, sig4, sig5, sig6])` - Unreliable
2. Conditional multiplication in SynthDefs - Causes runtime errors
3. Complex external routing - Not needed with current architecture

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

**Use jackdbus**: Let the audio service handle all JACK configuration automatically

This approach provides immediate multi-channel capability while maintaining backward compatibility with the existing Experimance audio system.
