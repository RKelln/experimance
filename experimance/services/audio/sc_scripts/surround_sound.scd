// Experimance Audio - 5// Your USB 5.1 device detected: card 1: ICUSBAUDIO7D
"Your USB 5.1 device should be at: hw:1,0 or plughw:1,0".postln;
"To use your ICUSBAUDIO7D device:".postln;
"// s.options.device = \"hw:1,0\";        // Direct hardware access".postln;
"// s.options.device = \"plughw:1,0\";    // With ALSA plugin layer (recommended)".postln;
"// Or use: ~setupICUSBAUDIO.() for automatic setup".postln;
"".postln;

"You can also test with:".postln;
"$ speaker-test -D hw:1,0 -c 6 -t wav".postln;
"".postln;

// This script tests a 5.1 surround system by playing different synths on different channels
// Designed to work with USB to 5.1 systems with line out, rear, and center/bass outputs

// -----------------------------------------------------
// Server configuration for 5.1 surround
// -----------------------------------------------------
s.options.numBuffers = 512;
s.options.memSize = 8192 * 4;
s.options.numOutputBusChannels = 6;  // 5.1 surround: L, R, C, LFE, LS, RS
s.options.numInputBusChannels = 2;
//s.options.device = nil;  // Use default device - change if needed
s.options.device = "plughw:1,0";

// List all available audio devices BEFORE booting (if primitive works)
"=== Checking Available Audio Devices ===".postln;
try {
    ServerOptions.devices.do({ |device, i|
        (i.asString ++ ": " ++ device).postln;
    });
} {
    "Device listing primitive failed (common on Linux/JACK systems)".postln;
    "Use ~findLinuxDevices.() after boot to check ALSA devices".postln;
};
"".postln;
"To select a specific device, try one of these formats:".postln;
"// s.options.device = \"hw:1,0\";        // ALSA hardware device".postln;
"// s.options.device = \"plughw:1,0\";    // ALSA with plugin layer".postln;
"// s.options.device = \"USB Audio\";     // Device name".postln;
"".postln;

// Boot server and set up test
s.waitForBoot {
    var testGroup;
    var synths = ();  // Dictionary to store active synths

    "Setting up 5.1 surround sound test...".postln;
    "=== Audio Server Status ===".postln;
    ("Server device: " ++ s.options.device).postln;
    ("Actual output channels: " ++ s.options.numOutputBusChannels).postln;
    ("Sample rate: " ++ s.sampleRate).postln;
    ("Server running: " ++ s.serverRunning).postln;
    "".postln;

    "Channel mapping:".postln;
    "0: Left Front".postln;
    "1: Right Front".postln;
    "2: Center".postln;
    "3: LFE (Low Frequency Effects/Subwoofer)".postln;
    "4: Left Surround (Rear Left)".postln;
    "5: Right Surround (Rear Right)".postln;
    "".postln;

    // Create main group for organization
    testGroup = Group.new;

    // -----------------------------------------------------
    // SynthDefs for different channel tests
    // -----------------------------------------------------

    // Simple sine wave test tone  
    SynthDef(\sine_test, {
        arg freq = 440, amp = 0.3, channel = 0, gate = 1;
        var sig, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sig = SinOsc.ar(freq) * amp * env;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // Pink noise for testing frequency response
    SynthDef(\noise_test, {
        arg amp = 0.2, channel = 0, gate = 1;
        var sig, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sig = PinkNoise.ar(amp) * env;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // Filtered sawtooth for musical testing
    SynthDef(\saw_test, {
        arg freq = 220, cutoff = 1000, amp = 0.3, channel = 0, gate = 1;
        var sig, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sig = Saw.ar(freq);
        sig = LPF.ar(sig, cutoff);
        sig = sig * amp * env;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // Bass test for LFE channel
    SynthDef(\bass_test, {
        arg freq = 60, amp = 0.4, channel = 3, gate = 1;
        var sig, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sig = SinOsc.ar(freq) + (SinOsc.ar(freq * 2) * 0.3);
        sig = LPF.ar(sig, 120);  // Low-pass for subwoofer
        sig = sig * amp * env;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // Percussive test for transient testing
    SynthDef(\perc_test, {
        arg freq = 800, amp = 0.5, channel = 0;
        var sig, env;
        env = EnvGen.kr(Env.perc(0.01, 0.3), doneAction: 2);
        sig = SinOsc.ar(freq) * env * amp;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // Sweep test for frequency response
    SynthDef(\sweep_test, {
        arg startFreq = 100, endFreq = 2000, duration = 3, amp = 0.3, channel = 0;
        var sig, freq, env;
        env = EnvGen.kr(Env.linen(0.1, duration - 0.2, 0.1), doneAction: 2);
        freq = XLine.kr(startFreq, endFreq, duration);
        sig = SinOsc.ar(freq) * amp * env;
        
        // Use explicit Out.ar for the specific channel
        Out.ar(channel, sig);
    }).add;

    // -----------------------------------------------------
    // Stereo Audio Routing Tests
    // -----------------------------------------------------

    // Test how stereo signals work with multi-channel output
    SynthDef(\stereo_test, {
        arg freq = 440, amp = 0.3, startChan = 0, gate = 1;
        var sigL, sigR, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        
        // Create slightly different signals for left and right
        sigL = SinOsc.ar(freq) * amp * env;
        sigR = SinOsc.ar(freq * 1.01) * amp * env;  // Slightly detuned for stereo effect
        
        // Output to two adjacent channels using separate Out.ar calls
        Out.ar(startChan, sigL);       // Left channel
        Out.ar(startChan + 1, sigR);   // Right channel
    }).add;

    // Test mono-to-stereo expansion
    SynthDef(\mono_to_stereo_test, {
        arg freq = 440, amp = 0.3, startChan = 0, gate = 1;
        var sig, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sig = SinOsc.ar(freq) * amp * env;
        
        // Output same mono signal to two channels using separate Out.ar calls
        Out.ar(startChan, sig);       // Left channel
        Out.ar(startChan + 1, sig);   // Right channel (same signal)
    }).add;

    // Test stereo downmix to mono
    SynthDef(\stereo_to_mono_test, {
        arg freq = 440, amp = 0.3, channel = 0, gate = 1;
        var sigL, sigR, mono, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        
        // Create stereo source
        sigL = SinOsc.ar(freq) * amp * env;
        sigR = SinOsc.ar(freq * 1.01) * amp * env;
        
        // Downmix to mono (average L+R)
        mono = (sigL + sigR) * 0.5;
        
        // Output to single channel
        Out.ar(channel, mono);
    }).add;

    // Test using audio buses for routing
    SynthDef(\bus_source, {
        arg freq = 440, amp = 0.3, busnum = 10;
        var sig = SinOsc.ar(freq) * amp;
        Out.ar(busnum, sig);         // Left channel to bus
        Out.ar(busnum + 1, sig * 0.8); // Right channel to bus (slightly different)
    }).add;

    SynthDef(\bus_router, {
        arg inBus = 10, outChan = 0, gate = 1;
        var sigL, sigR, env;
        env = EnvGen.kr(Env.asr(0.1, 1, 0.5), gate, doneAction: 2);
        sigL = In.ar(inBus, 1);     // Read left from bus
        sigR = In.ar(inBus + 1, 1); // Read right from bus
        Out.ar(outChan, sigL * env);     // Output left to specified channel
        Out.ar(outChan + 1, sigR * env); // Output right to next channel
    }).add;

    s.sync;  // Wait for SynthDefs to be added

    // -----------------------------------------------------
    // Test Functions
    // -----------------------------------------------------

    // Test individual channels with sine waves
    ~testChannel = { |channel, freq = 440, duration = 2|
        var channelNames = ["Left Front", "Right Front", "Center", "LFE", "Left Rear", "Right Rear"];
        ("Testing channel " ++ channel ++ ": " ++ channelNames[channel]).postln;

        synths[\current] = Synth(\sine_test, [
            \freq, freq,
            \channel, channel,
            \amp, if(channel == 3, 0.6, 0.4)  // Louder for LFE
        ], testGroup);

        // Stop after duration
        SystemClock.sched(duration, {
            synths[\current].set(\gate, 0);
            synths[\current] = nil;
        });
    };

    // Test all channels sequentially
    ~testAllChannels = { |duration = 2|
        "Testing all channels sequentially...".postln;
        6.do({ |i|
            SystemClock.sched(i * (duration + 0.5), {
                ~testChannel.(i, 440 + (i * 100), duration);
            });
        });
    };

    // Test stereo pairs
    ~testStereoPairs = {
        "Testing stereo pairs...".postln;

        // Front stereo (L/R)
        "Front stereo pair".postln;
        synths[\left] = Synth(\sine_test, [\freq, 440, \channel, 0, \amp, 0.4], testGroup);
        synths[\right] = Synth(\sine_test, [\freq, 554, \channel, 1, \amp, 0.4], testGroup);

        SystemClock.sched(3, {
            synths[\left].set(\gate, 0);
            synths[\right].set(\gate, 0);

            // Rear stereo (LS/RS)
            "Rear stereo pair".postln;
            synths[\leftRear] = Synth(\sine_test, [\freq, 330, \channel, 4, \amp, 0.4], testGroup);
            synths[\rightRear] = Synth(\sine_test, [\freq, 415, \channel, 5, \amp, 0.4], testGroup);

            SystemClock.sched(3, {
                synths[\leftRear].set(\gate, 0);
                synths[\rightRear].set(\gate, 0);
            });
        });
    };

    // Test with different synth types
    ~testSynthTypes = {
        "Testing different synth types on different channels...".postln;

        synths[\sine] = Synth(\sine_test, [\freq, 440, \channel, 0], testGroup);
        synths[\saw] = Synth(\saw_test, [\freq, 330, \cutoff, 800, \channel, 1], testGroup);
        synths[\noise] = Synth(\noise_test, [\channel, 2, \amp, 0.15], testGroup);
        synths[\bass] = Synth(\bass_test, [\freq, 50, \channel, 3], testGroup);

        SystemClock.sched(4, {
            synths[\sine].set(\gate, 0);
            synths[\saw].set(\gate, 0);
            synths[\noise].set(\gate, 0);
            synths[\bass].set(\gate, 0);
        });
    };

    // Frequency sweep test for each channel
    ~testFrequencySweep = { |channel|
        var channelNames = ["Left Front", "Right Front", "Center", "LFE", "Left Rear", "Right Rear"];
        ("Frequency sweep on " ++ channelNames[channel]).postln;

        Synth(\sweep_test, [
            \startFreq, if(channel == 3, 30, 200),  // Lower for LFE
            \endFreq, if(channel == 3, 120, 2000),
            \duration, 4,
            \channel, channel,
            \amp, if(channel == 3, 0.5, 0.3)
        ], testGroup);
    };

    // Percussive test around the room
    ~testPercussiveCircle = {
        var freqs = [800, 900, 700, 400, 600, 750];  // Different freq for each channel
        "Percussive test moving around the room...".postln;

        6.do({ |i|
            SystemClock.sched(i * 0.5, {
                Synth(\perc_test, [\freq, freqs[i], \channel, i, \amp, 0.6], testGroup);
            });
        });
    };

    // Test functions for stereo routing
    ~testStereoRouting = {
        "=== TESTING STEREO AUDIO ROUTING ===".postln;
        "".postln;
        
        "Testing stereo pair on front channels (0,1)...".postln;
        synths[\stereo] = Synth(\stereo_test, [\startChan, 0, \freq, 440]);
        
        SystemClock.sched(3, {
            synths[\stereo].set(\gate, 0);
            "Testing stereo pair on rear channels (4,5)...".postln;
            synths[\stereo] = Synth(\stereo_test, [\startChan, 4, \freq, 660]);
        });
        
        SystemClock.sched(6, {
            synths[\stereo].set(\gate, 0);
            "Testing complete.".postln;
        });
    };

    ~testMonoToStereo = {
        "=== TESTING MONO TO STEREO EXPANSION ===".postln;
        "".postln;
        
        "Playing mono signal as stereo on front channels...".postln;
        synths[\mono] = Synth(\mono_to_stereo_test, [\startChan, 0, \freq, 330]);
        
        SystemClock.sched(3, {
            synths[\mono].set(\gate, 0);
            "Playing mono signal as stereo on rear channels...".postln;
            synths[\mono] = Synth(\mono_to_stereo_test, [\startChan, 4, \freq, 550]);
        });
        
        SystemClock.sched(6, {
            synths[\mono].set(\gate, 0);
            "Testing complete.".postln;
        });
    };

    ~testStereoToMono = {
        "=== TESTING STEREO TO MONO DOWNMIX ===".postln;
        "".postln;
        
        "Playing stereo source downmixed to center channel...".postln;
        synths[\downmix] = Synth(\stereo_to_mono_test, [\channel, 2, \freq, 440]);
        
        SystemClock.sched(3, {
            synths[\downmix].set(\gate, 0);
            "Playing stereo source downmixed to LFE channel...".postln;
            synths[\downmix] = Synth(\stereo_to_mono_test, [\channel, 3, \freq, 220]);
        });
        
        SystemClock.sched(6, {
            synths[\downmix].set(\gate, 0);
            "Testing complete.".postln;
        });
    };

    ~testAudioBuses = {
        "=== TESTING AUDIO BUS ROUTING ===".postln;
        "".postln;
        
        "Creating audio source on bus 10, routing to front channels...".postln;
        synths[\source] = Synth(\bus_source, [\busnum, 10, \freq, 440]);
        synths[\router] = Synth(\bus_router, [\inBus, 10, \outChan, 0], synths[\source], \addAfter);
        
        SystemClock.sched(3, {
            synths[\router].set(\gate, 0);
            "Routing same bus to rear channels...".postln;
            synths[\router] = Synth(\bus_router, [\inBus, 10, \outChan, 4], synths[\source], \addAfter);
        });
        
        SystemClock.sched(6, {
            synths[\router].set(\gate, 0);
            synths[\source].free;
            "Bus routing test complete.".postln;
        });
    };

    ~testAllStereoMethods = {
        "=== COMPREHENSIVE STEREO ROUTING TEST ===".postln;
        "This will test various methods of handling stereo audio in multi-channel setups.".postln;
        "".postln;
        
        SystemClock.sched(1, { ~testStereoRouting.(); });
        SystemClock.sched(8, { ~testMonoToStereo.(); });
        SystemClock.sched(16, { ~testStereoToMono.(); });
        SystemClock.sched(24, { ~testAudioBuses.(); });
        
        SystemClock.sched(32, {
            "".postln;
            "=== ALL STEREO TESTS COMPLETE ===".postln;
            "Key findings:".postln;
            "1. Out.ar(startChan, sigL); Out.ar(startChan+1, sigR) works for stereo pairs".postln;
            "2. Mono-to-stereo: use same signal on two adjacent channels".postln;
            "3. Stereo-to-mono: use (sigL + sigR) * 0.5 for proper downmix".postln;
            "4. Audio buses allow flexible routing between synths".postln;
        });
    };

    // Stop all active synths
    ~stopAll = {
        "Stopping all test synths...".postln;
        testGroup.freeAll;
        synths.keysValuesDo({ |key, synth|
            if(synth.notNil, { synth.free });
        });
        synths.clear;
    };

    // -----------------------------------------------------
    // Interactive Commands
    // -----------------------------------------------------

    "".postln;
    "=== 5.1 Surround Sound Test Commands ===".postln;
    "".postln;
    "SUCCESS! ALL 6 CHANNELS CONNECTED IN SUPERCOLLIDER'S JACK:".postln;
    "~confirmSixChannelConnection.(); // Confirm the connection success".postln;
    "~testRearSpeakersOnly.();       // Test rear speakers specifically".postln;
    "~testChannelMapping.();         // Test each channel mapping".postln;
    "~testPhysicalWiring.();         // Check physical USB device wiring".postln;
    "~testUSBDeviceChannelOrder.();  // Test alternative channel orders".postln;
    "".postln;
    "General Troubleshooting:".postln;
    "~debugAudio.();             // Show detailed audio diagnostics".postln;
    "~scAudioDeepDive.();        // SuperCollider-specific diagnostics".postln;
    "~verboseRestart.();         // Restart with verbose device info".postln;
    "~checkSCAudioOutput.();     // Check if SC outputs ANY audio".postln;
    "~tryDifferentDrivers.();    // Try different driver methods".postln;
    "~showActualDevice.();       // Show what device SC actually uses".postln;
    "".postln;
    "Device Selection (ICUSBAUDIO7D detected):".postln;
    "~setupICUSBAUDIO.();        // Quick setup for your ICUSBAUDIO7D".postln;
    "~testICUSBAUDIO.();         // Test ICUSBAUDIO7D connection".postln;
    "".postln;
    "General Device Selection:".postln;
    "~checkAudioSetup.();        // Check current audio setup".postln;
    "~findLinuxDevices.();       // List Linux/ALSA audio devices".postln;
    "~setAudioDevice.(\"plughw:1,0\");   // Set to your USB device".postln;
    "".postln;
    "Basic Tests:".postln;
    "~testChannel.(0);           // Test channel 0 (Left Front)".postln;
    "~testChannel.(1);           // Test channel 1 (Right Front)".postln;
    "~testChannel.(2);           // Test channel 2 (Center)".postln;
    "~testChannel.(3);           // Test channel 3 (LFE/Sub)".postln;
    "~testChannel.(4);           // Test channel 4 (Left Rear)".postln;
    "~testChannel.(5);           // Test channel 5 (Right Rear)".postln;
    "".postln;
    "Automated Tests:".postln;
    "~testAllChannels.();        // Test all channels sequentially".postln;
    "~testStereoPairs.();        // Test stereo pairs".postln;
    "~testSynthTypes.();         // Test different synth types".postln;
    "~testPercussiveCircle.();   // Percussive test around room".postln;
    "".postln;
    "Frequency Tests:".postln;
    "~testFrequencySweep.(0);    // Frequency sweep on channel 0".postln;
    "~testFrequencySweep.(3);    // Frequency sweep on LFE channel".postln;
    "".postln;
    "Control:".postln;
    "~stopAll.();                // Stop all test synths".postln;
    "".postln;
    "Manual synth creation:".postln;
    "x = Synth(\\sine_test, [\\freq, 440, \\channel, 0]);".postln;
    "x.set(\\gate, 0);           // Stop the synth".postln;
    "".postln;

    // Optionally start with a basic test
    "Starting with a quick channel test in 2 seconds...".postln;
    "Type ~stopAll.() to stop, or ~testAllChannels.() to run full test".postln;

    SystemClock.sched(2, {
        ~testChannel.(0, 440, 1);  // Quick test of left front
    });
};

// -----------------------------------------------------
// Additional utility functions (run after boot)
// -----------------------------------------------------

// Check audio device information
~checkAudioSetup = {
    "=== Current Audio Setup ===".postln;
    ("Device: " ++ s.options.device).postln;
    ("Output channels: " ++ s.options.numOutputBusChannels).postln;
    ("Input channels: " ++ s.options.numInputBusChannels).postln;
    ("Sample rate: " ++ s.sampleRate).postln;
    ("Server running: " ++ s.serverRunning).postln;
    "".postln;

    "=== Available Devices ===".postln;
    try {
        ServerOptions.devices.do({ |device, i|
            (i.asString ++ ": " ++ device).postln;
        });
    } {
        "Device primitive failed. Use ~findLinuxDevices.() for ALSA device info".postln;
    };
    "".postln;
};

// Linux-specific device detection using system commands
~findLinuxDevices = {
    "=== Linux Audio Device Detection ===".postln;
    "Checking ALSA devices...".postln;

    // Use unixCmd to run aplay -l and capture output
    "Running 'aplay -l' to list playback devices:".postln;
    "aplay -l".unixCmd;
    "".postln;

    "Running 'cat /proc/asound/cards' to list sound cards:".postln;
    "cat /proc/asound/cards".unixCmd;
    "".postln;

    "Common device formats to try:".postln;
    "hw:0,0  - First card, first device (usually built-in)".postln;
    "hw:1,0  - Second card, first device (often USB)".postln;
    "hw:2,0  - Third card, first device".postln;
    "plughw:1,0 - USB device with ALSA plugin layer (recommended)".postln;
    "".postln;

    "To test a device outside SuperCollider:".postln;
    "speaker-test -D hw:1,0 -c 6 -t wav".postln;
    "".postln;
};

// Set specific audio device (run this before booting if needed)
~setAudioDevice = { |deviceName|
    if(s.serverRunning, {
        "Server is running. Stopping server to change device...".postln;
        s.quit;
        AppClock.sched(2, {
            s.options.device = deviceName;
            ("Audio device set to: " ++ deviceName).postln;
            "Rebooting server...".postln;
            s.boot;
        });
    }, {
        s.options.device = deviceName;
        ("Audio device set to: " ++ deviceName).postln;
        "Boot server with s.boot or re-evaluate this script".postln;
    });
};

// Quick device switcher - try common USB device names
~findUSBDevices = {
    var commonDevices = [
        "hw:1,0",
        "hw:2,0",
        "plughw:1,0",
        "plughw:2,0",
        "hw:CARD=Device,DEV=0",
        "USB Audio",
        "USB Audio Device"
    ];
	"=== Trying Common USB Device Names ===".postln;
    "Since device listing failed, trying common ALSA device names...".postln;


    "Available formats to try:".postln;
    commonDevices.do({ |device, i|
        (i.asString ++ ": ~setAudioDevice.(\"" ++ device ++ "\");").postln;
    });
    "".postln;

    "Use ~findLinuxDevices.() to see actual system devices".postln;
    "".postln;
};

// Test specific device without changing global settings
~testDeviceConnection = { |deviceName|
    var testServer = Server(\testServer);

	("Testing connection to device: " ++ deviceName).postln;
    "This will attempt to boot a temporary server with your device...".postln;

    testServer.options.device = deviceName;
    testServer.options.numOutputBusChannels = 6;

    testServer.boot;

    AppClock.sched(3, {
        if(testServer.serverRunning, {
            ("SUCCESS: Device '" ++ deviceName ++ "' is working!").postln;
            "You can now use: ~setAudioDevice.(\"" ++ deviceName ++ "\");".postln;
        }, {
            ("FAILED: Device '" ++ deviceName ++ "' could not be opened").postln;
        });

        testServer.quit;
    });
};

// Quick setup for ICUSBAUDIO7D device
~setupICUSBAUDIO = {
    "Setting up ICUSBAUDIO7D USB 5.1 device...".postln;
    ~setAudioDevice.("hw:1,0");  // Use plugin layer for better compatibility
};

// Test ICUSBAUDIO7D device connection
~testICUSBAUDIO = {
    "Testing ICUSBAUDIO7D device connection...".postln;
    ~testDeviceConnection.("hw:1,0");

    AppClock.sched(4, {
        "If that worked, use ~setupICUSBAUDIO.() to switch to your USB device".postln;
        "Then run the surround tests: ~testAllChannels.()".postln;
    });
};

// Alternative device names to try for USB 5.1
~tryUSBDevice = {
    "=== Quick USB Device Test ===".postln;
    "Trying hw:1,0 (most common USB device location)...".postln;
    ~testDeviceConnection.("hw:1,0");

    AppClock.sched(5, {
        "If that failed, try: ~testDeviceConnection.(\"plughw:1,0\");".postln;
        "Or use ~findLinuxDevices.() to see all available devices".postln;
    });
};

/*
~tryUSBDevice.();
~setupICUSBAUDIO.();
~testAllChannels.();
~findLinuxDevices.();
*/

// Audio debugging and diagnostics functions
~debugAudio = {
    "=== AUDIO DEBUGGING INFORMATION ===".postln;
    "".postln;

    "Server Status:".postln;
    ("- Server running: " ++ s.serverRunning).postln;
    ("- Device setting: " ++ s.options.device).postln;
    ("- Output channels: " ++ s.options.numOutputBusChannels).postln;
    ("- Input channels: " ++ s.options.numInputBusChannels).postln;
    ("- Sample rate: " ++ s.sampleRate).postln;
    ("- Block size: " ++ s.options.blockSize).postln;
    "".postln;

    "Audio Driver:".postln;
    if(s.serverRunning, {
        ("- Using audio driver").postln;
        "- Check post window for JACK/ALSA connection messages".postln;
    }, {
        "- Server not running - cannot check driver status".postln;
    });
    "".postln;

    "Quick Tests:".postln;
    "1. ~testSimpleBeep.()     // Test basic audio output".postln;
    "2. ~testChannelRouting.() // Test if audio is routing correctly".postln;
    "3. ~testVolumeLevel.()    // Test with different volume levels".postln;
    "".postln;
};

// Simple beep test - should work if audio is routing
~testSimpleBeep = {
    "Testing simple beep on default output (channels 0-1)...".postln;
    "If you hear this, SuperCollider audio is working but may be routing wrong".postln;

    {
        var sig = SinOsc.ar(800) * 0.3 * EnvGen.kr(Env.perc(0.1, 1), doneAction: 2);
        Out.ar(0, sig.dup);  // Output to both L/R
    }.play;
};

// Test specific channel routing with verbose feedback
~testChannelRouting = {
    "Testing if audio is reaching the USB device...".postln;
    "This will play a 2-second tone on each channel with announcements".postln;
    "Listen carefully to your USB device outputs!".postln;

    6.do({ |channel|
        SystemClock.sched(channel * 3, {
            ("TESTING CHANNEL " ++ channel ++ " - Listen for tone on USB device!").postln;

            {
                var sig = SinOsc.ar(440 + (channel * 100)) * 0.5;
                var env = EnvGen.kr(Env.linen(0.1, 1.8, 0.1), doneAction: 2);
                Out.ar(channel, sig * env);
            }.play;
        });
    });

    SystemClock.sched(20, {
        "Channel routing test complete. Did you hear tones from your USB device?".postln;
        "If not, try: ~switchToPlughw.() or check audio connections".postln;
    });
};

// Test with different volume levels
~testVolumeLevel = {
    "Testing volume levels - this should be LOUD if working...".postln;

    {
        var sig = SinOsc.ar(800) * 0.8;  // Loud volume
        var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
        Out.ar(0, sig * env);  // Left channel
    }.play;

    SystemClock.sched(3, {
        {
            var sig = SinOsc.ar(1000) * 0.8;  // Loud volume
            var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
            Out.ar(1, sig * env);  // Right channel
        }.play;
    });
};

// Force switch to plughw (often more compatible)
~switchToPlughw = {
    "Switching to plughw:1,0 (ALSA plugin layer)...".postln;
    "This is often more compatible than direct hw: access".postln;
    ~setAudioDevice.("plughw:1,0");
};

// Check if PulseAudio is interfering
~checkPulseAudio = {
    "=== PulseAudio Interference Check ===".postln;
    "PulseAudio can sometimes block direct ALSA access.".postln;
    "If you're having issues, try these terminal commands:".postln;
    "".postln;
    "# Temporarily stop PulseAudio:".postln;
    "pulseaudio --kill".postln;
    "".postln;
    "# Check if device is busy:".postln;
    "lsof /dev/snd/*".postln;
    "".postln;
    "# Restart PulseAudio (if needed later):".postln;
    "pulseaudio --start".postln;
    "".postln;
    "Try ~testSimpleBeep.() after stopping PulseAudio".postln;
};

// Emergency audio test - bypass all fancy routing
~emergencyTest = {
    "EMERGENCY AUDIO TEST - Simple direct output".postln;
    "If this doesn't work, there's a fundamental audio issue".postln;

    // Super simple synthesis
    {SinOsc.ar(440, 0, 0.5) * EnvGen.kr(Env.perc(0.01, 1), doneAction: 2)}.play;
};

// Advanced audio routing diagnostics
~checkAudioRouting = {
    "=== ADVANCED AUDIO ROUTING DIAGNOSTICS ===".postln;
    "".postln;

    "SuperCollider believes it's using: plughw:1,0".postln;
    "But audio is coming from laptop speakers instead of USB device.".postln;
    "".postln;

    "This suggests one of these issues:".postln;
    "1. ALSA is routing plughw:1,0 to wrong device".postln;
    "2. PulseAudio is intercepting and rerouting audio".postln;
    "3. USB device is not the default for ALSA".postln;
    "4. SuperCollider/JACK is connected to wrong outputs".postln;
    "".postln;

    "Run these terminal commands to investigate:".postln;
    "".postln;
    "# Check what's actually playing audio:".postln;
    "lsof /dev/snd/* | grep supercollider".postln;
    "".postln;
    "# Check ALSA routing for plughw:1,0:".postln;
    "aplay -D plughw:1,0 /usr/share/sounds/alsa/Front_Left.wav".postln;
    "".postln;
    "# Check if PulseAudio is running:".postln;
    "pulseaudio --check -v".postln;
    "".postln;
    "# List active audio streams:".postln;
    "pactl list short sinks".postln;
    "pactl list short sink-inputs".postln;
    "".postln;
};

// Force SuperCollider to bypass all audio systems
~forceDirectDevice = {
    "Attempting to force direct device access...".postln;
    "This will restart the server with aggressive device settings".postln;

    s.quit;

    AppClock.sched(2, {
        // Set very explicit options
        s.options.device = "hw:1,0";  // Direct hardware, no plugin layer
        s.options.numOutputBusChannels = 6;
        s.options.numInputBusChannels = 0;  // Disable inputs to avoid conflicts
        s.options.sampleRate = 48000;
        s.options.blockSize = 1024;  // Larger block size

        "Rebooting with direct hardware access...".postln;
        s.boot;

        AppClock.sched(3, {
            "Testing direct hardware connection...".postln;
            ~testSimpleBeep.();
        });
    });
};

// Check if JACK is interfering
~checkJACK = {
    "=== JACK Audio System Check ===".postln;
    "".postln;

    "JACK can sometimes route audio differently than expected.".postln;
    "Check these in terminal:".postln;
    "".postln;
    "# Check if JACK is running:".postln;
    "jack_lsp".postln;
    "".postln;
    "# If JACK is running, check connections:".postln;
    "jack_lsp -c".postln;
    "".postln;
    "# Kill JACK if needed:".postln;
    "killall jackd".postln;
    "".postln;

    "If JACK is routing audio, use qjackctl or jack_connect to route".postln;
    "SuperCollider outputs to your USB device inputs".postln;
};

// Test with explicit channel mapping
~testExplicitChannels = {
    "Testing explicit channel output to USB device...".postln;
    "Playing LOUD tones directly to USB hardware channels".postln;
    "If you don't hear this on USB speakers, there's a system routing issue".postln;

    // Test each channel with maximum clarity
    6.do({ |channel|
        SystemClock.sched(channel * 2, {
            ("=== CHANNEL " ++ channel ++ " - SHOULD BE LOUD ON USB DEVICE ===").postln;

            // Create a very obvious, loud tone
            {
                var freq = 200 + (channel * 200);  // Different freq per channel
                var sig = SinOsc.ar(freq) * 0.8;   // Loud volume
                var env = EnvGen.kr(Env.perc(0.1, 1.5), doneAction: 2);

                // Force output to specific channel only
                var output = Array.fill(6, 0);
                output[channel] = sig * env;

                Out.ar(0, output);
            }.play;
        });
    });
};

// Kill all audio processes and restart clean
~audioReset = {
    "=== NUCLEAR AUDIO RESET ===".postln;
    "This will kill PulseAudio, JACK, and restart SuperCollider".postln;
    "".postln;

    "Run these commands in terminal:".postln;
    "sudo killall pulseaudio".postln;
    "sudo killall jackd".postln;
    "sudo killall qjackctl".postln;
    "".postln;
    "Then restart this SuperCollider script".postln;
};

// Test if audio is going to default ALSA device instead
~testALSADefault = {
    "Testing if ALSA is using wrong default device...".postln;
    "".postln;

    "Check your ALSA configuration:".postln;
    "cat ~/.asoundrc     # User ALSA config".postln;
    "cat /etc/asound.conf # System ALSA config".postln;
    "".postln;

    "Check current default device:".postln;
    "aplay -l | grep -E '^card|default'".postln;
    "".postln;

    "You might need to create ~/.asoundrc with:".postln;
    "pcm.!default {".postln;
    "    type hw".postln;
    "    card 1".postln;
    "    device 0".postln;
    "}".postln;
    "ctl.!default {".postln;
    "    type hw".postln;
    "    card 1".postln;
    "}".postln;
};

// SuperCollider-specific audio routing debugging
~scAudioDeepDive = {
    "=== SUPERCOLLIDER AUDIO DEEP DIVE ===".postln;
    "".postln;

    "CONFIRMED: ALSA routing works (aplay to USB device successful)".postln;
    "PROBLEM: SuperCollider is not using the same audio path".postln;
    "".postln;

    "SuperCollider Server Details:".postln;
    if(s.serverRunning, {
        ("- Server address: " ++ s.addr).postln;
        ("- Server name: " ++ s.name).postln;
        ("- Options device: " ++ s.options.device).postln;
        ("- Actual device: " ++ s.options.device).postln;
        ("- Audio driver: Check post window for 'creating alsa driver' message").postln;
        "".postln;

        "Look in the post window for the server boot message.".postln;
        "It should show: 'creating alsa driver ... hw:1,0|hw:1,0|...'".postln;
        "If it shows 'hw:0,0' or something else, SC is not using your USB device!".postln;
    }, {
        "Server not running - cannot check actual device connection".postln;
    });
    "".postln;
};

// Force SuperCollider to show exactly what device it's using
~showActualDevice = {
    "Checking what device SuperCollider actually opened...".postln;
    "Look at the POST WINDOW for the 'creating alsa driver' line from server boot.".postln;
    "It should show: plughw:1,0 or hw:1,0".postln;
    "".postln;

    "If it shows hw:0,0, then SuperCollider ignored your device setting!".postln;
    "".postln;

    "Current server options:".postln;
    ("- device: " ++ s.options.device).postln;
    ("- numOutputBusChannels: " ++ s.options.numOutputBusChannels).postln;
    ("- sampleRate: " ++ s.options.sampleRate).postln;
    "".postln;
};

// Test with verbose server restart to see actual device connection
~verboseRestart = {
    "Restarting server with verbose output to see actual device connection...".postln;
    "WATCH THE POST WINDOW for 'creating alsa driver' message!".postln;
    "".postln;

    s.quit;

    AppClock.sched(2, {
        "Booting with device: plughw:1,0".postln;
        s.options.device = "plughw:1,0";
        s.options.numOutputBusChannels = 6;
        s.options.verbosity = 1;  // More verbose output

        s.boot;

        AppClock.sched(3, {
            "Server booted. Check post window for actual device used.".postln;
            "Look for line like: 'creating alsa driver ... plughw:1,0|plughw:1,0|...'".postln;
            "".postln;

            // Test immediately
            "Testing audio output...".postln;
            ~testSimpleBeep.();
        });
    });
};

// Try different SuperCollider audio driver approaches
~tryDifferentDrivers = {
    "Trying different SuperCollider audio driver configurations...".postln;
    "".postln;

    "Method 1: Direct hardware with explicit rate".postln;
    "~setDeviceMethod1.()".postln;
    "".postln;

    "Method 2: Plugin layer with buffer tweaks".postln;
    "~setDeviceMethod2.()".postln;
    "".postln;

    "Method 3: Force ALSA driver explicitly".postln;
    "~setDeviceMethod3.()".postln;
};

~setDeviceMethod1 = {
    "Trying Method 1: Direct hardware access".postln;
    s.quit;
    AppClock.sched(2, {
        s.options.device = "hw:1,0";
        s.options.numOutputBusChannels = 6;
        s.options.sampleRate = 48000;
        s.options.blockSize = 1024;
        s.options.numBuffers = 1024;
        s.boot;
        AppClock.sched(3, { ~testSimpleBeep.(); });
    });
};

~setDeviceMethod2 = {
    "Trying Method 2: Plugin layer with tweaks".postln;
    s.quit;
    AppClock.sched(2, {
        s.options.device = "plughw:1,0";
        s.options.numOutputBusChannels = 6;
        s.options.sampleRate = nil;  // Let ALSA choose
        s.options.blockSize = 64;
        s.options.numBuffers = 512;
        s.boot;
        AppClock.sched(3, { ~testSimpleBeep.(); });
    });
};

~setDeviceMethod3 = {
    "Trying Method 3: Force specific ALSA parameters".postln;
    s.quit;
    AppClock.sched(2, {
        // Try with very explicit ALSA device string
        s.options.device = "plughw:CARD=ICUSBAUDIO7D,DEV=0";
        s.options.numOutputBusChannels = 6;
        s.options.sampleRate = 48000;
        s.boot;
        AppClock.sched(3, { ~testSimpleBeep.(); });
    });
};

// Check if SuperCollider is actually outputting audio at all
~checkSCAudioOutput = {
    "Checking if SuperCollider is producing any audio output...".postln;
    "".postln;

    "This will play audio on ALL possible outputs simultaneously:".postln;
    "- Default system output".postln;
    "- USB device (if properly routed)".postln;
    "- Any other connected audio devices".postln;
    "".postln;

    "Listen carefully to ALL your speakers/headphones...".postln;

    {
        var sig = SinOsc.ar([400, 500, 600, 700, 800, 900]) * 0.3;
        var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
        Out.ar(0, sig * env);  // Output different tones to all 6 channels
    }.play;

    SystemClock.sched(3, {
        "Did you hear multiple tones from any device?".postln;
        "If yes: SuperCollider audio works, just routing wrong".postln;
        "If no: SuperCollider audio is completely broken".postln;
    });
};

// JACK is overriding your device setting! Force ALSA-only mode
~forceALSAMode = {
    "=== FORCING ALSA-ONLY MODE (NO JACK) ===".postln;
    "JACK is ignoring your USB device and using laptop audio instead.".postln;
    "Switching to pure ALSA mode...".postln;
    "".postln;

    s.quit;

    AppClock.sched(3, {
        // Kill any running JACK processes
        "Killing JACK processes...".postln;
        "killall jackd".unixCmd;
        "killall qjackctl".unixCmd;

        AppClock.sched(2, {
            // Configure for ALSA-only
            s.options.device = "plughw:1,0";
            s.options.numOutputBusChannels = 6;
            s.options.numInputBusChannels = 2;
            s.options.sampleRate = 48000;
            s.options.blockSize = 1024;

            // Force ALSA driver (not JACK)
            "Booting with ALSA driver (no JACK)...".postln;
            "If this works, you should see 'plughw:1,0' in the driver creation line".postln;

            s.boot;

            AppClock.sched(4, {
                "Testing ALSA-only audio...".postln;
                ~testSimpleBeep.();
            });
        });
    });
};

// Configure JACK to use your USB device instead
~configureJACKForUSB = {
    "=== CONFIGURING JACK FOR USB DEVICE ===".postln;
    "This will stop JACK and restart it with your USB device".postln;
    "".postln;

    s.quit;

    AppClock.sched(2, {
        "Stopping current JACK...".postln;
        "killall jackd".unixCmd;

        AppClock.sched(2, {
            "Starting JACK with USB device...".postln;
            "Run this command in terminal:".postln;
            "jackd -d alsa -d hw:1,0 -r 48000 -p 1024 -n 2".postln;
            "".postln;
            "Then restart this SuperCollider script".postln;
            "".postln;

            "Or use qjackctl (JACK Control) GUI:".postln;
            "1. Open qjackctl".postln;
            "2. Click Setup".postln;
            "3. Set Interface to 'hw:1,0' or 'ICUSBAUDIO7D'".postln;
            "4. Click OK, then Start".postln;
            "5. Restart this SuperCollider script".postln;
        });
    });
};

// Check current JACK status and connections
~checkJACKStatus = {
    "=== CURRENT JACK STATUS ===".postln;
    "".postln;

    "Check JACK connections in terminal:".postln;
    "jack_lsp -c".postln;
    "".postln;

    "Current JACK server info:".postln;
    "jack_sample_rate".postln;
    "jack_buffer_size".postln;
    "".postln;

    "SuperCollider JACK client should be connected to:".postln;
    "- Input: system:capture_1/2 -> SuperCollider:in_1/2".postln;
    "- Output: SuperCollider:out_1/2 -> system:playback_1/2".postln;
    "".postln;

    "But 'system' is currently your laptop audio, not USB!".postln;
    "Need to either:".postln;
    "1. ~forceALSAMode.() - Skip JACK entirely".postln;
    "2. ~configureJACKForUSB.() - Make JACK use USB device".postln;
};

// Simple JACK connection fix (if JACK is running with USB)
~connectJACKToUSB = {
    "Attempting to connect SuperCollider to USB device via JACK...".postln;
    "".postln;

    "This assumes JACK is already running with your USB device".postln;
    "Run these commands in terminal:".postln;
    "".postln;
    "# Disconnect from laptop audio:".postln;
    "jack_disconnect SuperCollider:out_1 system:playback_1".postln;
    "jack_disconnect SuperCollider:out_2 system:playback_2".postln;
    "".postln;
    "# Connect to USB device (check jack_lsp for exact names):".postln;
    "jack_connect SuperCollider:out_1 system:playback_1".postln;
    "jack_connect SuperCollider:out_2 system:playback_2".postln;
    "".postln;
    "Note: The exact connection names depend on how JACK sees your USB device".postln;
};

// Nuclear option - force SuperCollider to use ALSA without JACK
~forceALSANuclear = {
    "=== NUCLEAR ALSA MODE - JACK REFUSES TO DIE ===".postln;
    "SuperCollider keeps auto-starting JACK. Using more aggressive methods...".postln;
    "".postln;

    s.quit;

    AppClock.sched(2, {
        "Step 1: Kill ALL audio processes...".postln;
        "sudo killall -9 jackd".unixCmd;
        "sudo killall -9 qjackctl".unixCmd;
        "sudo killall -9 pulseaudio".unixCmd;

        AppClock.sched(3, {
            "Step 2: Set explicit SuperCollider server options for ALSA...".postln;

            // Try to force ALSA driver explicitly
            Server.default = Server.local;
            s = Server.default;

            // More aggressive options
            s.options.device = "plughw:1,0";
            s.options.numOutputBusChannels = 6;
            s.options.numInputBusChannels = 2;
            s.options.sampleRate = 48000;
            s.options.blockSize = 1024;
            s.options.numBuffers = 1024;
            s.options.memSize = 8192 * 16;

            // Try to disable JACK auto-start
            "export JACK_NO_AUDIO_RESERVATION=1".unixCmd;

            "Step 3: Booting server with ALSA-only options...".postln;
            s.boot;

            AppClock.sched(4, {
                "Check if JACK started again. If yes, try manual method...".postln;
                ~testSimpleBeep.();
            });
        });
    });
};

// Manual SuperCollider server start with explicit ALSA
~manualALSAStart = {
    "=== MANUAL ALSA SERVER START ===".postln;
    "Starting SuperCollider server manually with explicit ALSA command".postln;
    "".postln;

    s.quit;

    AppClock.sched(3, {
        "Kill everything first...".postln;
        "sudo killall -9 jackd qjackctl pulseaudio".unixCmd;

        AppClock.sched(2, {
            "You need to run this command in terminal:".postln;
            "".postln;
            "cd /usr/bin/".postln;
            "./scsynth -u 57110 -a 1024 -i 2 -o 6 -r 48000 -H plughw:1,0 -H plughw:1,0".postln;
            "".postln;
            "Then in SuperCollider run: s.addr = NetAddr(\"localhost\", 57110);".postln;
            "".postln;
            "This bypasses SuperCollider's automatic JACK startup".postln;
        });
    });
};

// Check SuperCollider compilation flags to see why it prefers JACK
~checkSCCompileFlags = {
    "=== SUPERCOLLIDER COMPILATION INFO ===".postln;
    "Checking why SuperCollider auto-starts JACK...".postln;
    "".postln;

    "SuperCollider version and audio drivers:".postln;
    Main.version.postln;
    "".postln;

    "Check what audio drivers are available:".postln;
    "Run in terminal: supercollider --help | grep -i audio".postln;
    "".postln;

    "Check SC configuration:".postln;
    thisProcess.platform.postln;
    "".postln;

    "If SuperCollider was compiled with JACK-preferred, it will always try JACK first".postln;
    "You might need to install a different SuperCollider build or compile from source with ALSA-only".postln;
};

// Try different SuperCollider server startup approach
~tryServerClass = {
    "=== TRYING DIFFERENT SERVER CLASS ===".postln;
    "Creating new server instance with explicit options...".postln;

    // Stop current server
    Server.killAll;

    AppClock.sched(3, {
        // Kill JACK manually
        "killall -9 jackd".unixCmd;

        AppClock.sched(2, {
            // Create new server with different name and explicit options
            var newServer = Server(\alsa_server, NetAddr("localhost", 57111));

            newServer.options.device = "plughw:1,0";
            newServer.options.numOutputBusChannels = 6;
            newServer.options.numInputBusChannels = 2;

            "Trying to boot alternate server on port 57111...".postln;
            newServer.boot;

            AppClock.sched(4, {
                "Testing alternate server...".postln;
                {SinOsc.ar(440, 0, 0.3) * EnvGen.kr(Env.perc, doneAction: 2)}.play(newServer);
            });
        });
    });
};

// Last resort: Configure JACK to use USB device
~configureJACKForUSBDetailed = {
    "=== CONFIGURE JACK FOR USB DEVICE (DETAILED) ===".postln;
    "Since JACK won't go away, let's make it use your USB device...".postln;
    "".postln;

    s.quit;

    AppClock.sched(2, {
        "Step 1: Stop current JACK server".postln;
        "killall jackd".unixCmd;

        AppClock.sched(2, {
            "Step 2: Start JACK with your USB device".postln;
            "Run this command in a terminal:".postln;
            "".postln;
            "jackd -r -d alsa -d hw:1,0 -r 48000 -p 1024 -n 2 -S".postln;
            "".postln;
            "Or install qjackctl and configure it:".postln;
            "sudo apt install qjackctl".postln;
            "qjackctl".postln;
            "".postln;
            "In qjackctl:".postln;
            "1. Click 'Setup'".postln;
            "2. Set Driver: alsa".postln;
            "3. Set Interface: hw:1,0 (or select ICUSBAUDIO7D)".postln;
            "4. Set Sample Rate: 48000".postln;
            "5. Set Frames/Period: 1024".postln;
            "6. Set Periods/Buffer: 2".postln;
            "7. Click OK, then Start".postln;
            "".postln;
            "Then restart this SuperCollider script".postln;
        });
    });
};

// Diagnose and fix missing rear speakers in JACK 5.1 setup
~checkJACK6ChannelSetup = {
    "=== CHECKING JACK 6-CHANNEL SETUP ===".postln;
    "You're getting: Front L/R + Center + LFE (4 channels)".postln;
    "Missing: Rear L/R (channels 4 & 5)".postln;
    "".postln;

    "Check JACK configuration in terminal:".postln;
    "jack_lsp".postln;
    "".postln;

    "You should see 6 system playback ports:".postln;
    "system:playback_1 (Front L)".postln;
    "system:playback_2 (Front R)".postln;
    "system:playback_3 (Center)".postln;
    "system:playback_4 (LFE)".postln;
    "system:playback_5 (Rear L)".postln;
    "system:playback_6 (Rear R)".postln;
    "".postln;

    "If you only see playback_1 through playback_4, JACK is not".postln;
    "configured for 6-channel output!".postln;
    "".postln;

    "Run: ~fixJACK6Channel.() to reconfigure JACK for 6 channels".postln;
};

~fixJACK6Channel = {
    "=== FIXING JACK FOR 6-CHANNEL OUTPUT ===".postln;
    "Reconfiguring JACK to expose all 6 USB device channels...".postln;
    "".postln;

    s.quit;

    AppClock.sched(2, {
        "Stopping JACK...".postln;
        "killall jackd".unixCmd;

        AppClock.sched(2, {
            "Starting JACK with explicit 6-channel configuration:".postln;
            "".postln;
            "Run this command in terminal:".postln;
            "jackd -d alsa -d hw:1,0 -r 48000 -p 1024 -n 2 -i 2 -o 6".postln;
            "".postln;
            "Key parameters:".postln;
            "-i 2   = 2 input channels".postln;
            "-o 6   = 6 output channels (important!)".postln;
            "-d hw:1,0 = your USB device".postln;
            "".postln;

            "Or in qjackctl:".postln;
            "1. Setup -> Advanced".postln;
            "2. Set 'Output Channels' to 6".postln;
            "3. Set 'Input Channels' to 2".postln;
            "4. Restart JACK".postln;
            "".postln;

            "After restarting JACK, run: jack_lsp".postln;
            "You should see system:playback_1 through playback_6".postln;
        });
    });
};

~testRearChannelsSpecifically = {
    "=== TESTING REAR CHANNELS SPECIFICALLY ===".postln;
    "Playing tones ONLY on rear channels (4 & 5)...".postln;
    "".postln;

    // Test rear left (channel 4)
    "Testing REAR LEFT (channel 4) in 3 seconds...".postln;
    SystemClock.sched(3, {
        "=== REAR LEFT SHOULD BE PLAYING NOW ===".postln;
        {
            var sig = SinOsc.ar(300) * 0.7;  // Low freq for rear left
            var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
            var output = Array.fill(6, 0);
            output[4] = sig * env;  // Rear left = channel 4
            Out.ar(0, output);
        }.play;
    });

    // Test rear right (channel 5)
    SystemClock.sched(6, {
        "=== REAR RIGHT SHOULD BE PLAYING NOW ===".postln;
        {
            var sig = SinOsc.ar(400) * 0.7;  // Higher freq for rear right
            var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
            var output = Array.fill(6, 0);
            output[5] = sig * env;  // Rear right = channel 5
            Out.ar(0, output);
        }.play;
    });

    SystemClock.sched(9, {
        "Rear channel test complete.".postln;
        "Did you hear the rear speakers?".postln;
        "If not, the issue might be:".postln;
        "1. Physical wiring of your USB device".postln;
        "2. USB device channel mapping".postln;
        "3. Your speakers aren't connected to rear outputs".postln;
        "".postln;
        "Try: ~testPhysicalWiring.()".postln;
    });
};

~testPhysicalWiring = {
    "=== TESTING PHYSICAL WIRING ===".postln;
    "Testing if your USB device's rear outputs are physically connected...".postln;
    "".postln;

    "Check your USB device connections:".postln;
    "- Front L/R: Should be obvious stereo outputs".postln;
    "- Center: Single speaker output".postln;
    "- LFE: Subwoofer output (low frequencies)".postln;
    "- Rear L/R: These might be labeled differently!".postln;
    "".postln;

    "Common USB 5.1 output labels:".postln;
    "- Line Out (Front L/R)".postln;
    "- Rear (Rear L/R)".postln;
    "- Center/Sub or C/LFE (Center + Subwoofer)".postln;
    "".postln;

    "Your USB device might combine Center+LFE into one output,".postln;
    "and you might need to check the 'Rear' output specifically.".postln;
    "".postln;

    "Testing with speaker-test to compare:".postln;
    "Run in terminal: speaker-test -D hw:1,0 -c 6 -t wav".postln;
    "This should play on ALL channels including rear.".postln;
    "If speaker-test rear works but SuperCollider doesn't,".postln;
    "there's a channel mapping issue.".postln;
};

~testChannelMapping = {
	var channelNames = [
        "Channel 0 (should be Front Left)",
        "Channel 1 (should be Front Right)",
        "Channel 2 (should be Center)",
        "Channel 3 (should be LFE/Subwoofer)",
        "Channel 4 (should be Rear Left)",
        "Channel 5 (should be Rear Right)"
    ];

    "=== TESTING CHANNEL MAPPING ===".postln;
    "Testing if SuperCollider channels map correctly to USB device outputs...".postln;
    "".postln;

    "Playing test on each SuperCollider channel with 2-second gaps:".postln;

    6.do({ |i|
        SystemClock.sched(i * 3, {
            ("*** " ++ channelNames[i] ++ " ***").postln;

            {
                var freq = 200 + (i * 100);  // Different frequency per channel
                var sig = SinOsc.ar(freq) * 0.7;
                var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
                var output = Array.fill(6, 0);
                output[i] = sig * env;
                Out.ar(0, output);
            }.play;
        });
    });

    SystemClock.sched(20, {
        "Channel mapping test complete.".postln;
        "Note which physical outputs produced sound for each channel.".postln;
        "If channels 4 & 5 didn't produce rear speaker output,".postln;
        "your USB device might have different channel mapping.".postln;
    });
};

~testUSBDeviceChannelOrder = {
    "=== USB DEVICE CHANNEL ORDER INVESTIGATION ===".postln;
    "Some USB devices use different channel orders than standard 5.1".postln;
    "".postln;

    "Standard 5.1 order: L, R, C, LFE, LS, RS".postln;
    "Some devices use: L, R, LS, RS, C, LFE".postln;
    "Others use: L, R, C, LFE, LS, RS (standard)".postln;
    "".postln;

    "Testing alternative channel mappings...".postln;
    "Try playing on channels 2 and 3 (might be rear on some devices):".postln;

    SystemClock.sched(1, {
        "Testing channel 2 as potential rear left...".postln;
        {
            var sig = SinOsc.ar(400) * 0.7;
            var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
            var output = Array.fill(6, 0);
            output[2] = sig * env;
            Out.ar(0, output);
        }.play;
    });

    SystemClock.sched(4, {
        "Testing channel 3 as potential rear right...".postln;
        {
            var sig = SinOsc.ar(600) * 0.7;
            var env = EnvGen.kr(Env.perc(0.1, 2), doneAction: 2);
            var output = Array.fill(6, 0);
            output[3] = sig * env;
            Out.ar(0, output);
        }.play;
    });
};

// SuperCollider is running its own JACK server with all 6 channels connected!
~confirmSixChannelConnection = {
    "=== SUPERCOLLIDER'S INTERNAL JACK SERVER ===".postln;
    "SUCCESS! SuperCollider boot log shows ALL 6 CHANNELS CONNECTED:".postln;
    "1,0:out_1 -> system:playback_1 (Front L)".postln;
    "1,0:out_2 -> system:playback_2 (Front R)".postln;
    "1,0:out_3 -> system:playback_3 (Center)".postln;
    "1,0:out_4 -> system:playback_4 (LFE)".postln;
    "1,0:out_5 -> system:playback_5 (Rear L)".postln;
    "1,0:out_6 -> system:playback_6 (Rear R)".postln;
    "".postln;

    "SuperCollider started its own JACK server using your USB device.".postln;
    "This is why external jack_lsp fails - it's a private JACK instance.".postln;
    "".postln;

    "Now testing if rear speakers actually work: ~testRearSpeakersOnly.()".postln;
};

~testRearSpeakersOnly = {
    "=== TESTING REAR SPEAKERS ONLY ===".postln;
    "All channels should be connected. Testing rear speakers specifically...".postln;
    "".postln;

    "Playing REAR LEFT (channel 4) in 2 seconds...".postln;
    SystemClock.sched(2, {
        "*** REAR LEFT SHOULD BE PLAYING NOW ***".postln;
        {
            var sig = SinOsc.ar(300) * 0.8;  // Low frequency, loud
            var env = EnvGen.kr(Env.perc(0.1, 3), doneAction: 2);
            var output = Array.fill(6, 0);
            output[4] = sig * env;  // Channel 4 = rear left
            Out.ar(0, output);
        }.play;
    });

    SystemClock.sched(6, {
        "*** REAR RIGHT SHOULD BE PLAYING NOW ***".postln;
        {
            var sig = SinOsc.ar(500) * 0.8;  // Higher frequency, loud
            var env = EnvGen.kr(Env.perc(0.1, 3), doneAction: 2);
            var output = Array.fill(6, 0);
            output[5] = sig * env;  // Channel 5 = rear right
            Out.ar(0, output);
        }.play;
    });

    SystemClock.sched(10, {
        "Rear speaker test complete.".postln;
        "Did you hear the rear speakers?".postln;
        "If not, the issue might be:".postln;
        "1. Physical wiring of your USB device".postln;
        "2. USB device channel mapping".postln;
        "3. Your speakers aren't connected to rear outputs".postln;
        "".postln;
        "Try: ~testPhysicalWiring.()".postln;
    });
};