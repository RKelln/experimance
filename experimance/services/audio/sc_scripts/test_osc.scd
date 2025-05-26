// Experimance Audio Test OSC Script
// Simple test script to echo all received OSC messages and make sounds

// -----------------------------------------------------
// Server configuration
// -----------------------------------------------------
s.options.numBuffers = 1024;
s.options.memSize = 8192 * 16;
s.options.numWireBufs = 64;
s.options.maxNodes = 1024 * 8;
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels = 2;

// Define OSC port constants based on experimance_common/constants.py
~recvPort = 5568;

//OSCFunc.trace(true);

s.waitForBoot {
    "Experimance Audio Test OSC Script Started".postln;
    "This script will echo all received OSC messages and play sounds".postln;

    // -----------------------------------------------------
    // Define cleanup function for proper shutdown
    // -----------------------------------------------------
    ~cleanup = {
        "Cleaning up Test OSC Script resources...".postln;

        // Free all synths and groups defined in this script
        s.sendMsg("/g_freeAll", 0);  // Free all nodes in the default group

        // Free all OSC responders
        OSCFunc.all.do { |func| func.free };

        // Send all-notes-off to ensure no hanging notes
        16.do { |chan| s.sendMsg("/n_set", 1, chan, 0) };

        // Run garbage collection
        Platform.freeAllExtensions;
        Interpreter.clearAll;

        // Wait briefly to ensure everything is released
        0.1.wait;

        // Force server cleanup before exiting
        s.sendMsg("/clearSched");
        s.sendMsg("/g_freeAll", 0);
        s.sendMsg("/clearSched");

        // Free any remaining server resources
        Server.freeAll;

        "Cleanup complete - JACK resources should be properly released.".postln;
    };

    // Register cleanup function for proper shutdown on Ctrl+C
    ShutDown.add({ ~cleanup.() });

    // -----------------------------------------------------
    // Define some simple synths for testing
    // -----------------------------------------------------
    SynthDef(\ping, {
        |freq=440, amp=0.5, pan=0, dur=0.3|
        var sig, env;
        env = EnvGen.kr(Env.perc(0.01, dur, amp), doneAction: 2);
        sig = SinOsc.ar(freq) * env;
        Out.ar(0, Pan2.ar(sig, pan));
    }).add;

    SynthDef(\simpleSaw, {
        |freq=440, amp=0.2, pan=0, dur=0.5|
        var sig, env;
        env = EnvGen.kr(Env.perc(0.01, dur, amp), doneAction: 2);
        sig = Saw.ar(freq) * env;
        Out.ar(0, Pan2.ar(sig, pan));
    }).add;

    SynthDef(\noise, {
        |amp=0.2, dur=0.5|
        var sig, env;
        env = EnvGen.kr(Env.perc(0.01, dur, amp), doneAction: 2);
        sig = WhiteNoise.ar() * env;
        Out.ar(0, sig!2);
    }).add;

    // Wait for synths to be added
    s.sync;
    "Synths loaded".postln;

    // -----------------------------------------------------
    // Set up OSC responders for all expected message types
    // -----------------------------------------------------

    // Generic OSC responder that will catch all messages
    OSCFunc({ |msg, time, addr, recvPort|
        // Print the message in a clear format
        "---------------------------------------------".postln;
        "OSC Message Received:".postln;
        "Address: %".format(msg[0]).postln;
        "Arguments:".postln;
        // Loop through arguments using a different approach that avoids the problematic syntax
        for(1, msg.size - 1, { |i|
            "  [%]: %".format(i, msg[i]).postln;
        });
        "From: % on port %".format(addr, recvPort).postln;
        "Timestamp: %".format(time).postln;
        "---------------------------------------------".postln;
    }, nil, nil, ~recvPort);

    // Specific responders with sounds

    // Main context: /spacetime <biome> <era>
    OSCFunc({ |msg, time, addr, recvPort|
        var biome = msg[1].asString;
        var era = msg[2].asString;

        "Spacetime context change: biome=%, era=%".format(biome, era).postln;

        // Play a pleasing chord
        Synth(\ping, [freq: 440, amp: 0.3, pan: -0.5]);
        Synth(\ping, [freq: 550, amp: 0.3, pan: 0]);
        Synth(\ping, [freq: 660, amp: 0.3, pan: 0.5]);
    }, '/spacetime', nil, ~recvPort);

    // Include tag: /include <tag>
    OSCFunc({ |msg, time, addr, recvPort|
        var tag = msg[1].asString;

        "Including tag: %".format(tag).postln;

        // Play an ascending note
        Synth(\ping, [freq: 880, amp: 0.2, pan: 0.3, dur: 0.2]);
    }, '/include');

    // Exclude tag: /exclude <tag>
    OSCFunc({ |msg, time, addr, recvPort|
        var tag = msg[1].asString;

        "Excluding tag: %".format(tag).postln;

        // Play a descending note
            Synth(\ping, [freq: 330, amp: 0.2, pan: -0.3, dur: 0.2]);
    }, '/exclude', nil, ~recvPort);

    // Listening status: /listening <start|stop>
    OSCFunc({ |msg, time, addr, recvPort|
        var status = msg[1].asString;

        "Listening status: %".format(status).postln;

        if(status == "start") {
            // Higher frequency for start
            Synth(\simpleSaw, [freq: 1200, amp: 0.1, dur: 0.2]);
        } {
            // Lower frequency for stop
            Synth(\simpleSaw, [freq: 800, amp: 0.1, dur: 0.2]);
        };
    }, '/listening', nil, ~recvPort);

    // Speaking status: /speaking <start|stop>
    OSCFunc({ |msg, time, addr, recvPort|
        var status = msg[1].asString;

        "Speaking status: %".format(status).postln;

        if(status == "start") {
            // Higher frequency for start
            Synth(\simpleSaw, [freq: 1500, amp: 0.15, dur: 0.3]);
        } {
            // Lower frequency for stop
            Synth(\simpleSaw, [freq: 1000, amp: 0.15, dur: 0.3]);
        };
    }, '/speaking', nil, ~recvPort);

    // Transition cue: /transition <start|stop>
    OSCFunc({ |msg, time, addr, recvPort|
        var status = msg[1].asString;

        "Transition: %".format(status).postln;

        if(status == "start") {
            // Noise burst for transition start
            Synth(\noise, [amp: 0.2, dur: 0.5]);
        } {
            // Two pings for transition end
            Synth(\ping, [freq: 660, amp: 0.3, dur: 0.3]);
            {
                0.2.wait;
                Synth(\ping, [freq: 880, amp: 0.3, dur: 0.3]);
            }.fork;
        };
    }, '/transition', nil, ~recvPort);

    // Reload configs: /reload
    OSCFunc({ |msg, time, addr, recvPort|
        "Reloading configurations".postln;

        // Sequence of notes
        {
            Synth(\ping, [freq: 440, amp: 0.2, dur: 0.15]);
            0.15.wait;
            Synth(\ping, [freq: 550, amp: 0.2, dur: 0.15]);
            0.15.wait;
            Synth(\ping, [freq: 660, amp: 0.2, dur: 0.15]);
            0.15.wait;
            Synth(\ping, [freq: 880, amp: 0.2, dur: 0.15]);
        }.fork;
    }, '/reload', nil, ~recvPort);

    // Quit: /quit
    OSCFunc({ |msg, time, addr, recvPort|
        "Quit command received - shutting down".postln;

        // Play a goodbye sound
        {
            Synth(\ping, [freq: 880, amp: 0.3, dur: 0.2]);
            0.2.wait;
            Synth(\ping, [freq: 660, amp: 0.3, dur: 0.2]);
            0.2.wait;
            Synth(\ping, [freq: 440, amp: 0.3, dur: 0.5]);
            // Give time for the last sound to finish before quitting
            0.6.wait;
            "Goodbye!".postln;
            ~cleanup.();     // Run cleanup first
            0.5.wait;        // Wait briefly to ensure cleanup completes
            0.exit;          // Exit with success code instead of s.quit
        }.fork;
    }, '/quit', nil, ~recvPort);

    "---------------------------------------------".postln;
    "OSC Test Script Ready".postln;
    "Listening for OSC messages from port %".format(~netAddress).postln;
    "Send the following message types:".postln;
    "  /spacetime <biome> <era>".postln;
    "  /include <tag>".postln;
    "  /exclude <tag>".postln;
    "  /listening <start|stop>".postln;
    "  /speaking <start|stop>".postln;
    "  /transition <start|stop>".postln;
    "  /reload".postln;
    "  /quit".postln;
    "---------------------------------------------".postln;
};
