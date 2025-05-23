// Experimance Audio System - SuperCollider Script
// This script receives OSC messages from the Experimance system and
// controls audio playback, including environmental sounds, music,
// and transition effects.

// -----------------------------------------------------
// Server configuration
// -----------------------------------------------------
s.options.numBuffers = 1024;
s.options.memSize = 8192 * 16;
s.options.numWireBufs = 64;
s.options.maxNodes = 1024 * 8;
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels = 2;

s.waitForBoot {
    // -----------------------------------------------------
    // Global variables
    // -----------------------------------------------------
    var configDir = "../config";  // Directory for config files (relative to this script)
    var audioDir = "../audio";    // Directory for audio files

    // Context tracking
    var currentBiome = "temperate_forest";  // Default biome
    var currentEra = "wilderness";          // Default era
    var activeTags = Set["temperate_forest", "wilderness"];  // Default active tags

    // Audio buffers and synths
    var environmentLayers;     // Dictionary of active environmental layers
    var musicLoops;            // Dictionary of active music loops
    var triggers;              // Dictionary of trigger samples
    var configurations;        // All configuration data

    // Mixer buses and groups
    var mainGroup;        // Main audio group
    var envGroup;         // Environmental sounds group
    var musicGroup;       // Music loops group
    var fxGroup;          // Sound effects group
    var speakingBus;      // Bus for speaking detection
    var listeningBus;     // Bus for listening detection

    // Status flags
    var isListening = false;   // Is agent listening?
    var isSpeaking = false;    // Is agent speaking?
    var isTransitioning = false;  // Is scene transitioning?

    // -----------------------------------------------------
    // Configuration loading
    // -----------------------------------------------------
    
    // Load JSON config file
    ~loadJsonConfig = { |filename|
        var path = configDir +/+ filename;
        var file = File(path, "r");
        var data;
        
        if(file.isOpen) {
            data = file.readAllString;
            file.close;
            data = data.parseYAML;  // SuperCollider's JSON parser
            data;
        } {
            "Could not open config file: %".format(path).error;
            nil;
        };
    };
    
    // Load all configurations
    ~loadAllConfigs = {
        "Loading audio configurations...".postln;
        
        // Load layers.json (environmental audio)
        configurations = Dictionary.new;
        configurations[\layers] = ~loadJsonConfig.("layers.json");
        
        // Load triggers.json (sound effects)
        configurations[\triggers] = ~loadJsonConfig.("triggers.json");
        
        // Load music_loops.json (music by era)
        var musicConfig = ~loadJsonConfig.("music_loops.json");
        if(musicConfig.notNil) {
            configurations[\music_loops] = musicConfig[\era_loops];
        };
        
        "Configurations loaded.".postln;
    };
    
    // -----------------------------------------------------
    // Audio playback functions
    // -----------------------------------------------------
    
    // Play an environmental sound layer
    ~playEnvironmentLayer = { |layerData|
        var path, buffer, synth, tags, volume;
        
        path = layerData[\path];
        tags = layerData[\tags];
        volume = layerData[\volume] ? 1.0;
        
        // Load audio file
        buffer = Buffer.read(s, audioDir +/+ path);
        
        // Create synth for looping playback
        synth = Synth(\loopPlayer, [
            \buf, buffer,
            \amp, volume,
            \fadeTime, 3.0  // 3 second crossfade
        ], envGroup);
        
        // Return info about this layer
        (\synth: synth, \buffer: buffer, \tags: tags, \volume: volume);
    };
    
    // Play a music loop
    ~playMusicLoop = { |loopData, slotIndex|
        var path, buffer, synth, prompt, volume;
        
        path = loopData[\path];
        prompt = loopData[\prompt];
        volume = loopData[\volume] ? 1.0;
        
        // Load audio file
        buffer = Buffer.read(s, audioDir +/+ path);
        
        // Create synth for looping playback
        synth = Synth(\musicPlayer, [
            \buf, buffer,
            \amp, volume,
            \fadeTime, 5.0,  // 5 second crossfade between eras
            \slotIndex, slotIndex  // Used for ducking specific slots
        ], musicGroup);
        
        // Return info about this music loop
        (\synth: synth, \buffer: buffer, \prompt: prompt, \volume: volume);
    };
    
    // Play a trigger sound (one-shot)
    ~playTrigger = { |triggerData|
        var path, buffer, synth, volume;
        
        path = triggerData[\path];
        volume = triggerData[\volume] ? 1.0;
        
        // Load audio file
        buffer = Buffer.read(s, audioDir +/+ path);
        
        // Create synth for one-shot playback
        synth = Synth(\triggerPlayer, [
            \buf, buffer,
            \amp, volume
        ], fxGroup);
        
        // Return the synth
        synth;
    };
    
    // Update audio based on current context
    ~updateAudioContext = {
        var matching, nonMatching, matchingTags;
        
        // Get layers that match current context (biome + era + active tags)
        matchingTags = activeTags.asArray;
        matching = configurations[\layers].select { |layer|
            var layerTags = layer[\tags];
            layerTags.any { |tag| matchingTags.includes(tag) };
        };
        
        // Handle environment layers - fade out non-matching, fade in matching
        environmentLayers.do { |layerInfo, path|
            if(matching.any { |m| m[\path] == path.asString }.not) {
                // Fade out and remove this layer
                layerInfo[\synth].set(\gate, 0);  // Trigger release
                { environmentLayers.removeAt(path); }.defer(5);  // Remove after fade
            };
        };
        
        // Add new matching layers
        matching.do { |layer|
            var path = layer[\path];
            if(environmentLayers[path.asSymbol].isNil) {
                // This is a new layer, add it
                environmentLayers[path.asSymbol] = ~playEnvironmentLayer.(layer);
            };
        };
        
        // Handle music loops
        var eraLoops = configurations[\music_loops][currentEra.asSymbol];
        if(eraLoops.notNil) {
            // Update music loops for this era
            eraLoops.do { |loop, i|
                var path = loop[\path].asSymbol;
                if(musicLoops[path].isNil) {
                    // New loop for this slot
                    musicLoops[path] = ~playMusicLoop.(loop, i);
                };
            };
            
            // Fade out loops that aren't in this era
            musicLoops.keys.do { |path|
                if(eraLoops.any { |l| l[\path].asSymbol == path }.not) {
                    // This loop isn't in the current era
                    musicLoops[path][\synth].set(\gate, 0);  // Trigger release
                    { musicLoops.removeAt(path); }.defer(6);  // Remove after fade
                };
            };
        };
    };
    
    // -----------------------------------------------------
    // OSC Message Handlers
    // -----------------------------------------------------
    
    // Set up OSC responders
    ~setupOSC = {
        // Main context: /spacetime <biome> <era>
        OSCdef(\spacetime, { |msg, time, addr, recvPort|
            var biome = msg[1].asString;
            var era = msg[2].asString;
            
            "Context changed: biome=%, era=%".format(biome, era).postln;
            
            // Update state
            currentBiome = biome;
            currentEra = era;
            
            // Clear and rebuild active tags
            activeTags = Set[biome, era];
            
            // Update audio context
            ~updateAudioContext.();
            
        }, '/spacetime');
        
        // Include tag: /include <tag>
        OSCdef(\include, { |msg, time, addr, recvPort|
            var tag = msg[1].asString;
            
            "Including tag: %".format(tag).postln;
            
            // Add tag to active set
            activeTags.add(tag);
            
            // Update audio context
            ~updateAudioContext.();
            
        }, '/include');
        
        // Exclude tag: /exclude <tag>
        OSCdef(\exclude, { |msg, time, addr, recvPort|
            var tag = msg[1].asString;
            
            "Excluding tag: %".format(tag).postln;
            
            // Remove tag from active set
            activeTags.remove(tag);
            
            // Update audio context
            ~updateAudioContext.();
            
        }, '/exclude');
        
        // Listening status: /listening <start|stop>
        OSCdef(\listening, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            
            isListening = (status == "start");
            "Listening: %".format(isListening).postln;
            
            // Duck environmental audio while listening
            envGroup.set(\duck, if(isListening || isSpeaking) { 0.5 } { 1.0 });
            
            // Play listening trigger if starting
            if(isListening) {
                var listeningTrigger = configurations[\triggers].detect { |t| t[\trigger] == "listening" };
                if(listeningTrigger.notNil) {
                    ~playTrigger.(listeningTrigger);
                };
            };
            
        }, '/listening');
        
        // Speaking status: /speaking <start|stop>
        OSCdef(\speaking, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            
            isSpeaking = (status == "start");
            "Speaking: %".format(isSpeaking).postln;
            
            // Duck environmental audio while speaking
            envGroup.set(\duck, if(isListening || isSpeaking) { 0.5 } { 1.0 });
            
            // Play speaking trigger if starting
            if(isSpeaking) {
                var speakingTrigger = configurations[\triggers].detect { |t| t[\trigger] == "speaking" };
                if(speakingTrigger.notNil) {
                    ~playTrigger.(speakingTrigger);
                };
            };
            
        }, '/speaking');
        
        // Transition cue: /transition <start|stop>
        OSCdef(\transition, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            
            isTransitioning = (status == "start");
            "Transition: %".format(isTransitioning).postln;
            
            // Play transition sound effect
            if(isTransitioning) {
                var transitionTrigger = configurations[\triggers].detect { |t| t[\trigger] == "transition" };
                if(transitionTrigger.notNil) {
                    ~playTrigger.(transitionTrigger);
                };
            };
            
        }, '/transition');
        
        // Reload configs: /reload
        OSCdef(\reload, { |msg, time, addr, recvPort|
            "Reloading audio configurations...".postln;
            ~loadAllConfigs.();
            "Configurations reloaded.".postln;
            
        }, '/reload');
    };
    
    // -----------------------------------------------------
    // Synthesizer definitions
    // -----------------------------------------------------
    
    // Define all synths
    ~defineSynths = {
        // Environmental sound player (looping with crossfade)
        SynthDef(\loopPlayer, { |buf, amp=1.0, fadeTime=3.0, duck=1.0, gate=1|
            var sig, env;
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), loop: 1);
            sig = sig * env * amp * duck;
            Out.ar(0, sig);
        }).add;
        
        // Music loop player (looping with crossfade, ducking by slot)
        SynthDef(\musicPlayer, { |buf, amp=1.0, fadeTime=5.0, slotIndex=0, gate=1|
            var sig, env;
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), loop: 1);
            sig = sig * env * amp;
            Out.ar(0, sig);
        }).add;
        
        // One-shot trigger player
        SynthDef(\triggerPlayer, { |buf, amp=1.0|
            var sig, env;
            env = EnvGen.kr(Env.linen(0.01, BufDur.kr(buf), 0.1), doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf));
            sig = sig * env * amp;
            Out.ar(0, sig);
        }).add;
    };
    
    // -----------------------------------------------------
    // Initialization
    // -----------------------------------------------------
    
    // Main initialization function
    ~init = {
        "Initializing Experimance Audio System...".postln;
        
        // Create groups for hierarchical audio organization
        mainGroup = Group.new;
        envGroup = Group.new(mainGroup, \addToHead);
        musicGroup = Group.new(mainGroup);
        fxGroup = Group.new(mainGroup, \addToTail);
        
        // Initialize dictionaries
        environmentLayers = Dictionary.new;
        musicLoops = Dictionary.new;
        triggers = Dictionary.new;
        
        // Load configurations
        ~loadAllConfigs.();
        
        // Define synthesizers
        ~defineSynths.();
        
        // Set up OSC responders
        ~setupOSC.();
        
        // Update audio with default context
        ~updateAudioContext.();
        
        "Experimance Audio System initialized and ready.".postln;
    };
    
    // Run initialization
    ~init.();
};

// EOF
