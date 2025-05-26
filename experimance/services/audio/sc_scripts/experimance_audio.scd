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

// debugging OSC:
//OSCFunc.trace(true);

s.waitForBoot {
    // -----------------------------------------------------
    // Global variables
    // -----------------------------------------------------
    var scriptDir = thisProcess.nowExecutingPath.dirname;  // Get the directory of this script
    var configDir, audioDir;

    var oscRecvPort = 5568;

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

    // Setup directories
    // Check if we have a valid script path (might be nil if running in IDE)
    if(scriptDir.notNil) {
        "Script directory: %".format(scriptDir).postln;
        configDir = scriptDir +/+ "../config";  // Config dir relative to script location
        audioDir = scriptDir +/+ "../audio";    // Audio dir relative to script location
    } {
        "Script directory not found, using default paths.".postln;
        configDir = "../config";  // Default config directory
        audioDir = "../audio";    // Default audio directory
    };

    "Config directory: %".format(configDir).postln;
    "Audio directory: %".format(audioDir).postln;

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
        var musicConfig;  // Declare all variables at the beginning

        "Loading audio configurations...".postln;

        // Load layers.json (environmental audio)
        configurations = Dictionary.new;
        configurations[\layers] = ~loadJsonConfig.("layers.json");

        // Load triggers.json (sound effects)
        configurations[\triggers] = ~loadJsonConfig.("triggers.json");

        // Load music_loops.json (music by era)
        musicConfig = ~loadJsonConfig.("music_loops.json");
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
        var fileExists = true;
        var placeholderResult;

        path = loopData[\path];
        prompt = loopData[\prompt];
        volume = loopData[\volume] ? 1.0;

        // Check if the audio file exists
        if(File.exists(audioDir +/+ path).not) {
            "Audio file not found: %. Using placeholder sound.".format(path).warn;
            fileExists = false;
        };

        if(fileExists) {
            // Load audio file if it exists
            buffer = Buffer.read(s, audioDir +/+ path);

            // Create synth for looping playback
            synth = Synth(\musicPlayer, [
                \buf, buffer,
                \amp, volume,
                \fadeTime, 5.0,  // 5 second crossfade between eras
                \slotIndex, slotIndex  // Used for ducking specific slots
            ], musicGroup);

            // Return info about this music loop with standard format
            (\synth: synth, \buffer: buffer, \prompt: prompt, \volume: volume);
        } {
            // Generate placeholder music based on era and slot
            placeholderResult = ~createPlaceholderMusic.(currentEra, slotIndex, volume);

            // Return info about this music loop with placeholder format
            (
                synth: placeholderResult,  // This contains the player and stop method
                buffer: nil,
                prompt: prompt,
                volume: volume,
                isPlaceholder: true
            );
        };
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
        var eraLoops;

        // Get layers that match current context (biome + era + active tags)
        matchingTags = activeTags.asArray;
        matching = if(configurations[\layers].notNil) {
            configurations[\layers].select { |layer|
                var layerTags = layer[\tags];
                layerTags.notNil and: { layerTags.any { |tag| matchingTags.includes(tag) } };
            };
        } {
            [];
        };

        // Handle environment layers - fade out non-matching, fade in matching
        if(environmentLayers.notNil) {
            environmentLayers.do { |layerInfo, path|
                if(matching.notNil and: { matching.any { |m| m[\path] == path.asString }.not }) {
                    // Fade out and remove this layer
                    if(layerInfo[\synth].notNil) {
                        layerInfo[\synth].set(\gate, 0);  // Trigger release
                    };
                    { environmentLayers.removeAt(path); }.defer(5);  // Remove after fade
                };
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
        eraLoops = if(configurations[\music_loops].notNil) {
            configurations[\music_loops][currentEra.asSymbol]
        } {
            nil
        };

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
                if(eraLoops.notNil and: { eraLoops.any { |l| l[\path].notNil and: { l[\path].asSymbol == path } }.not }) {
                    // This loop isn't in the current era
                    if(musicLoops[path][\isPlaceholder] == true) {
                        // Handle placeholder music with custom stop method
                        if(musicLoops[path][\synth].notNil) {
                            musicLoops[path][\synth].stop(5.0);  // 5 second fadeout
                        };
                    } {
                        // Handle regular buffer playback
                        if(musicLoops[path][\synth].notNil) {
                            musicLoops[path][\synth].set(\gate, 0);  // Trigger release
                        };
                    };
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
            var prevEra = currentEra;

            "Context changed: biome=%, era=%".format(biome, era).postln;

            // If era is changing and we have placeholder music, create a transition
            if(era != currentEra) {
                // Create crossfade for placeholder music if needed
                ~crossfadePlaceholder.(currentEra, era, 5.0);
            };

            // Update state
            currentBiome = biome;
            currentEra = era;

            // Clear and rebuild active tags
            activeTags = Set[biome, era];

            // Update audio context
            ~updateAudioContext.();

        }, '/spacetime', nil, oscRecvPort);

        // Include tag: /include <tag>
        OSCdef(\include, { |msg, time, addr, recvPort|
            var tag = msg[1].asString;

            "Including tag: %".format(tag).postln;

            // Add tag to active set
            activeTags.add(tag);

            // Update audio context
            ~updateAudioContext.();

        }, '/include', nil, oscRecvPort);

        // Exclude tag: /exclude <tag>
        OSCdef(\exclude, { |msg, time, addr, recvPort|
            var tag = msg[1].asString;

            "Excluding tag: %".format(tag).postln;

            // Remove tag from active set
            activeTags.remove(tag);

            // Update audio context
            ~updateAudioContext.();

        }, '/exclude', nil, oscRecvPort);

        // Listening status: /listening <start|stop>
        OSCdef(\listening, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            var listeningTrigger;

            isListening = (status == "start");
            "Listening: %".format(isListening).postln;

            // Duck environmental audio while listening
            envGroup.set(\duck, if(isListening || isSpeaking) { 0.5 } { 1.0 });

            // Play listening trigger if starting
            if(isListening) {
                listeningTrigger = configurations[\triggers].detect { |t| t[\trigger] == "listening" };
                if(listeningTrigger.notNil) {
                    ~playTrigger.(listeningTrigger);
                };
            };

        }, '/listening', nil, oscRecvPort);

        // Speaking status: /speaking <start|stop>
        OSCdef(\speaking, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            var speakingTrigger;

            isSpeaking = (status == "start");
            "Speaking: %".format(isSpeaking).postln;

            // Duck environmental audio while speaking
            envGroup.set(\duck, if(isListening || isSpeaking) { 0.5 } { 1.0 });

            // Play speaking trigger if starting
            if(isSpeaking) {
                speakingTrigger = configurations[\triggers].detect { |t| t[\trigger] == "speaking" };
                if(speakingTrigger.notNil) {
                    ~playTrigger.(speakingTrigger);
                };
            };

        }, '/speaking', nil, oscRecvPort);

        // Transition cue: /transition <start|stop>
        OSCdef(\transition, { |msg, time, addr, recvPort|
            var status = msg[1].asString;
            var transitionTrigger;

            isTransitioning = (status == "start");
            "Transition: %".format(isTransitioning).postln;

            // Play transition sound effect
            if(isTransitioning) {
                transitionTrigger = configurations[\triggers].detect { |t| t[\trigger] == "transition" };
                if(transitionTrigger.notNil) {
                    ~playTrigger.(transitionTrigger);
                };
            };

        }, '/transition', nil, oscRecvPort);

        // Reload configs: /reload
        OSCdef(\reload, { |msg, time, addr, recvPort|
            "Reloading audio configurations...".postln;
            ~loadAllConfigs.();
            "Configurations reloaded.".postln;

        }, '/reload', nil, oscRecvPort);

        // Quit: /quit
        OSCdef(\quit, { |msg, time, addr, recvPort|
            "Quit command received - shutting down".postln;
            // Run cleanup in a routine to allow for wait statements
            /*{
                ~cleanup.();     // Run cleanup first
                0.5.wait;        // Wait briefly to ensure cleanup completes
                0.exit;          // Exit with success code instead of s.quit
            }.fork;*/
			0.exit;
        }, '/quit', nil, oscRecvPort);
    };

    // -----------------------------------------------------
    // Synthesizer definitions and Placeholder Music System
    // -----------------------------------------------------

    // PLACEHOLDER MUSIC SYSTEM
    // -----------------------
    // This system creates procedurally generated music when audio files are missing.
    // Each era has:
    //  1. A unique musical key and scale (natural -> complex -> back to natural)
    //  2. Era-specific synthesizer sounds (pure -> electronic/mechanical -> distorted -> back to natural)
    //  3. Three layered musical elements:
    //     - Layer 0: Slow drone/bass notes (8 beat cycle)
    //     - Layer 1: Medium-speed arpeggiated chords (4 beat cycle)
    //     - Layer 2: Fast melodic patterns (2 beat cycle)
    //
    // The system gracefully handles missing audio files and creates musically
    // appropriate placeholder sounds that reflect the character of each era.

    // Musical keys for each era based on natural to electrical progression
    ~eraKeys = Dictionary[
        // Nature-based, pleasant harmonies
        \wilderness -> Dictionary[\root -> 60, \scale -> Scale.majorPentatonic, \name -> "C Pentatonic"],

        // Early civilization, consonant scales
        \pre_industrial -> Dictionary[\root -> 65, \scale -> Scale.mixolydian, \name -> "F Mixolydian"],
        \early_industrial -> Dictionary[\root -> 67, \scale -> Scale.major, \name -> "G Major"],

        // Industrial era, moving toward more tension
        \late_industrial -> Dictionary[\root -> 62, \scale -> Scale.dorian, \name -> "D Dorian"],

        // Modern to future, increasing complexity and dissonance
        \modern -> Dictionary[\root -> 69, \scale -> Scale.harmonicMinor, \name -> "A Harmonic Minor"],
        \current -> Dictionary[\root -> 64, \scale -> Scale.diminished, \name -> "E Diminished"],
        \future -> Dictionary[\root -> 71, \scale -> Scale.chromatic, \name -> "B Chromatic"],

        // Peak dissonance
        \dystopia -> Dictionary[\root -> 66, \scale -> Scale.enigmatic, \name -> "F# Enigmatic"],

        // Return to nature, peaceful
        \ruins -> Dictionary[\root -> 60, \scale -> Scale.majorPentatonic, \name -> "C Pentatonic"],
    ];

    // Create a placeholder music synth based on era and slot
    ~createPlaceholderMusic = { |era, slotIndex, volume=0.5|
        var key, timbre, pattern, synth, root, scale, tempo;
        var noteDur, patternType;

        // Get key info for this era, default to C major if not found
        key = ~eraKeys[era.asSymbol];
        if(key.isNil) { key = ~eraKeys[\wilderness]; };

        root = key[\root];
        scale = key[\scale];

        // Default tempo (beats per second)
        tempo = 1; // 60 BPM

        // Duration based on slot index (first slot is slowest)
        noteDur = switch(slotIndex,
            0, { 8 },    // Slowest - 8 beats
            1, { 4 },    // Medium - 4 beats
            2, { 2 },    // Fast - 2 beats
            { 1 }        // Default - 1 beat
        );

        // Create a unique timbre for each era with distinct sonic characteristics
        timbre = switch(era.asSymbol,
            \wilderness, { \placeholderSine },     // Pure, natural tones
            \pre_industrial, { \placeholderSine }, // Continued natural sounds
            \early_industrial, { \placeholderSaw },  // Early mechanical sounds
            \late_industrial, { \placeholderSaw },  // More developed mechanical sounds
            \modern, { \placeholderSquare },        // Early electronic sounds
            \current, { \placeholderPad },          // Contemporary digital/ambient
            \future, { \placeholderBell },          // Advanced digital/crystalline
            \dystopia, { \placeholderNoise },       // Broken, distorted sounds
            \ruins, { \placeholderFM },             // Return to nature, but complex
            { \placeholderDefault }                 // Default fallback
        );

		format("Created placeholder: root: %, scale %, tempo: %, timbre: %", root, scale, tempo, timbre).postln;

        // Create different pattern types based on slot
        patternType = switch(slotIndex,
            0, { // Root note drone
                var notePattern, amp, legato;

                // Adjust drone characteristics based on era
                if (era.asSymbol == \wilderness || era.asSymbol == \pre_industrial || era.asSymbol == \ruins) {
                    // Stable root with fifth for natural/early eras
                    notePattern = Pseq([root, root + 7], inf);
                    amp = volume * 0.7;
                    legato = 0.95;
                } {
                    if (era.asSymbol == \early_industrial || era.asSymbol == \late_industrial) {
                        // Root alternating with octave for industrial eras
                        notePattern = Pseq([root, root + 12], inf);
                        amp = volume * 0.65;
                        legato = 0.9;
                    } {
                        if (era.asSymbol == \modern || era.asSymbol == \current) {
                            // Root with some motion for modern/current
                            notePattern = Pseq([root, root + 7, root + 4, root], inf);
                            amp = volume * 0.6;
                            legato = 0.85;
                        } {
                            // Complex drone with tension for future/dystopia
                            notePattern = Pseq([root, root + 1, root, root - 1], inf);
                            amp = volume * 0.55;
                            legato = 0.75;
                        }
                    }
                };

                Pbind(
                    \instrument, timbre,
                    \note, notePattern,
                    \dur, noteDur / 2, // Split the drone into pulses
                    \amp, amp,
                    \legato, legato,
                    \slotIndex, slotIndex
                )
            },
            1, { // Arpeggiated chord
                var notes, pattern;

                // Different chord types based on era
                if (era.asSymbol == \wilderness || era.asSymbol == \ruins) {
                    // Open fifth + octave for natural sounds
                    notes = [0, 7, 12, 7];
                    pattern = Pseq(notes, inf);
                } {
                    if (era.asSymbol == \pre_industrial || era.asSymbol == \early_industrial) {
                        // Traditional triads for pre/early industrial
                        notes = [0, 4, 7, 4]; // Major triad
                        pattern = Pseq(notes, inf);
                    } {
                        if (era.asSymbol == \late_industrial || era.asSymbol == \modern) {
                            // More complex chords for modern
                            notes = [0, 4, 7, 11]; // Major 7th
                            pattern = Pseq(notes, inf);
                        } {
                            if (era.asSymbol == \current || era.asSymbol == \future) {
                                // Tension-filled chords for future era
                                notes = [0, 3, 6, 10]; // diminished 7th
                                pattern = Pseq(notes, inf);
                            } {
                                // Dissonant intervals for dystopia
                                notes = [0, 1, 6, 11];
                                pattern = Pshuf(notes, inf); // shuffle for more chaos
                            }
                        }
                    }
                };

                Pbind(
                    \instrument, timbre,
                    \degree, pattern,
                    \octave, 4,
                    \root, root,
                    \scale, scale,
                    \dur, noteDur / 4, // Four notes per cycle
                    \amp, volume * 0.5,
                    \legato, (era.asSymbol == \dystopia).if(0.4, 0.7), // shorter notes for dystopia
                    \slotIndex, slotIndex
                )
            },
            2, { // Melodic pattern
                var degrees, pattern;

                // Different melodic patterns based on era
                if (era.asSymbol == \wilderness || era.asSymbol == \pre_industrial || era.asSymbol == \ruins) {
                    // Natural, flowing melodies for natural eras
                    degrees = [0, 2, 4, 7, 9, 7, 4, 2]; // Pentatonic-based pattern
                    pattern = Pseq(degrees, inf);
                } {
                    if (era.asSymbol == \early_industrial || era.asSymbol == \late_industrial || era.asSymbol == \modern) {
                        // More structured, mechanical patterns for industrial eras
                        degrees = [0, 4, 7, 4, 0, 7, 4, 0]; // Arpeggiated pattern
                        pattern = Pseq(degrees, inf);
                    } {
                        // Complex, unpredictable patterns for future/dystopian eras
                        degrees = scale.degrees;
                        pattern = Pxrand(degrees, inf);
                    };
                };

                Pbind(
                    \instrument, timbre,
                    \degree, pattern,
                    \octave, 5,
                    \root, root,
                    \scale, scale,
                    \dur, noteDur / 8, // Eight notes per cycle
                    \amp, volume * 0.4,
                    \legato, (era.asSymbol == \wilderness).if(0.8, (era.asSymbol == \dystopia).if(0.3, 0.5)), // Smoother for natural, choppier for dystopia
                    \slotIndex, slotIndex
                )
            },
            { // Default pattern
                Pbind(
                    \instrument, timbre,
                    \midinote, Pseq([root, root+7, root+12], inf),
                    \dur, 1,
                    \amp, volume * 0.6,
                    \legato, 0.8,
                    \slotIndex, slotIndex
                )
            }
        );

        // Start the pattern with fade-in
        synth = pattern.play;
        "Started placeholder music pattern for era: % (%), slot: %".format(era, key[\name], slotIndex).postln;

        // Return the EventStreamPlayer as the "synth" with custom stop method
        (
            player: synth,
            stop: { |fadeTime=5|
                var oldAmp = synth.stream.originalStream.patternpairs.indexOf(\amp).notNil.if({
                    synth.stream.originalStream.patternpairs[synth.stream.originalStream.patternpairs.indexOf(\amp) + 1]
                }, { 0.5 });

                // Create a fade out over fadeTime seconds
                Routine({
                    var steps = fadeTime * 20; // 20 steps per second
                    var stepAmt = oldAmp / steps;

                    steps.do({ |i|
                        synth.set(\amp, oldAmp - (stepAmt * i));
                        (fadeTime / steps).wait;
                    });

                    synth.stop;
                    "Stopped placeholder music pattern for era: %, slot: %".format(era, slotIndex).postln;
                }).play;
            }
        );
    };

    // Define all synths
    ~defineSynths = {
        // Placeholder synthesizers for generated music
        SynthDef(\placeholderDefault, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur;
            dur = 0.4 * legato;
            // Generic but pleasant envelope
            env = EnvGen.kr(Env.perc(0.03, dur, curve: -2), gate, doneAction: 2);
            // Simple but rich tone
            sig = SinOsc.ar(freq) * 0.6 + Saw.ar(freq) * 0.1;
            sig = LPF.ar(sig, 2000);
            sig = sig * env * amp;
            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderSine, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur;
            dur = 0.5 * legato; // Longer, more natural tones
            // Gentle attack and release for natural sounds
            env = EnvGen.kr(Env.perc(0.05, dur, curve: -2), gate, doneAction: 2);
            // Pure sine with subtle fifth harmonic for natural character
            sig = SinOsc.ar(freq) + (SinOsc.ar(freq * 1.5) * 0.15);
            sig = sig * env * amp;
            // Add very slight chorus effect for organic quality
            sig = sig + DelayC.ar(sig, 0.02, SinOsc.kr(0.1, 0, 0.005, 0.005)) * 0.3;
            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderSaw, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur, filterEnv;
            dur = 0.3 * legato;
            // More mechanical envelope for industrial sound
            env = EnvGen.kr(Env.perc(0.01, dur, curve: 0), gate, doneAction: 2);
            // Filter envelope for movement
            filterEnv = EnvGen.kr(Env.perc(0.05, dur * 1.5, curve: -4), gate);
            // Saw wave with sub-oscillator for industrial era power
            sig = Saw.ar(freq) + (Saw.ar(freq * 0.5) * 0.4);
            // Moving filter typical of early industrial synth sounds
            sig = LPF.ar(sig, 500 + (filterEnv * 2500));
            sig = sig * env * amp * 0.4;
            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderSquare, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur, pwm;
            dur = 0.2 * legato;
            env = EnvGen.kr(Env.perc(0.01, dur), gate, doneAction: 2);
            // Pulse width modulation for "modern" character
            pwm = SinOsc.kr(3).range(0.3, 0.7);
            sig = Pulse.ar(freq, pwm) * env * amp * 0.35;
            // Resonant filter for modern 80s/90s synth character
            sig = RLPF.ar(sig, 1800 * (1 + (env * 2)), 0.6);
            // Stereo spread
            sig = Pan2.ar(sig, SinOsc.kr(0.4).range(-0.3, 0.3) + pan);
            Out.ar(0, sig);
        }).add;

        SynthDef(\placeholderPad, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur, chorus, detune;
            dur = 0.8 * legato; // Longer for pad-like sound
            // Slow attack, slow release for current era ambient pads
            env = EnvGen.kr(Env.perc(0.2, dur, curve: -3), gate, doneAction: 2);
            // Detuned oscillators for richness - current electronic music pads
            detune = 0.01;
            sig = Mix([
                SinOsc.ar(freq * (1 - detune)),
                SinOsc.ar(freq),
                SinOsc.ar(freq * (1 + detune)),
                Saw.ar(freq * 2) * 0.2 // Upper harmonic with saw
            ]);
            // Dynamic filtering
            sig = RLPF.ar(sig, 800 + SinOsc.kr(0.1).range(200, 1200), 0.5);
            // Add chorus effect for lush sound
            chorus = DelayC.ar(sig, 0.03, SinOsc.kr([0.15, 0.16]).range(0.01, 0.03)) * 0.4;
            sig = sig + chorus;
            sig = sig * env * amp * 0.25;
            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderBell, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env;
            // Define dur first before using it
            var dur = 0.6 * legato;

            // Carefully tuned harmonics for future bell-like tones
            var harmonics = [1, 2.38, 3.95, 5.1, 6.8];
            // Different envelope for each harmonic
            var envs = Array.fill(5, { |i|
                // Shorter decay for higher harmonics
                var envDur = dur * legato * (1 - (i * 0.1));
                EnvGen.kr(Env.perc(0.001, envDur, curve: -4 - i), gate);
            });

            env = EnvGen.kr(Env.perc(0.001, dur), gate, doneAction: 2);

            // Mix harmonics with individual envelopes
            sig = Mix.fill(5, { |i|
                SinOsc.ar(freq * harmonics[i]) * 0.2 * envs[i]
            });

            // Add digital artifacts for future feel
            sig = sig + HPF.ar(sig.clip(-0.5, 0.5) * 0.05, 3000);
            sig = sig * env * amp * 0.4;

            // Stereo delay for futuristic space
            sig = sig + DelayL.ar(sig, 0.05, [0.03, 0.05]) * 0.2;

            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderNoise, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur, noiseSig, ringMod;
            dur = 0.25 * legato; // Short and harsh for dystopian character
            // Sharp attack, faster decay for aggressive dystopian sound
            env = EnvGen.kr(Env.perc(0.005, dur, curve: 0), gate, doneAction: 2);

            // Base signal - dissonant, harsh
            sig = Pulse.ar(freq, SinOsc.kr(7).range(0.1, 0.9)) * 0.3;

            // Add noise component - dystopian grit
            noiseSig = HPF.ar(WhiteNoise.ar(0.7) * Pulse.ar(freq * 4, 0.1), 1500);
            sig = sig + noiseSig;

            // Ring modulation for harshness
            ringMod = sig * SinOsc.ar(freq * 7.1) * 0.2;
            sig = sig + ringMod;

            // Distort the result
            sig = sig.tanh * 1.2;

            // Heavy filter modulation
            sig = BPF.ar(sig, freq * SinOsc.kr(3).range(0.5, 2.5), 0.3);

            sig = sig * env * amp * 0.25;
            Out.ar(0, Pan2.ar(sig, pan));
        }).add;

        SynthDef(\placeholderFM, { |amp=0.5, freq=440, gate=1, pan=0, legato=0.8, slotIndex=0|
            var sig, env, dur, mod, modenv;
            dur = 0.7 * legato; // Longer for ruins era's nature return
            // Gentle envelope for ruins era - return to nature but with complexity
            env = EnvGen.kr(Env.perc(0.04, dur, curve: -2), gate, doneAction: 2);

            // Modulation envelope - starts strong then becomes gentle
            modenv = EnvGen.kr(Env.perc(0.01, dur * 1.5, curve: -3), gate);

            // FM synthesis with multiple carriers - nature with a twist
            mod = SinOsc.ar(freq * [1.5, 2.5, 0.75], 0, modenv * [100, 200, 50]);
            sig = SinOsc.ar(freq + mod);

            // Mix with a gentle sine for foundational warmth
            sig = Mix(sig) * 0.3 + SinOsc.ar(freq * 0.5) * 0.2;

            // Filter sweeping down - like returning to earth
            sig = LPF.ar(sig, 3000 * (1 - (env * 0.5)) + 500);

            sig = sig * env * amp * 0.35;

            // Gentle stereo effect
            sig = Pan2.ar(sig, SinOsc.kr(0.1).range(-0.2, 0.2) + pan);

            Out.ar(0, sig);
        }).add;

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

        // Register cleanup function for proper shutdown
        ShutDown.add({ ~cleanup.() });

        "Experimance Audio System initialized and ready.".postln;
    };

    // Cleanup function to ensure proper release of resources
    ~cleanup = {
        "Performing proper cleanup of audio resources...".postln;

        // Free all active synths and groups
        if(mainGroup.notNil) { mainGroup.free };
        if(envGroup.notNil) { envGroup.free };
        if(musicGroup.notNil) { musicGroup.free };
        if(fxGroup.notNil) { fxGroup.free };

        // Free all buffers
        if(environmentLayers.notNil) {
            environmentLayers.do { |layerInfo|
                if(layerInfo[\buffer].notNil) { layerInfo[\buffer].free };
                if(layerInfo[\synth].notNil) { layerInfo[\synth].free };
            };
        };

        if(musicLoops.notNil) {
            musicLoops.do { |loopInfo|
                if(loopInfo[\buffer].notNil) { loopInfo[\buffer].free };
                if(loopInfo[\synth].notNil) {
                    if(loopInfo[\isPlaceholder] == true) {
                        if(loopInfo[\synth][\player].notNil) { loopInfo[\synth][\player].stop };
                    } {
                        loopInfo[\synth].free;
                    };
                };
            };
        };

        // Release all buses
        if(speakingBus.notNil) { speakingBus.free };
        if(listeningBus.notNil) { listeningBus.free };

        // Release all OSC handlers
        OSCdef.all.do { |def| def.free };
/*
        // Send all-notes-off to ensure no hanging notes
        16.do { |chan| s.sendMsg("/n_set", 1, chan, 0) };

        // Release any active routines
        CmdPeriod.run;

        // Wait briefly to ensure everything is released
        0.1.wait;

        // Force server cleanup before exiting
        s.sendMsg("/clearSched");
        s.sendMsg("/g_freeAll", 0);
        s.sendMsg("/clearSched");

        // Release the server entirely
        Server.freeAll;
*/

        "Cleanup complete - JACK resources should be properly released.".postln;
    };

    // Run initialization
    ~init.();
};

// EOF
