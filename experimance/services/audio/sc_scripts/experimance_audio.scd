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
    var currentBiome = "temperate_forest".asSymbol;  // Default biome
    var currentEra = "wilderness".asSymbol;          // Default era
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

    // Helper function to convert JSON objects with string keys to objects with symbol keys
    ~convertJsonKeysToSymbols = { |jsonObj|
        var result;
        
        // Handle nil case gracefully
        if(jsonObj.isNil) {
            nil;
        } {
            // Handle different types appropriately
            if(jsonObj.isKindOf(Dictionary)) {
                // If it's a dictionary/object, create a new one with symbol keys
                result = Dictionary.new;
                try {
                    jsonObj.keysValuesDo { |k, v|
                        var symKey = k.asSymbol;
                        // Debug log for key conversion
                        //"Converting key: '%' to symbol: '%'".format(k, symKey).postln;
                        
                        // Recursively convert nested objects/arrays
                        result[symKey] = ~convertJsonKeysToSymbols.(v);
                    };
                } { |error|
                    "Error during key conversion: %".format(error.errorString).error;
                };
                result;
            } {
                if(jsonObj.isKindOf(SequenceableCollection) && jsonObj.isKindOf(String).not) {
                    // If it's an array/sequence (but not a string), process each element
                    try {
                        jsonObj.collect { |item|
                            ~convertJsonKeysToSymbols.(item);
                        };
                    } { |error|
                        "Error converting array elements: %".format(error.errorString).error;
                        jsonObj; // Return original on error
                    };
                } {
                    // For primitives (strings, numbers, booleans, etc.), return as-is
                    jsonObj;
                };
            };
        };
    };

    // Load JSON config file
    ~loadJsonConfig = { |filename|
        var path = configDir +/+ filename;
        var file, data, result = nil;
        
        // Check if file exists
        if(File.exists(path).not) {
            "Config file does not exist: %".format(path).error;
        } {
            // Try to open and read the file
            file = File(path, "r");
            
            if(file.isOpen) {
                // Read file contents
                data = file.readAllString;
                file.close;
                
                // Handle empty files
                if(data.size < 2) {
                    "Config file is empty: %".format(path).error;
                } {
                    // Parse the JSON with error handling
                    try {
                        data = data.parseJSON;
                        
                        // Convert JSON object with string keys to symbols
                        result = ~convertJsonKeysToSymbols.(data);
                        "Successfully loaded and converted %".format(filename).postln;
                    } { |error|
                        "Error parsing JSON in file %: %".format(path, error.errorString).error;
                    };
                };
            } {
                "Could not open config file: %".format(path).error;
            };
        };
        
        // Return the final result
        result;
    };

    // Load all configurations
    ~loadAllConfigs = {
        var musicConfig;  // Declare all variables at the beginning
        var i, j, path;

        "Loading audio configurations...".postln;

        // Load layers.json (environmental audio)
        configurations = Dictionary.new;
        configurations[\layers] = ~loadJsonConfig.("layers.json");

        // Verify file existence for each environmental layer
        if(configurations[\layers].notNil) {
            "Verifying environmental audio files...".postln;
            configurations[\layers].do { |layer, i|
                path = layer[\path];
                if(path.notNil) {
                    layer[\exists] = File.exists(audioDir +/+ path);
                    if(layer[\exists].not) {
                        "WARNING: Environmental audio file not found: %".format(path).warn;
                    } {
                        "Verified environmental audio file: %".format(path).postln;
                    };
                };
            };
        };

        // Load triggers.json (sound effects)
        configurations[\triggers] = ~loadJsonConfig.("triggers.json");

        // Verify file existence for each trigger
        if(configurations[\triggers].notNil) {
            "Verifying trigger audio files...".postln;
            configurations[\triggers].do { |trigger, i|
                path = trigger[\path];
                if(path.notNil) {
                    trigger[\exists] = File.exists(audioDir +/+ path);
                    if(trigger[\exists].not) {
                        "WARNING: Trigger audio file not found: %".format(path).warn;
                    } {
                        "Verified trigger audio file: %".format(path).postln;
                    };
                };
            };
        };

        // Load music_loops.json (music by era)
        musicConfig = ~loadJsonConfig.("music_loops.json");
        if(musicConfig.notNil) {
            "Loaded music config: %".format(musicConfig.class).postln;
            
            // Log music config keys for debugging
            "Music config keys: %".format(musicConfig.keys).postln;
            
            // Try both symbol and string access to be safe
            if(musicConfig[\era_loops].notNil) {
                "Using symbol access for era_loops".postln;
                configurations[\music_loops] = musicConfig[\era_loops];
            } {
                // Try string access as fallback
                if(musicConfig.respondsTo('at') and: { musicConfig["era_loops"].notNil }) {
                    "Using string access for era_loops - keys weren't properly converted".warn;
                    configurations[\music_loops] = musicConfig["era_loops"];
                    "Converting era_loops manually".postln;
                    configurations[\music_loops] = ~convertJsonKeysToSymbols.(configurations[\music_loops]);
                } {
                    "Warning: No era_loops found in music config".warn;
                    if(musicConfig.respondsTo('keys')) {
                        "Available keys: %".format(musicConfig.keys).postln;
                    };
                };
            };

            // Verify file existence for each music loop in each era
            if(configurations[\music_loops].notNil) {
                "Verifying music loop audio files...".postln;
                
                // Debug - show the type of keys
                "Music loops configuration type: %".format(configurations[\music_loops].class).postln;
                "Music loops keys: %".format(configurations[\music_loops].keys).postln;
                
                configurations[\music_loops].keysValuesDo { |era, loops|
                    "Processing era: % (type: %)".format(era, era.class).postln;
                    
                    loops.do { |loop, i|
                        path = loop[\path];
                        if(path.notNil) {
                            loop[\exists] = File.exists(audioDir +/+ path);
                            if(loop[\exists].not) {
                                "WARNING: Music loop file not found: % for era: %".format(path, era).warn;
                            } {
                                "Verified music loop file: % for era: %".format(path, era).postln;
                            };
                            
                            // Debug - show the keys in this loop
                            //"Loop % keys: %".format(i, loop.keys).postln;
                        } {
                            "WARNING: Loop % in era % has no path".format(i, era).warn;
                        };
                    };
                };
            };
        };

        "Configurations loaded and verified.".postln;
        "Environmental layers".postln;
        configurations[\layers].postln;
        "Triggers".postln;
        configurations[\triggers].postln;
        "Music".postln;
        configurations[\music_loops].postln;
    };

    // -----------------------------------------------------
    // Audio playback functions
    // -----------------------------------------------------

    // Play an environmental sound layer
    ~playEnvironmentLayer = { |layerData|
        var path, buffer, synth, tags, volume;
        var fileExists;

        path = layerData[\path];
        tags = layerData[\tags];
        volume = (layerData[\volume] ? 1.0).asFloat;  // Ensure volume is a float
        fileExists = layerData[\exists] ? false;  // Use the precomputed exists field, default to false if not set

        if(fileExists) {
            // Load audio file if it exists
            buffer = Buffer.read(s, audioDir +/+ path);

            // Create synth for looping playback
            synth = Synth(\loopPlayer, [
                \buf, buffer,
                \amp, volume,
                \fadeTime, 3.0  // 3 second crossfade
            ], envGroup);

            // Return info about this layer
            (\synth: synth, \buffer: buffer, \tags: tags, \volume: volume);
        } {
            // Return empty info for non-existing file (no placeholder for ambient sounds)
            "WARNING: Skipping non-existent environmental audio file: %".format(path).warn;
            (\synth: nil, \buffer: nil, \tags: tags, \volume: volume);
        };
    };

    // Play a music loop
    ~playMusicLoop = { |loopData, slotIndex|
        var path, buffer, synth, prompt, volume;
        var fileExists;
        var placeholderResult;

        // path, prompt, and volume are optional
		path = loopData[\path];
		volume = (loopData[\volume] ? 1.0).asFloat;  // Ensure volume is a float
        fileExists = loopData[\exists] ? false;  // Use the precomputed exists field, default to false if not set

        // Log the file status
        if(path.notNil && fileExists) {
            "Using existing audio file: %".format(path).postln;
        } {
            "Audio file unavailable: %. Using placeholder sound for era % slot %.".format(path, currentEra, slotIndex).warn;
        };

        if(path.notNil && fileExists) {
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
            (\synth: synth, \buffer: buffer, \volume: volume);
        } {
            // Generate placeholder music based on era and slot
            placeholderResult = ~createPlaceholderMusic.(currentEra, slotIndex, volume);

            // Return info about this music loop with placeholder format
            // placeholderResult already contains the player and properly defined stop method
            // We need to combine this with our standard info structure
            (
                synth: placeholderResult,  // Contains the player and stop method
                buffer: nil,
                volume: volume,
                isPlaceholder: true
            );
        };
    };

    // Play a trigger sound (one-shot)
    ~playTrigger = { |triggerData|
        var path, buffer, synth, volume;
        var fileExists;

        path = triggerData[\path];
        volume = (triggerData[\volume] ? 1.0).asFloat;  // Ensure volume is a float
        fileExists = triggerData[\exists] ? false;  // Use the precomputed exists field, default to false if not set

        if(fileExists) {
            // Load audio file if it exists
            buffer = Buffer.read(s, audioDir +/+ path);

            // Create synth for one-shot playback
            synth = Synth(\triggerPlayer, [
                \buf, buffer,
                \amp, volume
            ], fxGroup);

            // Return the synth
            synth;
        } {
            // Return nil for non-existing file (no placeholder for triggers)
            "WARNING: Skipping non-existent trigger audio file: %".format(path).warn;
            nil;
        };
    };

    // Update audio based on current context
    ~updateAudioContext = { |biome, era, fadeTime|
        var matching, nonMatching, matchingTags;
        var eraLoops;
        var prevEra, prevBiome;

        if (biome.notNil and: {biome != currentBiome}) {
            prevBiome = currentBiome;
            currentBiome = biome.asSymbol;  // Convert to symbol for consistency
            "Biome changed from % to %".format(prevBiome, currentBiome).postln;
            activeTags.add(currentBiome);
            if(prevBiome.notNil) {
                activeTags.remove(prevBiome);
            }
        } {
            prevBiome = nil;  // No change
        };

        if (era.notNil and: {era != currentEra}) {
            prevEra = currentEra;
            currentEra = era.asSymbol;  // Convert to symbol for consistency
            "Era changed from % to %".format(prevEra, currentEra).postln;
            activeTags.add(currentEra);
            if(prevEra.notNil) {
                activeTags.remove(prevEra);
            }
        } {
            prevEra = nil;  // No change
        };

        "Update audio context for biome: %, era: %".format(currentBiome, currentEra).postln;

        // Get layers that match current context (biome + era + active tags)
        matchingTags = activeTags.asArray;
        matching = if(configurations[\layers].notNil) {
            configurations[\layers].select { |layer|
                var layerTags = layer[\tags];
                var exists = layer[\exists] ? false; // Check if file exists
                exists and: { layerTags.notNil and: { layerTags.any { |tag| matchingTags.includes(tag) } } };
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

        // Handle music loops, which only change on era transitions
        if (currentEra.notNil) {
            "Updating music loops % -> %".format(prevEra, currentEra).postln;
            configurations[\music_loops].postln;

            eraLoops = if(configurations[\music_loops].notNil) {
                configurations[\music_loops][currentEra.asSymbol];
            } {
                nil
            };

            if(eraLoops.notNil) {
                "Era loops for %: %".format(currentEra, eraLoops).postln;

                musicLoops.postln;

                if (musicLoops.includesKey(currentEra).not) {
                    "New era: %".format(currentEra).postln;
                
                    // Initialize array with proper size and nil values
                    musicLoops[currentEra] = Array.fill(eraLoops.size, { nil });
                    
                    // Update music loops for this era
                    eraLoops.do { |loopInfo, i|
                        // New loop for this slot (note: playMusicLoop handles exists internally)
                        musicLoops[currentEra][i] = ~playMusicLoop.(loopInfo, i);
                        "Creating new % music loop for slot %".format(currentEra, i).postln;
                    
                        // TODO: if not a placeholder then set the timing to match the previous era
                        // i.e. new music loops should start at the same time as the current time previous era's loops

                        // TODO: fade in new music loop
                        fadeTime = fadeTime ? loopInfo[\fadeTime] ? 5.0;
                    };

                    if(musicLoops.includesKey(prevEra)) {
                        // Fade out loops that aren't in this era
                        musicLoops[prevEra].do { |loopInfo|
                            fadeTime = fadeTime ? loopInfo[\fadeTime] ? 5.0;  // Default fade time if not provided
                            if(loopInfo[\synth].notNil) {
                                if(loopInfo[\synth].respondsTo(\stop)) {
                                    // Make sure the fadeTime parameter is passed correctly
                                    "Fading out placeholder % with % second fade".format(i, fadeTime).postln;
                                    try {
                                        // Using the custom stop method 
                                        loopInfo[\synth].stop(fadeTime);
                                    } { |error|
                                        "Error fading out: %".format(error.errorString).error;
                                    };
                                } {
                                    // Handle regular buffer playback
                                    "Fading out music loop %".format(era).postln;
                                    loopInfo[\synth].set(\gate, 0);  // Trigger release
                                };
                            };
                        };
                        // remove the previous era's music loops
                        musicLoops.removeAt(prevEra);
                    } {
                        "No previous era music loops to fade out.".postln;
                    };
                } {
                    "Era % already exists".format(currentEra).postln;
                };
            } {
                "No music loops found for era: %".format(currentEra).warn;
            };
        } {
            "No current era.".warn;
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

            // Clear and rebuild active tags
            activeTags = Set[biome, era];

            // Update audio context
            ~updateAudioContext.(biome, era);

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
                if(listeningTrigger.notNil && (listeningTrigger[\exists] ? false)) {
                    ~playTrigger.(listeningTrigger);
                } {
                    "No valid listening trigger audio found.".postln;
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
                if(speakingTrigger.notNil && (speakingTrigger[\exists] ? false)) {
                    ~playTrigger.(speakingTrigger);
                } {
                    "No valid speaking trigger audio found.".postln;
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
                if(transitionTrigger.notNil && (transitionTrigger[\exists] ? false)) {
                    ~playTrigger.(transitionTrigger);
                } {
                    "No valid transition trigger audio found.".postln;
                };
            };

        }, '/transition', nil, oscRecvPort);

        // Reload configs: /reload
        OSCdef(\reload, { |msg, time, addr, recvPort|
            "Reloading audio configurations...".postln;
            ~loadAllConfigs.();
            "Configurations reloaded.".postln;

        }, '/reload', nil, oscRecvPort);

        // Test path command: /test_path <path>
        OSCdef(\test_path, { |msg, time, addr, recvPort|
            var path = msg[1].asString;
            
            "TEST: Checking path: %".format(path).postln;
            
            // Test accessing configuration elements with the path
            "TEST: Searching for path in configurations...".postln;
            
            if(configurations[\layers].notNil) {
                var found = configurations[\layers].detect { |layer| layer[\path] == path };
                if(found.notNil) {
                    "TEST: Found matching layer - Keys: %".format(found.keys).postln;
                    "TEST: Path: %, Tags: %, Volume: %".format(found[\path], found[\tags], found[\volume]).postln;
                };
            };
            
            if(configurations[\triggers].notNil) {
                var found = configurations[\triggers].detect { |trigger| trigger[\path] == path };
                if(found.notNil) {
                    "TEST: Found matching trigger - Keys: %".format(found.keys).postln;
                    "TEST: Path: %, Trigger: %, Volume: %".format(found[\path], found[\trigger], found[\volume]).postln;
                };
            };
            
            if(configurations[\music_loops].notNil) {
                configurations[\music_loops].keysValuesDo { |era, loops|
                    var found = loops.detect { |loop| loop[\path] == path };
                    if(found.notNil) {
                        "TEST: Found matching music loop in era % - Keys: %".format(era, found.keys).postln;
                        "TEST: Path: %, Prompt: %, Volume: %".format(found[\path], found[\prompt], found[\volume]).postln;
                    };
                };
            };
            
            "TEST: File exists? %".format(File.exists(audioDir +/+ path)).postln;
            
        }, '/test_path', nil, oscRecvPort);

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
        
        // Test placeholder sounds: /test_placeholders
        OSCdef(\test_placeholders, { |msg, time, addr, recvPort|
            "Starting placeholder sound test...".postln;
            ~testAllPlaceholderSounds.();
        }, '/test_placeholders', nil, oscRecvPort);
        
        // Test specific placeholder: /test_placeholder <era> <slot> <duration>
        OSCdef(\test_placeholder, { |msg, time, addr, recvPort|
            var era = msg[1].asString;
            var slot = msg[2].asInteger;
            var duration = msg[3].asFloat ? 5.0;
            var placeholder;
            
            "Testing specific placeholder - Era: %, Slot: %, Duration: %s".format(era, slot, duration).postln;
            
            placeholder = ~createPlaceholderMusic.(era, slot, 0.7);
            
            // Stop after the specified duration
            {
                duration.wait;
                placeholder.stop(2.0);
                "Finished playing test placeholder.".postln;
            }.fork;
            
        }, '/test_placeholder', nil, oscRecvPort);
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
        var result;

        // Ensure era is valid or use wilderness as default
        era = era ? "wilderness";

        // Ensure slotIndex is valid
        slotIndex = slotIndex ? 0;
        
        // Ensure volume is a number (in case it's passed as a string or other type)
        volume = volume.asFloat;

        // Log what we're creating
        "Creating placeholder music for era: % and slot: %".format(era, slotIndex).postln;

        // Get key info for this era, default to C major if not found
        key = ~eraKeys[era.asSymbol];
        if(key.isNil) {
            "Era % not found in key definitions, using wilderness default".format(era).warn;
            key = ~eraKeys[\wilderness];
        };

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

        // Error handling - make sure we have valid parameters before creating pattern
        if(root.isNil || scale.isNil || timbre.isNil) {
            "ERROR: Invalid parameters for placeholder music. Using defaults.".error;
            root = 60;  // Middle C
            scale = Scale.major;
            timbre = \placeholderDefault;
        };

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

        // Initialize result structure
        result = (
            player: nil,
            stop: { |fadeTime=5| 
                "No pattern to stop - placeholder already stopped or never started.".postln;
            }
        );
        
        // Safely start the pattern with error handling
        try {
            synth = pattern.play;
            "Started placeholder music pattern for era: % (%), slot: %".format(era, key[\name], slotIndex).postln;
            
            // Define full result with live synth and proper stop method
            result = (
                player: synth,
                stop: { |fadeTime=5|
                    // Only proceed if we have a valid synth
                    if(synth.notNil) {
                        var oldAmp;
                        
                        "Stopping placeholder music for era: % slot: % with % second fade".format(era, slotIndex, fadeTime).postln;
                        
                        try {
                            // Get current amplitude for smooth fade-out
                            oldAmp = synth.stream.originalStream.patternpairs.indexOf(\amp).notNil.if({
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
                        } { |error|
                            "Error during placeholder fadeout: %".format(error.errorString).error;
                            // Force stop on error
                            synth.stop;
                        };
                    } {
                        "No synth to stop for placeholder music".warn;
                    };
                },
                isPlaceholder: true
            );
        } { |error|
            "Error starting placeholder pattern: %".format(error.errorString).error;
            // We already defined a minimal result above that won't crash when stop is called
        };

        // Final log message if pattern was successfully started
        if(result.notNil and: { result[\player].notNil }) {
            "Created placeholder music result for era: % slot: %".format(era, slotIndex).postln;
        };
        
        // Return the result (either the full version with working synth or the minimal version if there was an error)
        result;
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
        "Initializing audio context with default values...".postln;
        ~updateAudioContext.();
        "Audio context initialization complete.".postln;

        // Register cleanup function for proper shutdown
        ShutDown.add({ ~cleanup.() });

        "Experimance Audio System initialized and ready.".postln;
    };

    // Test function to play all placeholder sounds in sequence
    ~testAllPlaceholderSounds = {
        var eras, placeholders, testRoutine;
        
        // List of all eras to test
        eras = [\wilderness, \pre_industrial, \early_industrial, \late_industrial, 
                 \modern, \current, \future, \dystopia, \ruins];
        
        placeholders = List.new;
        
        // Start the test routine
        "=== STARTING PLACEHOLDER SOUND TEST ===".postln;
        "This will play each placeholder sound for each era and slot for a few seconds".postln;
        
        testRoutine = Routine({
            eras.do { |era|
                // For each era, test all three slot types (drone, chord, melody)
                3.do { |slot|
                    "Testing era: % - slot: %".postln;
                    "  Creating placeholder...".format(era, slot).postln;
                    
                    // Create placeholder for this era and slot
                    placeholders.add(~createPlaceholderMusic.(era.asString, slot, 0.7));
                    
                    // Let it play for a few seconds
                    3.wait;
                    
                    // Stop current placeholder before moving to next
                    "  Stopping placeholder for era: % - slot: %".format(era, slot).postln;
                    placeholders.last.stop(1.0);
                    
                    // Wait for fade out
                    2.wait;
                };
            };
            
            "=== PLACEHOLDER SOUND TEST COMPLETE ===".postln;
        });
        
        testRoutine.play;
        
        // Return the routine so it can be stopped if needed
        testRoutine;
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
            musicLoops.keysValuesDo { |era, loops|
                loops.do { |loopInfo, i|
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
m = NetAddr("127.0.0.1", 5568); // loopback
m.sendMsg("/spacetime", "forest", "wilderness")