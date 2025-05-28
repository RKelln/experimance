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

    // Volume control variables
    var masterVolume = 1.0;        // Master volume (0.0 to 1.0)
    var environmentVolume = 1.0;   // Environmental sounds volume
    var musicVolume = 1.0;         // Music loops volume
    var sfxVolume = 1.0;           // Sound effects volume

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

            // Return info about this music loop
            (
                synth: placeholderResult,
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
                    };

                    if(musicLoops.includesKey(prevEra)) {
                        // Fade out loops that aren't in this era
                        musicLoops[prevEra].do { |loopInfo|
                            if(loopInfo[\synth].notNil) {
                                "Fading out music loop %".format(prevEra).postln;
                                loopInfo[\synth].set(\gate, 0);  // Trigger release
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

            // Play transition effect if starting
            if(isTransitioning) {
                transitionTrigger = configurations[\triggers].detect { |t| t[\trigger] == "transition" };
                if(transitionTrigger.notNil && (transitionTrigger[\exists] ? false)) {
                    ~playTrigger.(transitionTrigger);
                } {
                    "No valid transition trigger audio found.".postln;
                };
            };

        }, '/transition', nil, oscRecvPort);

        // Volume control: /volume/master <value>
        OSCdef(\volume_master, { |msg, time, addr, recvPort|
            var value = msg[1].asFloat;
            
            // Ensure volume is between 0.0 and 1.0
            masterVolume = value.clip(0.0, 1.0);
            "Master volume set to: %".format(masterVolume).postln;
            
            // Update all groups with new master volume
            mainGroup.set(\masterVolume, masterVolume);
            
        }, '/volume/master', nil, oscRecvPort);
        
        // Volume control: /volume/environment <value>
        OSCdef(\volume_environment, { |msg, time, addr, recvPort|
            var value = msg[1].asFloat;
            
            // Ensure volume is between 0.0 and 1.0
            environmentVolume = value.clip(0.0, 1.0);
            "Environment volume set to: %".format(environmentVolume).postln;
            
            // Update environment group
            envGroup.set(\envVolume, environmentVolume);
            
        }, '/volume/environment', nil, oscRecvPort);
        
        // Volume control: /volume/music <value>
        OSCdef(\volume_music, { |msg, time, addr, recvPort|
            var value = msg[1].asFloat;
            
            // Ensure volume is between 0.0 and 1.0
            musicVolume = value.clip(0.0, 1.0);
            "Music volume set to: %".format(musicVolume).postln;
            
            // Update music group
            musicGroup.set(\musicVolume, musicVolume);
            
        }, '/volume/music', nil, oscRecvPort);
        
        // Volume control: /volume/sfx <value>
        OSCdef(\volume_sfx, { |msg, time, addr, recvPort|
            var value = msg[1].asFloat;
            
            // Ensure volume is between 0.0 and 1.0
            sfxVolume = value.clip(0.0, 1.0);
            "SFX volume set to: %".format(sfxVolume).postln;
            
            // Update fx group
            fxGroup.set(\sfxVolume, sfxVolume);
            
        }, '/volume/sfx', nil, oscRecvPort);

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
                
                if(placeholder.isPlaying) {
                    "Releasing test placeholder...".postln;
                    placeholder.set(\gate, 0);  // Trigger release phase
                    1.0.wait;  // Wait for release to complete
                    placeholder.free;  // Ensure synth is freed
                } {
                    "Placeholder already stopped.".postln;
                };
                
                "Finished playing test placeholder.".postln;
            }.fork;

        }, '/test_placeholder', nil, oscRecvPort);
    };

    // -----------------------------------------------------
    // Synthesizer definitions and Placeholder Music System
    // -----------------------------------------------------

    // PLACEHOLDER MUSIC SYSTEM
    // -----------------------
    // This system provides simple placeholder sounds when audio files are missing.
    // Each era and slot (layer) has a slightly different sound character.
    // The system gracefully handles missing audio files with minimal complexity.

    // Create a simple placeholder synth based on era and slot
    ~createPlaceholderMusic = { |era, slotIndex, volume=0.5|
        var synth, freq, eraIndex;

        // Ensure era is valid or use wilderness as default
        era = era ? "wilderness";

        // Ensure slotIndex is valid (0, 1, 2 for bass, mid, high)
        slotIndex = slotIndex ? 0;

        // Ensure volume is a float
        volume = volume.asFloat;

        // Log what we're creating
        "Creating simple placeholder for era: % and slot: %".format(era, slotIndex).postln;

        // Map era to a base frequency multiplier (different sound character per era)
        // This gives each era a slightly different pitch profile
        eraIndex = switch(era.asSymbol,
            \wilderness, { 0.6 },
            \pre_industrial, { 0.7 },
            \early_industrial, { 0.8 },
            \late_industrial, { 0.9 },
            \modern, { 1.0 },
            \current, { 1.1 },
            \future, { 1.2 },
            \dystopia, { 1.3 },
            \ruins, { 1.4 },
            { 1.0 }  // Default
        );

        // Base frequency varies by slot: bass (low), mid, high
        freq = switch(slotIndex,
            0, { 60 * eraIndex },    // Bass/drone (low C)
            1, { 200 * eraIndex },   // Mid-range
            2, { 320 * eraIndex },   // Higher
            { 100 * eraIndex }       // Default
        );

        // Create synth with the calculated parameters
        synth = Synth(\simplePlaceholder, [
            \amp, volume,
            \freq, freq,
            \fadeTime, 1.0,  // 1 second fade for more responsive testing
            \slotIndex, slotIndex
        ], musicGroup);

        // Return the synth instance
        synth
    };

    // Define all synths
    ~defineSynths = {

        // Simple placeholder SynthDef - provides basic sound when audio files are missing
        SynthDef(\simplePlaceholder, { |freq=220, amp=0.5, fadeTime=5.0, gate=1, slotIndex=0, masterVolume=1.0, musicVolume=1.0|
            var sig, env, mod, filterFreq;

            // Sustain envelope with proper fade in/out
            // gate=1 keeps the sound playing, gate=0 triggers release phase
            // doneAction: 2 frees the synth after release completes
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);

            // Different modulation per slot for variety
            mod = Select.kr(slotIndex, [
                // Very slow bass drone (slot 0) - slowed down for more stability
                SinOsc.kr(0.05).range(0.98, 1.02),
                // Medium rhythmic element (slot 1)
                SinOsc.kr(0.5).range(0.95, 1.05),
                // Fast high element (slot 2)
                SinOsc.kr(1.0).range(0.93, 1.07)
            ]);

            // Create a signal based on slot index
            sig = Select.ar(slotIndex, [
                // Bass drone (sine + subtle saw) - increased volume for the bass component
                SinOsc.ar(freq * mod) * 0.8 + (Saw.ar(freq * 0.5) * 0.4),
                // Mid range (triangle + pulse)
                LFTri.ar(freq * mod) * 0.6 + (Pulse.ar(freq * mod * 1.01, 0.3) * 0.2),
                // High range (FM-like sound)
                SinOsc.ar(freq * mod + (SinOsc.ar(freq * 3) * 20))
            ]);

            // Filter based on slot
            filterFreq = Select.kr(slotIndex, [500, 2000, 4000]);
            sig = LPF.ar(sig, filterFreq);

            // Add some stereo width
            sig = Pan2.ar(sig, SinOsc.kr(0.1).range(-0.5, 0.5));

            // Apply envelope, amplitude and volume controls
            sig = sig * env * amp * 0.5 * musicVolume * masterVolume; // Apply both music and master volume

            Out.ar(0, sig);
        }).add;

        // Environmental sound player (looping with crossfade)
        SynthDef(\loopPlayer, { |buf, amp=1.0, fadeTime=3.0, duck=1.0, gate=1, masterVolume=1.0, envVolume=1.0|
            var sig, env;
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), loop: 1);
            sig = sig * env * amp * duck * envVolume * masterVolume; // Apply both environment and master volume
            Out.ar(0, sig);
        }).add;

        // Music loop player (looping with crossfade, ducking by slot)
        SynthDef(\musicPlayer, { |buf, amp=1.0, fadeTime=5.0, slotIndex=0, gate=1, masterVolume=1.0, musicVolume=1.0|
            var sig, env;
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), loop: 1);
            sig = sig * env * amp * musicVolume * masterVolume; // Apply both music and master volume
            Out.ar(0, sig);
        }).add;

        // One-shot trigger player
        SynthDef(\triggerPlayer, { |buf, amp=1.0, masterVolume=1.0, sfxVolume=1.0|
            var sig, env;
            env = EnvGen.kr(Env.linen(0.01, BufDur.kr(buf), 0.1), doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), doneAction: 2);
            sig = sig * env * amp * sfxVolume * masterVolume; // Apply both SFX and master volume
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

        // Apply initial volume settings to all groups
        "Setting up initial volume controls...".postln;
        mainGroup.set(\masterVolume, masterVolume);
        envGroup.set(\envVolume, environmentVolume);
        musicGroup.set(\musicVolume, musicVolume);
        fxGroup.set(\sfxVolume, sfxVolume);
        "Volume controls initialized.".postln;

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
        var eras, testRoutine;

        // List of all eras to test
        eras = [\wilderness, \pre_industrial, \early_industrial, \late_industrial,
                 \modern, \current, \future, \dystopia, \ruins];

        // Start the test routine
        "=== STARTING PLACEHOLDER SOUND TEST ===".postln;
        "This will play all three slots (bass, mid, high) simultaneously for each era".postln;
        "Each era's sounds will play for 5 seconds, then fade out for 1 second before the next era".postln;

        // stop any existing music
        ~stopMusic.();

        testRoutine = Routine({
            eras.do { |era|
                var placeholders;
                
                "=== Testing era: % with all slots simultaneously ===".format(era).postln;
                
                // Create three placeholders (bass, mid, high) for this era at once
                placeholders = Array.fill(3, { |slot|
                    "Creating placeholder for era % - slot %".format(era, slot).postln;
                    ~createPlaceholderMusic.(era.asString, slot, 0.7);
                });
                
                // Let them play together for a moment (5 seconds)
                "Playing all slots for era % simultaneously...".format(era).postln;
                5.0.wait;
                
                // Release all placeholders at once
                "Releasing all placeholders for era %".format(era).postln;
                placeholders.do { |synth, i|
                    synth.set(\gate, 0);
                };
                
                // Wait for release to complete (based on fadeTime parameter)
                1.5.wait;
            };

            "=== PLACEHOLDER SOUND TEST COMPLETE ===".postln;
        });

        testRoutine.play;

        // Return the routine so it can be stopped if needed
        testRoutine;
    };

    ~stopMusic = {
        "Stopping all music...".postln;

        if(musicLoops.notNil) {
            musicLoops.keysValuesDo { |era, loops|
                loops.do { |loopInfo, i|
                    if(loopInfo[\buffer].notNil) { loopInfo[\buffer].free };
                    if(loopInfo[\synth].notNil) {
                        "Stopping music loop % for era %".format(i, era).postln;
                        // Free the synth node whether it's a placeholder or regular buffer player
                        loopInfo[\synth].free;
                    };
                };
            };
        };
    };

    // Cleanup function to ensure proper release of resources
    ~cleanup = {
        "Performing proper cleanup of audio resources...".postln;

        ~stopMusic.();

        // Free all buffers
        if(environmentLayers.notNil) {
            environmentLayers.do { |layerInfo|
                if(layerInfo[\buffer].notNil) { layerInfo[\buffer].free };
                if(layerInfo[\synth].notNil) { layerInfo[\synth].free };
            };
        };

        // Free all active synths and groups
        if(mainGroup.notNil) { mainGroup.free };
        if(envGroup.notNil) { envGroup.free };
        if(musicGroup.notNil) { musicGroup.free };
        if(fxGroup.notNil) { fxGroup.free };

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
//m = NetAddr("127.0.0.1", 5568); // loopback
//m.sendMsg("/spacetime", "forest", "wilderness")

