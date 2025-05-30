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
    var currentBiome = \temperate_forest;  // Default biome as symbol
    var currentEra = \wilderness;          // Default era as symbol
    var activeTags = Set[\temperate_forest];  // Default active tags as symbols

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
    var preloadedBuffers;      // Dictionary of preloaded music buffers

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

    // Helper function to convert tag arrays to symbols
    ~convertStringsToSymbols = { |stringsArray|
        if(stringsArray.notNil) {
            stringsArray.collect { |string|
                if(string.isKindOf(String)) {
                    string.replace(" ", "_").asSymbol;
                };
            };
        } {
            nil;
        };
    };

    // Helper function to:
    // 1) Convert all keys to symbols
    // 2) convert all tag strings in all \tags keys to symbols
    ~processJSON = { |jsonObj|
        var result;
        var keysToConvertToSymbols = [\tags, \requires, \requires_any, \requires_none];

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
                        var symKey = k.replace(" ", "_").asSymbol;
                        // Debug log for key conversion
                        //"Converting key: '%' to symbol: '%'".format(k, symKey).postln;

                        // If the key is \tags, convert its values to symbols
                        if (keysToConvertToSymbols.includes(symKey) and: { v.isKindOf(Array) }) {
                            // Convert tags to symbols
                            result[symKey] = ~convertStringsToSymbols.(v);
                        } {
                            // Recursively convert nested objects/arrays
                            result[symKey] = ~processJSON.(v);
                        }
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
                            ~processJSON.(item);
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
                        result = ~processJSON.(data);
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
                        //"Verified environmental audio file: %".format(path).postln;
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
                configurations[\music_loops] = musicConfig[\era_loops];
            } {
                "Warning: No era_loops found in music config".warn;
            };

            // Verify file existence for each music loop in each era
            if(configurations[\music_loops].notNil) {
                "Verifying music loop audio files...".postln;

                configurations[\music_loops].keysValuesDo { |era, loops|
                    "Processing era: % (type: %)".format(era, era.class).postln;

                    loops.do { |loop, i|
                        path = loop[\path];
                        if(path.notNil) {
                            loop[\exists] = File.exists(audioDir +/+ path);
                            if(loop[\exists].not) {
                                "WARNING: Music loop file not found: % for era: %".format(path, era).warn;
                            } {
                                //"Verified music loop file: % for era: %".format(path, era).postln;
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
    // Volume helper functions for live parameter access
    // -----------------------------------------------------
    
    // Helper functions to access current volume values from within routines
    // These are needed because SuperCollider captures variables by value in closures
    // When a Routine is created, it captures the current values of variables like
    // masterVolume and environmentVolume. If these change later, the routine still
    // uses the old captured values. These helper functions allow routines to access
    // the current live values, enabling real-time volume updates for interval-based sounds.
    ~getCurrentMasterVolume = {
        masterVolume;
    };
    
    ~getCurrentEnvironmentVolume = {
        environmentVolume;
    };
    
    ~getCurrentMusicVolume = {
        musicVolume;
    };
    
    ~getCurrentSfxVolume = {
        sfxVolume;
    };

    // -----------------------------------------------------
    // Audio playback functions
    // -----------------------------------------------------

    // Play an environmental sound layer
    ~playEnvironmentLayer = { |layerData|
        var path, buffer, synth, tags, volume, interval;
        var fileExists, routine;

        path = layerData[\path];
        tags = layerData[\tags];
        volume = (layerData[\volume] ? 1.0).asFloat;  // Ensure volume is a float
        interval = layerData[\interval] ? "loop";  // Default to loop if not specified
        fileExists = layerData[\exists] ? false;  // Use the precomputed exists field, default to false if not set

        if(fileExists) {
            // Load audio file if it exists
            buffer = Buffer.read(s, audioDir +/+ path);

            // Handle different interval types
            case { interval == "loop" } {
                // Create synth for continuous looping playback (existing behavior)
                synth = Synth(\loopPlayer, [
                    \buf, buffer,
                    \amp, volume,
                    \fadeTime, 3.0,  // 3 second crossfade
                    \masterVolume, masterVolume,  // Apply current master volume
                    \envVolume, environmentVolume  // Apply current environment volume
                ], envGroup);
                
                routine = nil;  // No routine needed for continuous loops
            } {
                // For non-loop intervals, create a routine that plays the sound periodically
                var minInterval, maxInterval, nextPlayTime;
                
                // Define timing ranges for each interval type
                case { interval == "frequent" } {
                    minInterval = 3;   // 3-8 seconds between plays
                    maxInterval = 8;
                } { interval == "occasional" } {
                    minInterval = 15;  // 15-45 seconds between plays  
                    maxInterval = 45;
                } { interval == "rare" } {
                    minInterval = 60;  // 1-3 minutes between plays
                    maxInterval = 180;
                } {
                    // Default fallback to occasional
                    minInterval = 15;
                    maxInterval = 45;
                };
                
                // Create routine for periodic playback
                routine = Routine({
                    loop {
                        // Random wait time within the interval range
                        nextPlayTime = rrand(minInterval, maxInterval);
                        nextPlayTime.wait;
                        
                        // Play the sound once (non-looping)
                        // Access current live volume values from global scope
                        Synth(\oneShot, [
                            \buf, buffer,
                            \amp, volume,
                            \masterVolume, ~getCurrentMasterVolume.(),
                            \envVolume, ~getCurrentEnvironmentVolume.()
                        ], envGroup);
                    };
                });
                
                // Start the routine
                routine.play;
                
                synth = nil;  // No persistent synth for periodic sounds
            };

            // Return info about this layer
            (\synth: synth, \routine: routine, \buffer: buffer, \tags: tags, \volume: volume, \interval: interval, \routine: routine);
        } {
            // Return empty info for non-existing file (no placeholder for ambient sounds)
            "WARNING: Skipping non-existent environmental audio file: %".format(path).warn;
            (\synth: nil, \buffer: nil, \tags: tags, \volume: volume, \interval: interval, \routine: nil);
        };
    };

    // Play a music loop
    // This function must be called within a Routine when using file loading
    ~playMusicLoop = { |loopData, slotIndex, startPos=0|
        var era = currentEra.asSymbol; // Ensure era is always a symbol for consistency
        var synth, path, volume;
        var bufferInfo, buffer;
        var result;
        
        // Check for nil parameters
        if(loopData.isNil) {
            "ERROR: loopData is nil in playMusicLoop".error;
            ^nil;
        };
        
        if(era.isNil) {
            "ERROR: current era is nil in playMusicLoop, using 'wilderness' as fallback".error;
            era = \wilderness; // Default fallback
        };
        
        // Get volume from loop data
        volume = (loopData[\volume] ? 1.0).asFloat;
        path = loopData[\path];
        
        // Add detailed debugging for buffer search
        "DEBUG: Looking for preloaded buffer for era % slot %".format(era, slotIndex).postln;
        "DEBUG: era class: %, value: %".format(era.class, era).postln;
        if(preloadedBuffers.notNil) {
            "DEBUG: Available eras in preloadedBuffers: %".format(preloadedBuffers.keys.asArray).postln;
            preloadedBuffers.keysValuesDo { |eraKey, buffers|
                "DEBUG: Era % has % buffers".format(eraKey, buffers.size).postln;
            };
        } {
            "DEBUG: preloadedBuffers is nil!".warn;
        };
        
        // Try to get preloaded buffer info with extra validation
        if(preloadedBuffers.isNil) {
            "ERROR: preloadedBuffers is nil".error;
            bufferInfo = nil;
        } {
            // Verify that the era exists in preloadedBuffers
            if(preloadedBuffers[era].isNil) {
                "ERROR: No buffers found for era % in preloadedBuffers".format(era).error;
                "Available eras: %".format(preloadedBuffers.keys.asArray).postln;
                bufferInfo = nil;
            } {
                // Verify slot index is valid
                if(slotIndex.isNil || slotIndex.isNumber.not) {
                    "ERROR: Invalid slotIndex: %".format(slotIndex).error;
                    bufferInfo = nil;
                } {
                    // Make sure we're not accessing beyond array bounds
                    if(preloadedBuffers[era].size.isNil || slotIndex >= preloadedBuffers[era].size) {
                        "ERROR: slotIndex % out of bounds for era % (size: %)".format(
                            slotIndex, era, preloadedBuffers[era].size ? "nil").error;
                        bufferInfo = nil;
                    } {
                        // All validations passed, get the buffer
                        bufferInfo = preloadedBuffers[era][slotIndex];
                        "DEBUG: Found buffer for era % slot %".format(era, slotIndex).postln;
                    };
                };
            };
        };
        
        // Fallback if no buffer found
        if(bufferInfo.isNil) {
            "No preloaded buffer found for era % slot %".format(era, slotIndex).warn;
        };
        
        if(bufferInfo.notNil && bufferInfo[\buffer].notNil) {
            // We have a preloaded buffer - use it
            buffer = bufferInfo[\buffer];
            
            // Log that we're using a preloaded buffer
            "Using preloaded buffer for era % slot %: %".format(era, slotIndex, path).postln;
            
            // Ensure startPos is within valid bounds
            if(bufferInfo[\numFrames].notNil && bufferInfo[\numFrames].isNumber) {
                startPos = min(max(startPos, 0), bufferInfo[\numFrames] - 1);
            } {
                startPos = 0;
                "Warning: Invalid buffer numFrames, using startPos = 0".warn;
            };
            
            // Don't validate buffer frames here - it may not be fully loaded yet
            // Instead, rely on the preloading process which already syncs with the server
            // The "Buffer UGen: no buffer data" warning is harmless and should resolve itself
            // when the buffer data is fully available
            
            // Create synth with the preloaded buffer
            synth = Synth(\musicPlayer, [
                \buf, buffer,
                \amp, volume,
                \fadeTime, 5.0,
                \slotIndex, slotIndex,
                \masterVolume, masterVolume,
                \musicVolume, musicVolume,
                \startPos, startPos
            ], musicGroup);
            
            result = (
                \synth: synth,
                \buffer: buffer, 
                \volume: volume,
                \startTime: Main.elapsedTime,
                \bufferDuration: bufferInfo[\duration]
            );
            
            "Started music player with duration: % seconds".format(bufferInfo[\duration]).postln;
        } {
            // No preloaded buffer - use placeholder
            var placeholderSynth = ~createPlaceholderMusic.(era, slotIndex, volume);
            
            "Using placeholder sound for era % slot %".format(era, slotIndex).warn;
            
            result = (
                \synth: placeholderSynth,
                \buffer: nil,
                \volume: volume,
                \isPlaceholder: true,
                \startTime: Main.elapsedTime,
                \bufferDuration: 8.0 // Fixed duration for placeholders
            );
        };
        
        // Return result
        result;
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
                \amp, volume,
                \masterVolume, masterVolume,  // Apply current master volume
                \sfxVolume, sfxVolume  // Apply current sfx volume
            ], fxGroup);

            // Return the synth
            synth;
        } {
            // Return nil for non-existing file (no placeholder for triggers)
            "WARNING: Skipping non-existent trigger audio file: %".format(path).warn;
            nil;
        };
    };
    
    // Determine if a layer should be played based on active tags
    // Implements tag filtering logic:
    // 1. At least one tag in layer's "tags" must be in active tags
    // 2. All "requires" tags must be in active tags (if specified)
    // 3. At least one "requires_any" tag must be in active tags (if specified)
    // 4. No "requires_none" tag can be in active tags (if specified)
    ~isLayerEligible = { |layer, validTags, debug=false|
        var tags, requiresAll, requiresAny, requiresNone;
        var debugStatus, allTagsPresent;
        var result = true; // Start assuming eligible
        
        // Get the layer's tag arrays
        tags = layer[\tags];
        requiresAll = layer[\requires];
        requiresAny = layer[\requires_any];
        requiresNone = layer[\requires_none];
        
        // Start debug status with layer info
        debugStatus = "Layer %".format(layer[\path]);
        
        // TEST 1: Basic tag match - at least one tag must be active
        if(tags.isNil or: { tags.any { |tag| validTags.includes(tag) }.not }) {
            result = false; // Not eligible
        } {
            // Update debug status for passed tag test
            debugStatus = "% - Tags match: %".format(debugStatus, tags);

            // TEST 2: If requires is specified, ALL of those tags must be present
            if(requiresAll.notNil and: { result }) {
                if(requiresAll.every { |tag| validTags.includes(tag) }.not) {
                    debugStatus = "FAIL: % | missing required_all: %".format(debugStatus, requiresAll);
                    if(debug) { debugStatus.postln; };
                    result = false; // Not eligible
                } {
                    debugStatus = "% | all required tags present: %".format(debugStatus, requiresAll);
                };
            };

            // TEST 3: If requires_any is specified, AT LEAST ONE of those tags must be present
            if(requiresAny.notNil and: { result }) {
                if(requiresAny.any { |tag| validTags.includes(tag) }.not) {
                    debugStatus = "FAIL: % | no requires_any tags present: %".format(debugStatus, requiresAny);
                    if(debug) { debugStatus.postln; };
                    result = false; // Not eligible
                } {
                    debugStatus = "% | at least one required tag present: %".format(debugStatus, requiresAny);
                };
            };
            
            // TEST 4: If requires_none is specified, NONE of those tags can be present
            if(requiresNone.notNil and: { result }) {
                if(requiresNone.every { |tag| validTags.includes(tag).not }.not) {
                    debugStatus = "FAIL: % | forbidden tag present: %".format(debugStatus, requiresNone);
                    if(debug) { debugStatus.postln; };
                    result = false; // Not eligible
                } {
                    debugStatus = "% | no forbidden tags present: %".format(debugStatus, requiresNone);
                };
            };
        };
        
        // Only print debug status for passing layers to reduce console clutter
        if(result and: {debug}) { debugStatus.postln; };
        
        // Return the result value - no caret operator
        result;
    };

    // Update audio based on current context
    ~updateAudioContext = { |biome, era, fadeTime|
        var matching, nonMatching, matchingTags;
        var eraLoops;
        var prevEra, prevBiome;

        "+++++++++++++++++++++++++++++++++++++++".postln;
        "updateAudioContext: Biome: %, Era: %".format(biome, era).postln;

        if(biome.notNil) { biome = biome.asSymbol; }; // Ensure biome is a symbol
        if(era.notNil) { era = era.asSymbol; };   // Ensure era is a symbol

        if(biome.notNil) {
            case
            { biome == currentBiome } {
                prevBiome = currentBiome;
            }
            { biome != currentBiome } {
                prevBiome = currentBiome;
                currentBiome = biome;
                "Biome changed from % to %".format(prevBiome, currentBiome).postln;
                activeTags.add(currentBiome);
                if(prevBiome.notNil) {
                    activeTags.remove(prevBiome);
                };
            };
        } {
            prevBiome = nil;  // No change
        };

        if(era.notNil) {
            case
            { era == currentEra } {
                prevEra = currentEra;
            }
            { era != currentEra } {
                prevEra = currentEra;
                currentEra = era;
                "Era changed from % to %".format(prevEra, currentEra).postln;
                activeTags.add(currentEra);
                if(prevEra.notNil) {
                    activeTags.remove(prevEra);
                };
            };
        } {
            prevEra = nil;  // No change
        };

        "Update audio context for biome: %, era: %".format(currentBiome, currentEra).postln;

        // Get layers that match current context using new eligibility logic
        matchingTags = activeTags.asArray;
        "Active tags: %".format(matchingTags).postln;
        
        matching = if(configurations[\layers].notNil) {
            "Evaluating % layers for eligibility".format(configurations[\layers].size).postln;
            
            configurations[\layers].select { |layer|
                ~isLayerEligible.(layer, activeTags);
            };
        } {
            "No layers configuration found".warn;
            [];
        };
        
        // debugging, if matching layers are found
        if (matching.size > 0) {
            "Found % matching environmental layers:".format(matching.size).postln;
            matching.do { |layer|
                "Adding: %, tags: %".format(layer[\path], layer[\tags]).postln;
            };
        } {
            "No matching environmental layers found.".warn;
        };

        // Handle environment layers - fade out non-matching, fade in matching
        if(environmentLayers.notNil) {
            environmentLayers.pairsDo { |path, layerInfo|
                // If path is not in the matching layers, fade it out
                if(matching.notNil and: { matching.any { |m| m[\path] == path.asString }.not }) {
                    "Fading out non-matching layer: %".format(path).postln;
                    // Fade out and remove this layer
                    if(layerInfo[\synth].notNil) {
                        layerInfo[\synth].set(\gate, 0);  // Trigger release
                    };
                    environmentLayers.removeAt(path);
                };
            };
        };

        // Add new matching layers
        matching.do { |layer|
            var path = layer[\path];
            if(environmentLayers[path].isNil) {
                // This is a new layer, add it
                environmentLayers[path] = ~playEnvironmentLayer.(layer);
            };
        };

        // Handle music loops, which only change on era transitions
        // If the current era is different from the previous one
        // and era != currentEra, update music loops
        "Current era: %, Previous era: %, requested era: %".format(currentEra, prevEra, era).postln;
        if (currentEra.notNil and: { currentEra != prevEra }) {
            "Updating music loops % -> %".format(prevEra, currentEra).postln;
            //configurations[\music_loops].postln;

            eraLoops = if(configurations[\music_loops].notNil) {
                var eraKey = currentEra.asSymbol; // Always use symbol for consistent lookup
                "DEBUG: Searching for era loops with key % (currentEra: %)".format(eraKey, currentEra).postln;
                "DEBUG: Available era keys in config: %".format(configurations[\music_loops].keys.asArray).postln;
                
                // Try to get era loops with symbol key first
                configurations[\music_loops][eraKey] ?? {
                    // If not found, try string key as fallback
                    "DEBUG: Symbol key not found, trying string key".format.postln;
                    configurations[\music_loops][eraKey.asString];
                };
            } {
                nil
            };

            if(eraLoops.notNil) {
                "Era loops for %: %".format(currentEra, eraLoops).postln;
                
                // First, ALWAYS stop and clean up the previous era's loops
                if(musicLoops.includesKey(prevEra)) {
                    "Fading out previous era music loops from %".format(prevEra).postln;
                    musicLoops[prevEra].do { |loopInfo|
                        if(loopInfo[\synth].notNil) {
                            "Fading out music loop %".format(prevEra).postln;
                            loopInfo[\synth].set(\gate, 0);  // Trigger release
                        };
                    };
                    // Don't remove the previous era's music loops until we've created the new ones
                    // We'll remove it after creating the new loops
                } {
                    "No previous era music loops to fade out.".postln;
                };

                // Now create the new era's loops
                if (musicLoops.includesKey(currentEra).not) {
                    "New era: %".format(currentEra).postln;

                    // Initialize array with proper size and nil values
                    musicLoops[currentEra] = Array.fill(eraLoops.size, { nil });

                    // Update music loops for this era - must be done in a Routine for proper synchronization
                    Routine({
                        eraLoops.do { |loopInfo, i|
                            // Calculate starting position based on previous era's loop if available
                            var startPos = 0;
                            var syncSuccess = false;
                            
                            // Check if we need to synchronize with previous era's loop
                            if(musicLoops.includesKey(prevEra) and: { i < musicLoops[prevEra].size }) {
                                var prevEraLoop = musicLoops[prevEra][i];
                                var curTime = Main.elapsedTime;
                                
                                // Only sync if prev loop has timing information and duration 
                                if(prevEraLoop.notNil and: { 
                                    prevEraLoop[\startTime].notNil and:
                                    prevEraLoop[\bufferDuration].notNil
                                }) {
                                    var elapsedTime, playbackPosition, bufferDuration;
                                    var bufferInfo, numFrames;
                                    
                                    // Calculate how long the sound has been playing
                                    elapsedTime = curTime - prevEraLoop[\startTime];
                                    bufferDuration = prevEraLoop[\bufferDuration];
                                    
                                    "Elapsed: %, prevStart: %, bufferDuration: %"
                                        .format(elapsedTime.round(0.01), 
                                                prevEraLoop[\startTime].round(0.01), 
                                                bufferDuration.round(0.01)).postln;
                                    
                                    // Calculate position within the loop only if duration is valid
                                    if(bufferDuration > 0) {
                                        playbackPosition = elapsedTime % bufferDuration;
                                        
                                        // Get current era's buffer info for this slot to get numFrames
                                        if(preloadedBuffers.notNil && preloadedBuffers[currentEra].notNil && 
                                           (i.isNumber && preloadedBuffers[currentEra].size.isNumber && 
                                            i < preloadedBuffers[currentEra].size)) {
                                            bufferInfo = preloadedBuffers[currentEra][i];
                                            
                                            if(bufferInfo.notNil && bufferInfo[\numFrames].notNil && 
                                               bufferInfo[\numFrames].isNumber && bufferInfo[\numFrames] > 0) {
                                                var ratio;
                                                numFrames = bufferInfo[\numFrames];
                                                
                                                // Calculate ratio and frames
                                                if(bufferDuration > 0) {
                                                    ratio = playbackPosition / bufferDuration;
                                                    if(ratio.isNaN) { 
                                                        ratio = 0; 
                                                        "Warning: NaN ratio in sync calculation".warn;
                                                    };
                                                    
                                                    startPos = (ratio * numFrames).asInteger;
                                                } {
                                                    startPos = 0;
                                                    "Warning: Invalid buffer duration, using startPos = 0".warn;
                                                };
                                                
                                                "Syncing era % slot % using preloaded buffer position: % sec (% of % frames)"
                                                    .format(currentEra, i, playbackPosition.round(0.01), 
                                                            startPos, numFrames).postln;
                                                
                                                syncSuccess = true;
                                            };
                                        } {
                                            "No valid preloaded buffer info for era % slot %"
                                                .format(currentEra, i).warn;
                                        };
                                    } {
                                        "Cannot calculate sync position: invalid buffer duration for slot %"
                                            .format(i).warn;
                                    };
                                };
                            };
                        
                        // New loop for this slot with synced position if available
                        musicLoops[currentEra][i] = ~playMusicLoop.(loopInfo, i, startPos);
                        
                            if(syncSuccess) {
                                "Created new % music loop for slot % synchronized to previous era"
                                    .format(currentEra, i).postln;
                            } {
                                "Created new % music loop for slot % (no sync)"
                                    .format(currentEra, i).postln;
                            }
                        };
                        
                        // Now that new loops are created, remove the previous era's musicLoops
                        if(musicLoops.includesKey(prevEra)) {
                            musicLoops.removeAt(prevEra);
                            "Removed previous era % music loop data after transition".format(prevEra).postln;
                        };
                    }).play; // Play the routine
                } {
                    "Era % already exists".format(currentEra).postln;
                };
            } {
                "No music loops found for era: %".format(currentEra).warn;
            };
        } {
            "Current (%)and previous (%) eras the same".format(currentEra, prevEra).warn;
        };
    };

    // -----------------------------------------------------
    // OSC Message Handlers
    // -----------------------------------------------------

    // Set up OSC responders
    ~setupOSC = {
        // Main context: /spacetime <biome> <era>
        OSCdef(\spacetime, { |msg, time, addr, recvPort|
            var biome = msg[1].asSymbol;
            var era = msg[2].asSymbol;

            "========================================================".postln;
            "Spacetime: biome=%, era=%".format(biome, era).postln;

            // Clear and rebuild active tags
            activeTags = Set[biome, era];
            "Reset active tags to: %".format(activeTags.asArray).postln;

            // Update audio context
            ~updateAudioContext.(biome, era);

        }, '/spacetime', nil, oscRecvPort);

        // Include tag: /include <tag>
        OSCdef(\include, { |msg, time, addr, recvPort|
            var tag = msg[1].asSymbol;

            "Including tag: %".format(tag).postln;

            // Add tag to active set (ensure it's a string)
            activeTags.add(tag);
            "Active tags now: %".format(activeTags.asArray).postln;

            // Update audio context
            ~updateAudioContext.();

        }, '/include', nil, oscRecvPort);

        // Exclude tag: /exclude <tag>
        OSCdef(\exclude, { |msg, time, addr, recvPort|
            var tag = msg[1].asSymbol;

            "Excluding tag: %".format(tag).postln;

            // Remove tag from active set (convert to both string and symbol to be safe)
            activeTags.remove(tag);
            "Active tags now: %".format(activeTags.asArray).postln;

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

        // Synth info: /synth_info - prints debug information about all playing music loops
        OSCdef(\synth_info, { |msg, time, addr, recvPort|
            "==== MUSIC SYNTHS INFORMATION ====".postln;
            "Current era: %".format(currentEra).postln;
            "Current time: %".format(Main.elapsedTime).postln;
            
            // Show active nodes on the server for comparison
            s.queryAllNodes;
            "Server node status requested (will print separately)".postln;
            
            if(musicLoops.notNil) {
                var currentTime = Main.elapsedTime;
                
                "Active music loops by era:".postln;
                musicLoops.keysValuesDo { |era, loops|
                    "Era: % (%):".format(era, loops.size).postln;
                    loops.do { |loopInfo, slotIndex|
                        if(loopInfo.notNil) {
                            var elapsedTime, position, bufferDuration;
                            var isPlaceholder = loopInfo[\isPlaceholder] ? false;
                            
                            "  Slot %:".format(slotIndex).postln;
                            
                            // Print synth status
                            if(loopInfo[\synth].notNil) {
                                var synth = loopInfo[\synth];
                                var nodeID = synth.nodeID;
                                
                                // Better way to check if the synth is actually playing
                                // For SuperCollider, we assume the synth is playing if it's less than 10 minutes old
                                // This is because nodeID checks can be unreliable during buffer loading
                                var elapsedSinceStart = currentTime - (loopInfo[\startTime] ? 0);
                                if(elapsedSinceStart < 600) { // Less than 10 minutes
                                    "    Synth: PLAYING (nodeID: %)".format(nodeID).postln;
                                } {
                                    // More sophisticated check for older synths
                                    if(nodeID.notNil && {s.nodeAllocator.addresses.includes(nodeID)}) {
                                        "    Synth: PLAYING (nodeID: %)".format(nodeID).postln;
                                    } {
                                        "    Synth: STOPPED (not on server)".postln;
                                    };
                                }
                            } {
                                "    Synth: NIL".postln;
                            };
                            
                            // Print timing information
                            if(loopInfo[\startTime].notNil) {
                                elapsedTime = currentTime - loopInfo[\startTime];
                                "    Started: % seconds ago".format(elapsedTime.round(0.01)).postln;
                                
                                // Calculate current position in loop
                                bufferDuration = loopInfo[\bufferDuration];
                                if(bufferDuration.notNil && bufferDuration > 0) {
                                    position = elapsedTime % bufferDuration;
                                    "    Duration: % seconds, Current Position: % seconds (% %)".format(
                                        bufferDuration.round(0.01), 
                                        position.round(0.01),
                                        (position / bufferDuration * 100).round(0.1),
                                        "%"
                                    ).postln;
                                };
                            };
                            
                            // Print buffer and placeholder info
                            if(isPlaceholder) {
                                "    Type: PLACEHOLDER".postln;
                            } {
                                if(loopInfo[\buffer].notNil) {
                                    var numFrames = loopInfo[\buffer].numFrames ? 0;
                                    "    Type: BUFFER (% frames)".format(numFrames).postln;
                                } {
                                    "    Type: INVALID (no buffer)".postln;
                                };
                            };
                            "    Volume: %".format(loopInfo[\volume] ? "unknown").postln;
                        } {
                            "  Slot %: NIL".format(slotIndex).postln;
                        };
                    };
                };
            } {
                "No active music loops".postln;
            };
            
            // Add info about preloaded buffers
            if(preloadedBuffers.notNil) {
                "".postln;
                "Preloaded buffer status:".postln;
                preloadedBuffers.keysValuesDo { |era, buffers|
                    "  Era %: % buffers".format(era, buffers.size).postln;
                    
                    buffers.do { |bufferInfo, i|
                        if(bufferInfo.notNil) {
                            var buffer = bufferInfo[\buffer];
                            var path = bufferInfo[\path];
                            if(buffer.notNil) {
                                "    Slot %: LOADED (% frames, % sec)".format(
                                    i, buffer.numFrames, bufferInfo[\duration].round(0.01)
                                ).postln;
                            } {
                                "    Slot %: NOT LOADED (%)".format(i, path ? "no path").postln;
                            };
                        } {
                            "    Slot %: NIL".format(i).postln;
                        };
                    };
                };
            };
            
            "====================================".postln;
            
        }, '/synth_info', nil, oscRecvPort);
        
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
            \slotIndex, slotIndex,
            \masterVolume, masterVolume,  // Apply current master volume
            \musicVolume, musicVolume  // Apply current music volume (placeholders are music)
        ], musicGroup);

        // Return the synth instance
        synth
    };

    // Define all synths
    ~defineSynths = {
        
        // We'll use a client-side approach instead of this SynthDef
        // by tracking loop start times and calculating positions

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
        SynthDef(\musicPlayer, { |buf, amp=1.0, fadeTime=5.0, slotIndex=0, gate=1, masterVolume=1.0, musicVolume=1.0, startPos=0|
            var sig, env, rate, bufFrames, checkedStartPos, validBuf;
            
            // Make sure we have a valid buffer - but be more lenient about buffer readiness
            // We know the buffer is correctly allocated from the preloaded system
            bufFrames = BufFrames.kr(buf);
            
            // Don't do strict validation - the Buffer UGen will handle errors internally
            // and the "Buffer UGen: no buffer data" warning is harmless during loading
            // Use a boolean value (not an integer) for validBuf
            validBuf = (bufFrames > 0); // Simple check that returns a proper boolean
            
            // Ensure startPos is valid and within buffer bounds
            checkedStartPos = min(max(startPos, 0), bufFrames.max(1) - 1); // Safer way to clamp startPos
            
            env = EnvGen.kr(Env.asr(fadeTime, 1, fadeTime), gate, doneAction: 2);
            rate = BufRateScale.kr(buf);
            
            // Use safer PlayBuf implementation without conditional that could cause boolean errors
            // This avoids the "Non Boolean in test" error
            sig = PlayBuf.ar(2, buf, rate, startPos: checkedStartPos, loop: 1);
            
            // Add silent detection to mute output during initial buffer loading
            // This helps avoid clicks and pops when buffer isn't ready
            sig = sig * (CheckBadValues.kr(sig) <= 0).lag(0.1);
            
            // Apply envelope and volume controls
            sig = sig * env * amp * musicVolume * masterVolume; // Apply both music and master volume
            Out.ar(0, sig);
        }).add;

        // One-shot environmental sound player (for periodic intervals)
        SynthDef(\oneShot, { |buf, amp=1.0, masterVolume=1.0, envVolume=1.0|
            var sig, env;
            env = EnvGen.kr(Env.linen(0.01, BufDur.kr(buf), 0.1), doneAction: 2);
            sig = PlayBuf.ar(2, buf, BufRateScale.kr(buf), doneAction: 2);
            sig = sig * env * amp * envVolume * masterVolume; // Apply both environment and master volume
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
        // Create a condition variable for synchronization
        var condition = Condition.new;
        var readyFlag = false;

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
        
        // Preload all music buffers first, and then continue with initialization
        "Starting music buffer preloading... System will wait until complete.".postln;
        
        // Use fork instead of Routine for better exception handling
        fork {
            // Preload all buffers
            ~preloadMusicBuffers.();
            "Buffer preloading complete. Continuing initialization...".postln;
            
            // Notify the condition that we're done
            condition.unhang;
        };
        
        // Wait for preloading to finish before proceeding
        condition.hang;

        if(preloadedBuffers.notNil) {
            "Checking preloaded music buffers...".postln;
            preloadedBuffers.keysValuesDo { |era, eraBuffers|
                eraBuffers.do { |bufferInfo, i|
                    if(bufferInfo.notNil && bufferInfo[\buffer].notNil) {
                        "Preloaded buffer % for era %: %".format(i, era, bufferInfo[\path]).postln;
                    } {
                        "Warning: Missing buffer % for era %".format(i, era).warn;
                    };
                };
            };
        };
        
        // Set up OSC responders (only after buffers are loaded)
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
        // Force update with current default biome and era values
        activeTags.addAll([currentBiome, currentEra]);
        ~updateAudioContext.();
        "Audio context initialization complete.".postln;

        // Register cleanup function for proper shutdown
        ShutDown.add({ ~cleanup.() });

        // Mark system as ready
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
                    // Don't free the buffer - it's now managed by the preloadedBuffers system
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

        // Release all OSC handlers
        OSCdef.all.do { |def| def.free };

        ~stopMusic.();

        // Free all environment buffers
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

        // Free all preloaded music buffers
        if(preloadedBuffers.notNil) {
            "Freeing all preloaded music buffers...".postln;
            preloadedBuffers.keysValuesDo { |era, eraBuffers|
                eraBuffers.do { |bufferInfo|
                    if(bufferInfo.notNil && bufferInfo[\buffer].notNil) {
                        bufferInfo[\buffer].free;
                    };
                };
            };
        };
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

    // Preload all music buffers on initialization to improve reliability
    ~preloadMusicBuffers = {
        var bufferCount = 0;
        
        "Preloading all music buffers...".postln;
        preloadedBuffers = Dictionary.new;
        
        if(configurations[\music_loops].notNil) {
            "DEBUG: Music loops config keys: %".format(configurations[\music_loops].keys.asArray).postln;
            
            configurations[\music_loops].keysValuesDo { |era, loops|
                var eraBuffers;
                var eraKey = era.asSymbol; // Always use symbol for consistent lookup
                
                "DEBUG: Processing era % (type: %, converted to %)".format(era, era.class, eraKey).postln;
                
                // Ensure loops is valid array or collection
                if(loops.isNil) {
                    "No loops defined for era %".format(eraKey).warn;
                } {
                    // Create buffer array
                    eraBuffers = Array.newClear(loops.size);
                    preloadedBuffers[eraKey] = eraBuffers; // Store using symbol
                    
                    loops.do { |loopData, slotIndex|
                        var path, buffer, duration, fileExists, bufferInfo;
                        
                        // Skip if loopData is nil
                        if(loopData.isNil) {
                            "Nil loop data for era % slot %".format(era, slotIndex).warn;
                        } {
                            path = loopData[\path];
                            fileExists = loopData[\exists] ? false;
                            
                            // Fix the boolean comparison by checking explicitly if fileExists is true
                            if(path.notNil && fileExists) {
                                // Load audio file into buffer
                                "Preloading buffer for era % slot %: %".format(era, slotIndex, path).postln;
                                buffer = Buffer.read(s, audioDir +/+ path);
                                
                                // Wait for buffer info to be available
                                s.sync;
                                
                                // Store buffer info
                                duration = buffer.duration;
                                bufferInfo = (
                                    \buffer: buffer,
                                    \path: path,
                                    \numFrames: buffer.numFrames,
                                    \duration: duration,
                                    \volume: (loopData[\volume] ? 1.0).asFloat
                                );
                                
                                // Store in preloaded buffers array for this era using the symbol key
                                //eraBuffers[slotIndex] = bufferInfo;
                                preloadedBuffers[eraKey][slotIndex] = bufferInfo; // Fixed: Use consistent eraKey (symbol)
                                bufferCount = bufferCount + 1;
                                
                                "Loaded buffer for % slot %: % frames, % seconds".format(
                                    era, slotIndex, buffer.numFrames, duration.round(0.01)).postln;
                            } {
                                // Store nil for unavailable buffer
                                "No valid buffer found for era % slot %, will use placeholder".format(era, slotIndex).warn;
                                //eraBuffers[slotIndex] = nil;
                                preloadedBuffers[eraKey][slotIndex] = nil; // Fixed: Use consistent eraKey (symbol)
                            };
                        };
                    };
                };
            };
        };
        
        "Preloaded % music buffers successfully.".format(bufferCount).postln;
    };

    // Run initialization
    ~init.();
};

// EOF
//m = NetAddr("127.0.0.1", 5568); // loopback
//m.sendMsg("/spacetime", "forest", "wilderness")

