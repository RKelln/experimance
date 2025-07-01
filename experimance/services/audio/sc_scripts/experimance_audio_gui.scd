// Experimance Audio GUI Controller
// Interactive interface for testing OSC messages with experimance_audio.scd

// Load utilities first
var scriptDir = thisProcess.nowExecutingPath.dirname;
var configDir, audioDir, projectRoot, generalConfigDir;

// GUI variables
var window, oscAddr, configs;

var biomeMenu, eraMenu, tagsList, activeTagsList, triggerMenu;
var masterVolumeSlider, envVolumeSlider, musicVolumeSlider, sfxVolumeSlider;
var statusText, connectionButton, reloadButton;
var spacetimeButton, listeningButton, speakingButton, transitionButton;
var includeTagButton, excludeTagButton, playTriggerButton;
var currentBiome = "temperate_forest", currentEra = "wilderness";
var activeTags = Set.new;
var isConnected = false, isListening = false, isSpeaking = false, isTransitioning = false;

// Initialize direct paths
projectRoot = scriptDir +/+ "../../..";
configDir = scriptDir +/+ "../config";
audioDir = scriptDir +/+ "../audio";
generalConfigDir = projectRoot +/+ "data";

"Config directory: %".format(configDir).postln;
"Audio directory: %".format(audioDir).postln;
"General config directory: %".format(generalConfigDir).postln;


// Helper function to convert tag arrays to symbols
~convertStringsToSymbols = { |stringsArray|
    if(stringsArray.notNil) {
        stringsArray.collect { |string|
            if(string.isKindOf(String)) {
                string.replace(" ", "_").asSymbol; // Replace spaces with underscores and convert to symbol
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
                    if (symKey == \tags and: { v.isKindOf(Array) }) {
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

// Load general project configuration from data directory
~loadGeneralConfig = { |filename="experimance_config.json"|
    var path, altPath, file, data, result = nil;

    // Try the specified data directory first
    path = generalConfigDir +/+ filename;
    "Loading general config from: %".format(path).postln;

    // If file doesn't exist in the general config dir, try project root
    if(File.exists(path).not) {
        "General config file not found at %".format(path).warn;
        
        // Try alternate location - direct from project root
        altPath = configDir +/+ filename;
        "Trying alternate location: %".format(altPath).postln;
        
        if(File.exists(altPath).not) {
            "CRITICAL ERROR: General config file not found at either location:".error;
            "1. %".format(path).error;
            "2. %".format(altPath).error;
            "Please ensure the experimance_config.json file exists in one of these locations.".error;
            ^nil;
        } {
            path = altPath;
            "Found config file at alternate location: %".format(path).postln;
        };
    } {
        "Found config file at: %".format(path).postln;
    };

    // Open and read the file
    file = File(path, "r");

    if(file.isOpen) {
        "Successfully opened config file. Reading content...".postln;
        data = file.readAllString;
        file.close;

        if(data.size < 2) {
            "ERROR: General config file is empty: %".format(path).error;
            ^nil;
        } {
            "Config file contains % bytes of data".format(data.size).postln;
            try {
                "Parsing JSON data...".postln;
                result = data.parseJSON;
                result = ~processJSON.(result);

                "Successfully parsed general config".postln;
                result;
            } { |error|
                "ERROR: Failed to parse JSON from %: %".format(path, error.errorString).error;
                nil;
            };
        };
    } {
        "ERROR: Could not open general config file: %".format(path).error;
        nil;
    };
};


// Load all configurations and extract unique values
~loadAllConfigs = {
    var configsDict = Dictionary.new;  // Renamed from configs to configsDict to avoid name conflict
    var generalConfig;
    var triggersData, tagsData;
    var allTags = Set.new;
    var allBiomes = Set.new;
    var allEras = Set.new;
    var allTriggers = Set.new;

    "Loading all configurations...".postln;

    // Load general project configuration
    generalConfig = ~loadGeneralConfig.();
    if(generalConfig.isNil) {
        "CRITICAL ERROR: Failed to load general configuration. No biomes, eras, or tags will be available.".error;
        "Hint: Make sure experimance_config.json exists in the data directory.".error;
        "Searched paths: ".error;
        (generalConfigDir +/+ "experimance_config.json").error;
        (projectRoot +/+ "data" +/+ "experimance_config.json").error;
    } {
        // Store in configs dictionary
        configsDict.put(\generalConfig, generalConfig);

        "Successfully loaded general config".postln;
        generalConfig.keysValuesDo { |k, v|
            "% %: % %".format(k, k.class.name, v, v.class.name).postln;
        };

        // Extract biomes
        if(generalConfig[\biomes].notNil) {
            allBiomes = ~convertStringsToSymbols.(generalConfig[\biomes]).as(Set);
            "Loaded % biomes".format(allBiomes.size).postln;
        } {
            "Warning: No biomes found in general config".warn;
        };

        // Extract eras
        if(generalConfig[\eras].notNil) {
            allEras = ~convertStringsToSymbols.(generalConfig[\eras]).as(Set);
            "Loaded % eras".format(allEras.size).postln;
        } {
            "Warning: No eras found in general config".warn;
        };

        // Extract common tags
        if(generalConfig[\commonTags].notNil) {
            allTags = ~convertStringsToSymbols.(generalConfig[\commonTags]).as(Set);
            "Loaded % common tags".format(allTags.size).postln;
        } {
            "Warning: No common tags found in general config".warn;
        };
    };

    // load tags from layers.json
    tagsData = ~loadJsonConfig.("layers.json");

    // Verify file existence for each environmental layer
    if(tagsData.notNil) {
        tagsData.do { |layer, i|
            "layer %: %".format(i, layer).postln;
            if(layer[\path].notNil) {
                layer[\tags].do { |tag|
                    allTags.add(tag);
                };
            };
        };
    };

    // load trigger definitions from config/triggers.json
    // Load triggers.json (sound effects)
    triggersData = ~loadJsonConfig.("triggers.json");

    // Verify file existence for each trigger
    if(triggersData.notNil) {
        "Verifying trigger audio files...".postln;
        triggersData.do { |trigger, i|
            allTriggers.add(trigger[\trigger].asSymbol);
        };
    };

    // Store extracted data in the configs dictionary
    configsDict.put(\allTags, allTags.asArray.sort);
    configsDict.put(\allBiomes, allBiomes.asArray.sort);
    configsDict.put(\allEras, allEras.asArray.sort);
    configsDict.put(\allTriggers, allTriggers.asArray.sort);
    
    // Debug output
    "ConfigsDict contains:".postln;
    configsDict.keysValuesDo { |k, v|
        "Key: %, Value type: %, Size: %".format(k, v.class, if(v.isKindOf(Collection)) {v.size} {1}).postln;
    };

    configsDict;  // Return the dictionary
};

// Initialize OSC connection
~initOSC = {
    oscAddr = NetAddr("127.0.0.1", 5570); // Default experimance_audio.scd port
    isConnected = true;
    "OSC connection initialized to localhost:5568".postln;
};

// Update active tags list view and mark active tags in the available tags list
~updateActiveTagsList = {
    if(activeTagsList.notNil) {
        var sortedTags = activeTags.asArray.sort;
        var decoratedTags = sortedTags.collect { |tag|
            if(tag == currentBiome) {
                "[BIOME] " ++ tag;
            } {
                if(tag == currentEra) {
                    "[ERA] " ++ tag;
                } {
                    tag;
                };
            };
        };
        activeTagsList.items = decoratedTags;
        
        // Update the available tags list to mark which ones are active
        if(tagsList.notNil and: { configs[\allTags].notNil }) {
            var allTags = configs[\allTags].collect { |tag|
                if(activeTags.includes(tag)) {
                    "✓ " ++ tag;
                } {
                    "   " ++ tag;
                };
            };
            tagsList.items = allTags;
        };
        
        "Active tags updated: %".format(sortedTags).postln;
    };
};

// Update connection status
~updateConnectionStatus = {
    if(connectionButton.notNil) {
        connectionButton.string = if(isConnected) { "Connected" } { "Disconnected" };
        connectionButton.background = if(isConnected) { Color.green(0.8) } { Color.red(0.8) };
    };

    if(statusText.notNil) {
        var statusStr = "Status: ";
        statusStr = statusStr ++ if(isConnected) { "Connected" } { "Disconnected" };
        statusStr = statusStr ++ " | Biome: " ++ currentBiome ++ " | Era: " ++ currentEra;
        statusStr = statusStr ++ " | Tags: " ++ activeTags.size;
        if(isListening) { statusStr = statusStr ++ " | LISTENING" };
        if(isSpeaking) { statusStr = statusStr ++ " | SPEAKING" };
        if(isTransitioning) { statusStr = statusStr ++ " | TRANSITIONING" };
        statusText.string = statusStr;
    };
};

// Send OSC message
~sendOSC = { |oscPath, args|
    if(isConnected and: { oscAddr.notNil }) {
        try {
            oscAddr.sendMsg(oscPath, *args);
        } { |error|
            "Error sending OSC message: %".format(error.errorString).error;
        };
    } {
        "Not connected to audio system".warn;
    };
};

// Update spacetime context
~updateSpacetime = {
    activeTags.clear;
    activeTags.add(currentBiome);
    activeTags.add(currentEra);
    "Sending spacetime: Biome: %, Era: %".format(currentBiome, currentEra).postln;
    ~sendOSC.('/spacetime', [currentBiome.asString, currentEra.asString]);
    ~updateConnectionStatus.();
    ~updateActiveTagsList.();
};

// Create the GUI
~createGUI = {
    var margin = 10, buttonHeight = 30, sliderHeight = 25;
    var windowWidth = 800, windowHeight = 850; // Increased height from 700 to 900
	var volumeY = margin + 535;
    var sliderWidth = 200;
	var volumePresetY = volumeY + 40;

    // Create volume display labels that we can update
    var masterVolumeLabel, envVolumeLabel, musicVolumeLabel, sfxVolumeLabel;

	// Buttons
	var testButtonWidth = 120;
    var testX = margin + 400;
    var testY = margin + 395;

    window = Window("Experimance Audio Controller", Rect(100, 100, windowWidth, windowHeight));
    window.background = Color.gray(0.95);
    window.front;

    // Connection section
    StaticText(window, Rect(margin, margin, windowWidth - (2*margin), 20))
        .string_("CONNECTION")
        .font_(Font.default.size_(14).bold_(true));

    connectionButton = Button(window, Rect(margin, margin + 25, 120, buttonHeight))
        .states_([["Connect"], ["Disconnect"]])
        .action_({ |but|
            if(but.value == 1) {
                ~initOSC.();
            } {
                isConnected = false;
                oscAddr = nil;
            };
            ~updateConnectionStatus.();
        });

    reloadButton = Button(window, Rect(margin + 130, margin + 25, 100, buttonHeight))
        .string_("Reload Configs")
        .action_({
            var result = ~loadAllConfigs.();
            if(result.notNil) {
                configs = result;  // Properly update the global configs
                "Configs updated: %".format(configs.class).postln;
                ~updateMenus.();
                ~sendOSC.('/reload', []);
            } {
                "Failed to load configs".error;
            };
        });

    // Status text
    statusText = StaticText(window, Rect(margin, margin + 60, windowWidth - (2*margin), 20))
        .string_("Status: Disconnected")
        .font_(Font.monospace.size_(10))
        .background_(Color.white);

    // Spacetime section
    StaticText(window, Rect(margin, margin + 90, windowWidth - (2*margin), 20))
        .string_("SPACETIME CONTEXT")
        .font_(Font.default.size_(14).bold_(true));

    StaticText(window, Rect(margin, margin + 115, 80, 20)).string_("Biome:");
    biomeMenu = PopUpMenu(window, Rect(margin + 85, margin + 115, 150, 25))
        .action_({ |menu|
            currentBiome = configs[\allBiomes][menu.value];
            "Selected biome: %".format(currentBiome).postln;
        });

    StaticText(window, Rect(margin + 250, margin + 115, 80, 20)).string_("Era:");
    eraMenu = PopUpMenu(window, Rect(margin + 335, margin + 115, 150, 25))
        .action_({ |menu|
            currentEra = configs[\allEras][menu.value];
            "Selected era: %".format(currentEra).postln;
        });

    spacetimeButton = Button(window, Rect(margin + 500, margin + 115, 120, buttonHeight))
        .string_("Update Spacetime")
        .background_(Color.blue(0.8))
        .action_({ ~updateSpacetime.() });

    // Tags section
    StaticText(window, Rect(margin, margin + 155, windowWidth - (2*margin), 20))
        .string_("TAG MANAGEMENT")
        .font_(Font.default.size_(14).bold_(true));

    StaticText(window, Rect(margin, margin + 180, 300, 20))
        .string_("Available Tags: (double-click to add)")
        .align_(\left)
        .font_(Font.default.size_(12));
        
    StaticText(window, Rect(margin + 420, margin + 180, 300, 20))
        .string_("Active Tags: (double-click to remove)")
        .align_(\left)
        .font_(Font.default.size_(12));
        
    // Add a note about biome/era tags
    StaticText(window, Rect(margin + 420, margin + 205 + 155, 300, 20))
        .string_("Note: Current biome and era tags cannot be removed")
        .align_(\center)
        .font_(Font.default.size_(10).italic_(true));

    tagsList = ListView(window, Rect(margin, margin + 205, 300, 150))
        .background_(Color.white)
        .mouseDownAction_({ |view, x, y, modifiers, buttonNumber, clickCount|
            if(clickCount == 2) {  // Double-click
                var selectedIndex = view.value;
                if(selectedIndex.notNil and: { configs[\allTags].notNil }) {
                    var displayedTag = view.items[selectedIndex];
                    var actualTag = configs[\allTags][selectedIndex];
                    
                    activeTags.add(actualTag);
                    ~sendOSC.('/include', [actualTag]);
                    ~updateConnectionStatus.();
                    ~updateActiveTagsList.();
                };
            };
        });

    activeTagsList = ListView(window, Rect(margin + 420, margin + 205, 300, 150))
        .background_(Color.white.alpha_(0.9))
        .hiliteColor_(Color.blue(0.7, 0.3))
        .mouseDownAction_({ |view, x, y, modifiers, buttonNumber, clickCount|
            if(clickCount == 2) {  // Double-click
                var selectedIndex = view.value;
                if(selectedIndex.notNil and: { view.items.size > 0 }) {
                    var displayedTag = view.items[selectedIndex].asString;
                    var actualTag;
                    
                    // Extract the actual tag name by removing any prefixes
                    if(displayedTag.beginsWith("[BIOME] ")) {
                        actualTag = displayedTag.replace("[BIOME] ", "");
                    } {
                        if(displayedTag.beginsWith("[ERA] ")) {
                            actualTag = displayedTag.replace("[ERA] ", "");
                        } {
                            actualTag = displayedTag;
                        };
                    };
                    actualTag = actualTag.asSymbol; // Ensure it's a symbol
                    
                    // Don't remove biome or era on double-click
                    if((actualTag != currentBiome) && (actualTag != currentEra)) {
                        activeTags.remove(actualTag);
                        ~sendOSC.('/exclude', [actualTag]);
                        ~updateConnectionStatus.();
                        ~updateActiveTagsList.();
                    } {
                        "Cannot remove current biome or era tags".warn;
                    };
                };
            };
        });

    includeTagButton = Button(window, Rect(margin + 320, margin + 205, 80, buttonHeight))
        .string_("Include Tag")
        .background_(Color.green(0.8))
        .action_({
            var selectedIndex = tagsList.value;
            if(selectedIndex.notNil and: { configs[\allTags].notNil }) {
                var actualTag = configs[\allTags][selectedIndex];
                activeTags.add(actualTag);
                ~sendOSC.('/include', [actualTag]);
                ~updateConnectionStatus.();
                ~updateActiveTagsList.();
            };
        });

    excludeTagButton = Button(window, Rect(margin + 320, margin + 240, 80, buttonHeight))
        .string_("Exclude Tag")
        .background_(Color.red(0.8))
        .action_({
            var selectedIndex = tagsList.value;
            if(selectedIndex.notNil and: { configs[\allTags].notNil }) {
                var actualTag = configs[\allTags][selectedIndex];
                activeTags.remove(actualTag);
                ~sendOSC.('/exclude', [actualTag]);
                ~updateConnectionStatus.();
                ~updateActiveTagsList.();
            };
        });

    // State controls section
    StaticText(window, Rect(margin, margin + 370, windowWidth - (2*margin), 20))
        .string_("STATE CONTROLS")
        .font_(Font.default.size_(14).bold_(true));

    listeningButton = Button(window, Rect(margin, margin + 395, 100, buttonHeight))
        .states_([["Start Listening"], ["Stop Listening"]])
        .background_(Color.yellow(0.8))
        .action_({ |but|
		    var state;
            isListening = but.value == 1;
            state = if(isListening) { "start" } { "stop" };
            ~sendOSC.('/listening', [state]);
            ~updateConnectionStatus.();
        });

    speakingButton = Button(window, Rect(margin + 110, margin + 395, 100, buttonHeight))
        .states_([["Start Speaking"], ["Stop Speaking"]])
        .background_(Color.cyan(0.8))
        .action_({ |but|
		    var state;
            isSpeaking = but.value == 1;
            state = if(isSpeaking) { "start" } { "stop" };
            ~sendOSC.('/speaking', [state]);
            ~updateConnectionStatus.();
        });

    transitionButton = Button(window, Rect(margin + 220, margin + 395, 100, buttonHeight))
        .states_([["Start Transition"], ["Stop Transition"]])
        .background_(Color.magenta(0.8))
        .action_({ |but|
            var state;
		    isTransitioning = but.value == 1;
            state = if(isTransitioning) { "start" } { "stop" };
            ~sendOSC.('/transition', [state]);
            ~updateConnectionStatus.();
        });

    // Triggers section
    StaticText(window, Rect(margin, margin + 440, windowWidth - (2*margin), 20))
        .string_("TRIGGERS")
        .font_(Font.default.size_(14).bold_(true));

    StaticText(window, Rect(margin, margin + 465, 80, 20)).string_("Trigger:");
    triggerMenu = PopUpMenu(window, Rect(margin + 85, margin + 465, 150, 25));

    playTriggerButton = Button(window, Rect(margin + 250, margin + 465, 100, buttonHeight))
        .string_("Play Trigger")
        .background_(Color(1, 0.6, 0.2)) // orange-ish color
        .action_({
            var selectedIndex = triggerMenu.value;
            if(selectedIndex.notNil and: { configs[\allTriggers].notNil }) {
                var trigger = configs[\allTriggers][selectedIndex];
                ~sendOSC.('/trigger', [trigger]);
            };
        });

    // Volume controls section
    StaticText(window, Rect(margin, margin + 510, windowWidth - (2*margin), 20))
        .string_("VOLUME CONTROLS")
        .font_(Font.default.size_(14).bold_(true));

    StaticText(window, Rect(margin, volumeY, 80, 20)).string_("Master:");
    masterVolumeSlider = Slider(window, Rect(margin + 85, volumeY, sliderWidth, sliderHeight))
        .value_(1.0);
    masterVolumeLabel = StaticText(window, Rect(margin + 290, volumeY, 50, 20)).string_("1.00");
    masterVolumeSlider.action = { |slider|
        ~sendOSC.('/volume/master', [slider.value]);
        masterVolumeLabel.string = slider.value.round(0.01).asString;
    };

    volumeY = volumeY + 30;
    StaticText(window, Rect(margin, volumeY, 80, 20)).string_("Environment:");
    envVolumeSlider = Slider(window, Rect(margin + 85, volumeY, sliderWidth, sliderHeight))
        .value_(1.0);
    envVolumeLabel = StaticText(window, Rect(margin + 290, volumeY, 50, 20)).string_("1.00");
    envVolumeSlider.action = { |slider|
        ~sendOSC.('/volume/environment', [slider.value]);
        envVolumeLabel.string = slider.value.round(0.01).asString;
    };

    volumeY = volumeY + 30;
    StaticText(window, Rect(margin, volumeY, 80, 20)).string_("Music:");
    musicVolumeSlider = Slider(window, Rect(margin + 85, volumeY, sliderWidth, sliderHeight))
        .value_(1.0);
    musicVolumeLabel = StaticText(window, Rect(margin + 290, volumeY, 50, 20)).string_("1.00");
    musicVolumeSlider.action = { |slider|
        ~sendOSC.('/volume/music', [slider.value]);
        musicVolumeLabel.string = slider.value.round(0.01).asString;
    };

    // SFX slider
    volumeY = volumeY + 30;
    StaticText(window, Rect(margin, volumeY, 80, 20)).string_("SFX:");
    sfxVolumeSlider = Slider(window, Rect(margin + 85, volumeY, sliderWidth, sliderHeight))
        .value_(1.0);
    sfxVolumeLabel = StaticText(window, Rect(margin + 290, volumeY, 50, 20)).string_("1.00");
    sfxVolumeSlider.action = { |slider|
        ~sendOSC.('/volume/sfx', [slider.value]);
        sfxVolumeLabel.string = slider.value.round(0.01).asString;
    };

    // Move presets below all sliders
    volumePresetY = volumeY + 40;
    StaticText(window, Rect(margin, volumePresetY, 150, 20))
        .string_("Volume Presets:")
        .font_(Font.default.size_(12).bold_(true));

    volumePresetY = volumePresetY + 25;
    Button(window, Rect(margin, volumePresetY, 80, buttonHeight))
        .string_("Silent")
        .action_({
            [masterVolumeSlider, envVolumeSlider, musicVolumeSlider, sfxVolumeSlider].do { |slider|
                slider.value = 0.0;
                slider.doAction;
            };
        });
    Button(window, Rect(margin + 85, volumePresetY, 80, buttonHeight))
        .string_("Quiet")
        .action_({
            // Default values for "Quiet" preset
            masterVolumeSlider.value = 0.5; masterVolumeSlider.doAction;
            envVolumeSlider.value = 0.25; envVolumeSlider.doAction;
            musicVolumeSlider.value = 0.6; musicVolumeSlider.doAction;
            sfxVolumeSlider.value = 0.6; sfxVolumeSlider.doAction;
        });
    Button(window, Rect(margin + 170, volumePresetY, 80, buttonHeight))
        .string_("Normal")
        .action_({
            // Default values for "Normal" preset
            masterVolumeSlider.value = 0.8; masterVolumeSlider.doAction;
            envVolumeSlider.value = 0.25; envVolumeSlider.doAction;
            musicVolumeSlider.value = 0.8; musicVolumeSlider.doAction;
            sfxVolumeSlider.value = 0.8; sfxVolumeSlider.doAction;
        });
    Button(window, Rect(margin + 255, volumePresetY, 100, buttonHeight)) // wider for last button
        .string_("Music Only")
        .action_({
            // Default values for "Music Only" preset
            masterVolumeSlider.value = 0.8; masterVolumeSlider.doAction;
            envVolumeSlider.value = 0.0; envVolumeSlider.doAction;
            musicVolumeSlider.value = 0.8; musicVolumeSlider.doAction;
            sfxVolumeSlider.value = 0.0; sfxVolumeSlider.doAction;
        });

    // Test buttons section
    StaticText(window, Rect(margin + 400, margin + 370, 350, 20))
        .string_("QUICK TESTS")
        .font_(Font.default.size_(14).bold_(true));

    // Widen right column buttons
    testButtonWidth = 200;
    testX = margin + 400;

    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Test Forest/Wild")
        .action_({
            currentBiome = "temperate_forest";
            currentEra = "wilderness";
            biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
            eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
            ~updateSpacetime.();
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Test Desert/Modern")
        .action_({
            currentBiome = "desert";
            currentEra = "modern";
            biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
            eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
            ~updateSpacetime.();
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Test Ocean/Future")
        .action_({
            currentBiome = "tropical_island";
            currentEra = "future";
            biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
            eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
            ~updateSpacetime.();
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Test Mountain/Ruins")
        .action_({
            currentBiome = "mountain";
            currentEra = "ruins";
            biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
            eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
            ~updateSpacetime.();
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Test Arctic/Dystopia")
        .action_({
            currentBiome = "arctic";
            currentEra = "dystopia";
            biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
            eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
            ~updateSpacetime.();
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Random Test")
        .action_({
            if(configs[\allBiomes].size > 0 and: { configs[\allEras].size > 0 }) {
                currentBiome = configs[\allBiomes].choose;
                currentEra = configs[\allEras].choose;
                biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
                eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
                ~updateSpacetime.();
            };
        });

    // Add Musical Journey button
    testY = testY + 45;
    StaticText(window, Rect(testX, testY, testButtonWidth + 50, 20))
        .string_("MUSICAL JOURNEYS")
        .font_(Font.default.size_(12).bold_(true));

    testY = testY + 25;
    Button(window, Rect(testX, testY, testButtonWidth + 30, buttonHeight))
        .string_("Era Journey (Wild→Future)")
        .action_({
            var eras = ["wilderness", "pre_industrial", "early_industrial", "late_industrial", "modern", "current", "future"];
            var currentBiomeForJourney = currentBiome; // Keep current biome
            var eraIndex = 0;

            // Start the journey
            {
                eras.do { |era, i|
                    {
                        currentEra = era;
                        eraMenu.value = configs[\allEras].indexOf(currentEra) ? 0;
                        ~updateSpacetime.();
                        ("Era Journey: % -> %".format(currentBiomeForJourney, era)).postln;
                    }.defer(i * 4); // 4 second intervals
                };
            }.fork;
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth + 30, buttonHeight))
        .string_("Biome Journey (Around World)")
        .action_({
            var biomes = ["temperate_forest", "desert", "tropical_island", "arctic", "mountain", "temperate_forest"];
            var currentEraForJourney = currentEra; // Keep current era

            // Start the journey
            {
                biomes.do { |biome, i|
                    {
                        currentBiome = biome;
                        biomeMenu.value = configs[\allBiomes].indexOf(currentBiome) ? 0;
                        ~updateSpacetime.();
                        ("Biome Journey: % -> %".format(biome, currentEraForJourney)).postln;
                    }.defer(i * 3); // 3 second intervals
                };
            }.fork;
        });

    // Add system controls
    testY = testY + 45;
    StaticText(window, Rect(testX, testY, testButtonWidth + 50, 20))
        .string_("SYSTEM CONTROLS")
        .font_(Font.default.size_(12).bold_(true));

    testY = testY + 25;
    reloadButton = Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Reload Audio Config")
        .background_(Color.yellow(0.6))
        .action_({
            var result;
            ~sendOSC.('/reload', []);
            // Reload our config too
            result = ~loadAllConfigs.();
            if(result.notNil) {
                configs = result;  // Properly update the global configs
                "Configs updated: %".format(configs.class).postln;
                ~updateMenus.();
                "Configuration reloaded".postln;
            } {
                "Failed to load configs".error;
            };
        });

    testY = testY + 35;
    Button(window, Rect(testX, testY, testButtonWidth, buttonHeight))
        .string_("Emergency Stop")
        .background_(Color.red(0.8))
        .action_({
            // Stop all agent states
            listeningButton.value = 0; listeningButton.doAction;
            speakingButton.value = 0; speakingButton.doAction;
            transitionButton.value = 0; transitionButton.doAction;

            // Set volumes to safe levels
            masterVolumeSlider.value = 0.5; masterVolumeSlider.doAction;
            envVolumeSlider.value = 0.5; envVolumeSlider.doAction;
            musicVolumeSlider.value = 0.3; musicVolumeSlider.doAction;
            sfxVolumeSlider.value = 0.3; sfxVolumeSlider.doAction;

            "Emergency stop activated - volumes reduced, states cleared".postln;
        });

    window.onClose = {
        "Audio GUI closed".postln;
        if(isConnected) {
            // Send stop messages on close
            ~sendOSC.('/listening', ["stop"]);
            ~sendOSC.('/speaking', ["stop"]);
            ~sendOSC.('/transition', ["stop"]);
        };
    };

    ~updateConnectionStatus.();
};

// Update menus with config data
~updateMenus = {
    if(configs.isNil) {
        "Warning: configs is nil, cannot update menus".warn;
        ^this;
    };

    if(configs.class != Dictionary) {
        "Error: configs is not a Dictionary (actual type: %)".format(configs.class).error;
        ^this;
    };

    "Updating menus with config data...".postln;
    "Available config keys: %".format(configs.keys).postln;
    
    if(biomeMenu.notNil and: { configs[\allBiomes].notNil }) {
        "Setting % biomes in dropdown".format(configs[\allBiomes].size).postln;
        biomeMenu.items = configs[\allBiomes];
        biomeMenu.value = configs[\allBiomes].indexOf(currentBiome.asSymbol) ? 0;
    } {
        "Cannot update biome menu: menu=%, allBiomes=%".format(
            if(biomeMenu.isNil) {"nil"} {"valid"}, 
            if(configs[\allBiomes].isNil) {"nil"} {configs[\allBiomes].size}
        ).warn;
    };

    if(eraMenu.notNil and: { configs[\allEras].notNil }) {
        "Setting % eras in dropdown".format(configs[\allEras].size).postln;
        eraMenu.items = configs[\allEras];
        eraMenu.value = configs[\allEras].indexOf(currentEra.asSymbol) ? 0;
    } {
        "Cannot update era menu: menu=%, allEras=%".format(
            if(eraMenu.isNil) {"nil"} {"valid"}, 
            if(configs[\allEras].isNil) {"nil"} {configs[\allEras].size}
        ).warn;
    };

    if(tagsList.notNil and: { configs[\allTags].notNil }) {
        var allTags = configs[\allTags].collect { |tag|
            if(activeTags.includes(tag)) {
                "✓ " ++ tag;
            } {
                "   " ++ tag;
            };
        };
        "Setting % tags in list".format(configs[\allTags].size).postln;
        tagsList.items = allTags;
    } {
        "Cannot update tags list: list=%, allTags=%".format(
            if(tagsList.isNil) {"nil"} {"valid"}, 
            if(configs[\allTags].isNil) {"nil"} {configs[\allTags].size}
        ).warn;
    };

    if(triggerMenu.notNil and: { configs[\allTriggers].notNil }) {
        "Setting % triggers in dropdown".format(configs[\allTriggers].size).postln;
        triggerMenu.items = configs[\allTriggers];
    } {
        "Cannot update trigger menu: menu=%, allTriggers=%".format(
            if(triggerMenu.isNil) {"nil"} {"valid"}, 
            if(configs[\allTriggers].isNil) {"nil"} {configs[\allTriggers].size}
        ).warn;
    };

    // Update the active tags list
    ~updateActiveTagsList.();

    "Menu update complete".postln;
};


// Initialize everything
{
    "Loading Experimance Audio GUI...".postln;

    // Load configurations using our local utility function
    configs = ~loadAllConfigs.();

    // Create the GUI
    ~createGUI.();

    // Update menus with loaded data
    ~updateMenus.();

    // Auto-connect
    ~initOSC.();
    ~updateConnectionStatus.();

    "Audio GUI ready! Window should be visible.".postln;
    "".postln;
    "USAGE:".postln;
    "1. Select biome and era, then click 'Update Spacetime'".postln;
    "2. Include/exclude additional tags as needed".postln;
    "3. Use state controls for listening, speaking, transitions".postln;
    "4. Adjust volume controls as needed".postln;
    "5. Use quick test buttons for common scenarios".postln;
    "".postln;
    "Make sure experimance_audio.scd is running to receive OSC messages!".postln;
}.defer(0.1);
