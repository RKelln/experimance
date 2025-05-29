# Experimance Audio GUI - Musician's Guide

This GUI interface allows you to easily test and control the Experimance audio system without needing to understand the underlying OSC messaging protocol.

## Quick Start

1. **Start SuperCollider** and open the GUI:
   ```supercollider
   // In SuperCollider, evaluate this line:
   "services/audio/sc_scripts/experimance_audio_gui.scd".loadPath;
   ```

2. **For testing without the main audio system**, you can use:
   ```supercollider
   // This creates a simple OSC listener to see what messages are sent:
   "services/audio/sc_scripts/test_gui.scd".loadPath;
   ```

## Interface Sections

### 🌍 SPACETIME CONTEXT
Controls the main environmental context for the audio system.

- **Biome**: Select the environmental setting (forest, desert, arctic, etc.)
- **Era**: Select the time period (wilderness, modern, future, etc.)
- **Update Spacetime**: Sends the biome+era combination to the audio system

**Quick Presets**: Fast buttons for common combinations:
- Forest+Wild: Natural forest in wilderness era
- Desert+Modern: Desert environment in modern times
- Ocean+Future: Tropical island in future era

### 🏷️ TAG MANAGEMENT
Tags add additional audio layers beyond the main biome/era combination.

- **Available Tags**: List of all possible audio tags (birds, water, machinery, etc.)
- **Include Tag**: Add the selected tag to active audio layers
- **Exclude Tag**: Remove the selected tag from active audio layers

### 🤖 STATE CONTROLS
Controls the agent's behavioral state, which affects audio ducking and filtering.

- **Start/Stop Listening**: Agent is actively listening (may duck music)
- **Start/Stop Speaking**: Agent is speaking (may duck environmental sounds)
- **Start/Stop Transition**: Scene is transitioning (may add transition effects)

### 🔊 VOLUME CONTROLS
Independent volume controls for different audio layers.

- **Master**: Overall system volume
- **Environment**: Environmental ambient sounds (birds, wind, water, etc.)
- **Music**: Musical loops and generated music
- **SFX**: Sound effects and triggers

**Volume Presets**:
- **Silent**: All volumes to 0
- **Quiet**: Reduced volumes for testing
- **Normal**: All volumes to 100%
- **Music Only**: Focus on music, reduce environment and effects

### 🔧 TRIGGERS
One-shot sound effects for specific events.

- Select a trigger type from the dropdown
- Click "Play Trigger" to fire the sound effect

### 🎭 QUICK TESTS
Pre-configured scenarios for easy testing.

**Single Tests**:
- Various biome/era combinations
- Random Test: Selects random biome and era

**Musical Journeys** (great for testing transitions):
- **Era Journey**: Progresses through time periods while keeping the same biome
- **Biome Journey**: Travels around the world while keeping the same era

**System Controls**:
- **Reload Audio Config**: Reloads all audio configuration files
- **Emergency Stop**: Resets all states and reduces volumes to safe levels

## Creative Usage Tips

### For Composition Testing
1. Start with "Music Only" volume preset to focus on the generated music
2. Use "Era Journey" to hear how music evolves through time periods
3. Try different biomes with the same era to hear environmental variations

### For Soundscape Design
1. Start with a biome/era combination you like
2. Add relevant tags (e.g., "birds", "water" for nature scenes)
3. Use state controls to test how agent interaction affects the mix
4. Adjust volume balances to find the right environmental blend

### For Testing Transitions
1. Use the Musical Journey buttons to test automated transitions
2. Try manual spacetime changes while audio is playing
3. Test state changes (listening/speaking) during different scenes

### For Performance
1. Use Quick Presets for rapid scene changes during live performance
2. Volume sliders allow real-time mixing
3. Emergency Stop provides safety during live shows

## Audio File Structure

The system automatically detects missing audio files and generates placeholder music:

- **Environmental sounds**: Located in `audio/environment/`
- **Music loops**: Located in `audio/music/`
- **Sound effects**: Located in `audio/sfx/`

When audio files are missing, the system creates:
- Era-appropriate synthesized music with proper scales and timbres
- Smooth transitions between eras
- Appropriate volume balancing

## Troubleshooting

### GUI doesn't appear
- Make sure SuperCollider is running
- Check that the file path is correct relative to your working directory
- Look for error messages in the SuperCollider console

### OSC messages not working
- Ensure `experimance_audio.scd` is running and listening on port 5568
- Use `test_gui.scd` to see if messages are being sent
- Check the status display at the bottom of the GUI window

### No audio changes
- Verify volume levels (check if Master volume is > 0)
- Ensure the audio system has loaded the configuration files
- Try the "Reload Audio Config" button

### Missing biomes/eras/tags in dropdowns
- Check that configuration files exist in `../config/` relative to the script
- Verify JSON files are properly formatted
- Look for loading errors in the SuperCollider console

## Configuration Files

The GUI reads from these JSON configuration files:

- `layers.json`: Environmental audio layers with tags
- `music_loops.json`: Era-based music loops
- `triggers.json`: Sound effect triggers
- `master_schema.json`: Complete list of biomes, eras, and tags

You can edit these files to add new audio content or modify the available options in the GUI dropdowns.

## Advanced Features

### Custom OSC Commands
If you need to send custom OSC messages, you can use the underlying utility functions:

```supercollider
// Send custom spacetime
~audioUtils.sendSpacetime.(NetAddr("127.0.0.1", 5568), "custom_biome", "custom_era");

// Send custom volume
~audioUtils.sendVolume.(NetAddr("127.0.0.1", 5568), "master", 0.75);
```

### Configuration Inspection
```supercollider
// See all loaded configurations
configs[\allBiomes].postln;  // All available biomes
configs[\allEras].postln;    // All available eras  
configs[\allTags].postln;    // All available tags
```

Happy composing! 🎵
