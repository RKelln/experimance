# Experimance Audio SuperCollider Scripts

This folder contains the SuperCollider scripts that power the audio system for the Experimance installation.

## Main Script: `experimance_audio.scd`

The main script (`experimance_audio.scd`) handles all audio playback, including environmental sounds, music, and transition effects. It receives OSC messages from the Python service and manages the audio output accordingly.

## Placeholder Music Generation System

The script includes a sophisticated placeholder music generation system that automatically creates appropriate background music when audio files are missing. This ensures continuous audio even during development or when specific audio assets haven't been created yet.

### Key Features

#### 1. Era-specific Musical Keys

Each era has its unique musical key and scale following a progression from natural to electronic and back:

- **Wilderness**: C Pentatonic (natural, consonant sounds)
- **Pre-industrial**: F Mixolydian (early civilization, still natural)
- **Early Industrial**: G Major (structured, traditional harmony)
- **Late Industrial**: D Dorian (increasing complexity)
- **Modern**: A Harmonic Minor (more tension and complexity)
- **Current**: E Diminished (dissonance and uncertainty)
- **Future**: B Chromatic (maximum complexity, all tones)
- **Dystopia**: F# Enigmatic (peak dissonance and tension)
- **Ruins**: C Pentatonic (return to nature sounds)

The progression follows a circle-of-fifths inspired pattern while increasing musical complexity, then returning to the original simplicity for the "ruins" era.

#### 2. Era-specific Synthesizer Timbres

Eight distinct synthesizer definitions create unique sonic identities for each era:

- **placeholderSine**: Pure tones with subtle harmonics for wilderness/pre-industrial
- **placeholderSaw**: Filtered saw waves for early/late industrial periods
- **placeholderSquare**: Square waves with PWM for modern era
- **placeholderPad**: Detuned pad sounds with chorus for current era
- **placeholderBell**: Bell-like FM sounds for future era
- **placeholderNoise**: Distorted, harsh sounds for dystopian era
- **placeholderFM**: Complex FM synthesis for ruins era (nature with complexity)
- **placeholderDefault**: Simple fallback synthesizer

The timbres progress from pure and natural to mechanical, then electronic, distorted, and finally back to a more complex natural sound.

#### 3. Layered Musical Structure

Each era's music consists of three layered patterns with different rhythmic intervals:

- **Layer 0**: Root note drones (slowest, 8-beat cycle)
  - Different drone patterns per era (stable for natural eras, more movement for modern/future)
  - Longest note durations provide foundation

- **Layer 1**: Arpeggiated chords (medium, 4-beat cycle)
  - Era-appropriate chord types (open fifths → traditional triads → complex chords → dissonant structures)
  - Varied arpeggiation patterns matching era character

- **Layer 2**: Melodic patterns (fastest, 2-beat cycle)
  - Natural flowing patterns for wilderness/pre-industrial
  - Structured mechanical patterns for industrial eras
  - Complex, unpredictable patterns for future/dystopian eras

#### 4. Smooth Transitions

- **Graceful Fading**: Custom fade methods ensure smooth amplitude transitions
- **Era Crossfading**: Musical transitions between eras with glissando effects
- **Automatic Fallback**: System detects missing files and generates appropriate sounds

## Usage

This script is automatically loaded by the Experimance Audio Service when it starts. It can also be opened directly in SuperCollider for development and testing.

The script responds to OSC messages sent to port 57120 (SuperCollider's default port):

- `/spacetime <biome> <era>`: Set the current biome and era
- `/include <tag>`: Include an audio tag
- `/exclude <tag>`: Exclude an audio tag
- `/listening <start|stop>`: Signal when the agent is listening
- `/speaking <start|stop>`: Signal when the agent is speaking
- `/transition <start|stop>`: Signal a transition effect
- `/reload`: Reload audio configurations

## Development Notes

- All synthesizer definitions are at the end of the script
- Audio configurations are loaded from JSON files in the `../config` directory
- The placeholder system creates music programmatically when audio files are missing
- The system automatically ducks environmental sound during agent interactions

To test the placeholder system, simply reference non-existent audio files in the configuration, and the system will generate appropriate music for each era.


## Testing

To test OSC functionality is working correctly try:
```
sclang services/audio/sc_scripts/test_osc.scd
```
