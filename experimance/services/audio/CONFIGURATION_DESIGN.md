# Experimance Audio System Configuration Design

## Overview

The Experimance audio system relies on configuration files in both general project config and audio-specific schema. This document explains the design decisions and implementation details.

## Configuration Architecture

### Files

1. **General Project Configuration**
   - Located at `data/experimance_config.json`
   - Contains common configuration used across the entire Experimance project
   - Defines biomes, eras, common tags, and trigger types

2. **Audio-Specific Configuration**
   - Located at `services/audio/config/audio_schema.json`
   - Contains audio-specific configuration
   - Defines audio-specific tags, volume controls, and sound categories

3. **Sound Layer Configurations**
   - Located in `services/audio/config/`
   - Files like `layers.json`, `triggers.json`, and `music_loops.json`
   - Define the specific sound files and their properties for playback

## Implementation Notes

### SuperCollider Implementation

After experimentation with different approaches, we found that SuperCollider's class system has limitations when working with utility classes. We've decided to use a direct copy of utility functions in each script rather than using a shared class like AudioUtils.

This decision was made because:
1. SuperCollider's class system does not handle utility functions well across scripts
2. The return value semantics for class methods are inconsistent
3. Direct function copies are more reliable in this context

### Configuration Loading

The configuration loading process follows these steps:

1. Load the general project configuration from `data/experimance_config.json`
2. Load the audio-specific schema from `services/audio/config/audio_schema.json`
3. Extract common data like biomes, eras, and tags from both files
4. Validate that audio tags are consistent with general tags

### Environment-Specific Data

The system supports environment-specific variations in sound, organized by:

1. **Biome** - Geographic/environmental context (forest, desert, etc.)
2. **Era** - Temporal context (wilderness, industrial, modern, etc.)
3. **Tags** - Sound characteristics and categories

## Validation

Schema validation is performed by the `scripts/validate_schemas.py` script, which ensures:
1. Python enums match JSON data definitions
2. All required configuration fields are present
3. Audio-specific tags are consistent with the general tag system

## Scripts

- `experimance_audio.scd` - Main audio processor script
- `experimance_audio_gui.scd` - GUI control panel for testing
- `test_direct_config_loading.scd` - Test script for configuration loading
