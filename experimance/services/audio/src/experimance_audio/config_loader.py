"""
Configuration handler and loader for the Experimance Audio Service.

This module handles loading and parsing JSON configuration files for:
- Environmental audio layers
- Sound effect triggers
- Era-based music loops

It provides a unified interface to access audio configurations with hot-reload capability.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_DIR = "config"

class AudioConfigLoader:
    """Loads and manages audio configuration files."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize the audio configuration loader.
        
        Args:
            config_dir: Directory containing audio config files. If None, uses default.
        """
        self.config_dir = config_dir or DEFAULT_CONFIG_DIR
        self.layers = []  # Environmental audio layers
        self.triggers = []  # Sound effect triggers
        self.music_loops = {}  # Era-based music loops
        
        # Flag to track if configs have been loaded
        self.is_loaded = False
    
    def load_configs(self) -> bool:
        """Load all audio configuration files.
        
        Returns:
            bool: True if loading was successful, False otherwise
        """
        success = True
        
        try:
            # Load environmental layers
            layers_path = os.path.join(self.config_dir, "layers.json")
            if os.path.exists(layers_path):
                with open(layers_path, 'r') as f:
                    self.layers = json.load(f)
                logger.info(f"Loaded {len(self.layers)} audio layers")
            else:
                logger.warning(f"Layers config file not found: {layers_path}")
                success = False
                
            # Load sound effect triggers
            triggers_path = os.path.join(self.config_dir, "triggers.json")
            if os.path.exists(triggers_path):
                with open(triggers_path, 'r') as f:
                    self.triggers = json.load(f)
                logger.info(f"Loaded {len(self.triggers)} audio triggers")
            else:
                logger.warning(f"Triggers config file not found: {triggers_path}")
                success = False
                
            # Load music loops
            music_loops_path = os.path.join(self.config_dir, "music_loops.json")
            if os.path.exists(music_loops_path):
                with open(music_loops_path, 'r') as f:
                    music_loops_data = json.load(f)
                    self.music_loops = music_loops_data.get("era_loops", {})
                logger.info(f"Loaded music loops for {len(self.music_loops)} eras")
            else:
                logger.warning(f"Music loops config file not found: {music_loops_path}")
                success = False
                
            self.is_loaded = success
            return success
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing audio config file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading audio configs: {e}")
            return False
    
    def get_layers_for_context(self, biome: str, era: str) -> List[Dict[str, Any]]:
        """Get audio layers appropriate for the current context.
        
        Args:
            biome: Current biome name
            era: Current era name
            
        Returns:
            List of audio layer configurations matching the context
        """
        if not self.is_loaded:
            self.load_configs()
            
        # Filter layers by biome and era tags
        matching_layers = []
        for layer in self.layers:
            tags = layer.get("tags", [])
            if biome in tags or era in tags:
                matching_layers.append(layer)
                
        return matching_layers
    
    def get_trigger_by_name(self, trigger_name: str) -> Optional[Dict[str, Any]]:
        """Get a sound effect trigger by name.
        
        Args:
            trigger_name: Name of the trigger to find
            
        Returns:
            Trigger configuration if found, None otherwise
        """
        if not self.is_loaded:
            self.load_configs()
            
        for trigger in self.triggers:
            if trigger.get("trigger") == trigger_name:
                return trigger
                
        return None
    
    def get_music_loops_for_era(self, era: str) -> List[Dict[str, Any]]:
        """Get music loops for the specified era.
        
        Args:
            era: Era name
            
        Returns:
            List of music loops for the era
        """
        if not self.is_loaded:
            self.load_configs()
            
        return self.music_loops.get(era, [])
